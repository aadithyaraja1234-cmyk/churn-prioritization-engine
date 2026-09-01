"""Stage 2b: async background-job infrastructure for tenant model
training. The job lifecycle (queued -> running -> succeeded/failed) was
proven correct in Stage 2a with simulated work; _run_training() now
replaces that placeholder with a real call into
src.models.tenant_training.prepare_and_train(), which itself calls the
EXISTING src.models.train.train_model() - the same trainer used for
Telco/Banking, no new training logic.

Uses a plain Python thread (threading.Thread, daemon, fire-and-forget),
not FastAPI's BackgroundTasks. A background thread outside the request
lifecycle needs its own DB session regardless (the request's Session
closes the instant the response is sent), and a real detached thread makes
"the endpoint returns before the job finishes" a direct, deterministically
testable property - FastAPI's BackgroundTasks are awaited as part of the
same request coroutine, which is invisible to a real HTTP client (the
response bytes are flushed to the socket first either way) but makes the
"returns immediately" behavior awkward to assert directly in a test built
on TestClient's synchronous, fully-drained request cycle. No new
infrastructure dependency (Celery/Redis/etc.) - deliberately out of scope
at this project's size.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.auth import CurrentUser, get_current_user
from database.db import SessionLocal as _DEFAULT_SESSION_LOCAL, get_db, get_tenant_scoped_query
from database.migrate_csv_to_db import migrate_customers, migrate_predictions
from database.models import Company, Customer, OnboardingUpload, Prediction, TrainingJob
from src.data.split import load_tenant_config
from src.models.tenant_training import TrainingDataError, TrainingOutcome, prepare_and_train

router = APIRouter(prefix="/api/training", tags=["Training"])
ROOT = Path(__file__).resolve().parent.parent

# Module-level, reassignable, so tests can point this at a test database's
# sessionmaker. The background thread runs outside FastAPI's request/DI
# lifecycle, so app.dependency_overrides[get_db] (which only intercepts
# Depends(get_db) at request time) can never reach it - this attribute is
# the only hook available for redirecting where the thread writes.
SessionLocal = _DEFAULT_SESSION_LOCAL


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingStartRequest(BaseModel):
    upload_id: int


class TrainingStartResponse(BaseModel):
    job_id: int
    status: str
    created_at: str


def _run_training(db: Session, tenant_id: str) -> dict[str, Any]:
    """Loads tenant_id's confirmed, validated upload (column_mapping_json
    is only ever set by POST /api/onboarding/validate) and hands it to
    prepare_and_train(), which calls the existing train_model(). Any
    failure - no validated upload found, un-trainable data
    (TrainingDataError), or a real crash inside train_model() - is left to
    propagate straight through to _run_training_job's except-block, so the
    job is marked failed with a real error_message and no artifact is left
    half-written.

    A completed-but-unsane classifier result (ROC-AUC outside
    prepare_and_train's sane bounds) is NOT an exception - it's a
    legitimate, successful run of the pipeline that happens to warrant
    human review, reported via is_sane/sanity_warning in the returned
    dict. survival/segments/anomalies/clv are gated independently of the
    classifier's own sanity verdict - see _apply_training_outcome for why
    Company.model_dir/data_path/feature_flags_json are touched whenever
    ANY module (classifier included) actually passed its own gate, not
    only when the classifier did."""
    upload = (
        db.query(OnboardingUpload)
        .filter(OnboardingUpload.tenant_id == tenant_id, OnboardingUpload.column_mapping_json.isnot(None))
        .order_by(OnboardingUpload.id.desc())
        .first()
    )
    if upload is None:
        raise TrainingDataError(
            "No confirmed, validated upload found for this tenant - complete "
            "POST /api/onboarding/validate before starting a training job."
        )

    # Read what's needed into locals, then commit (a no-op - nothing was
    # changed) to end the implicit read transaction before the
    # multi-second train_model() call below. Otherwise this session would
    # hold a transaction open across all of training, contending with the
    # concurrent HTTP polling thread for the same pooled SQLite connection
    # (this project's tests use a single shared StaticPool connection) -
    # every GET /status during that window fights this thread for the
    # same lock instead of just reading the last-committed row.
    csv_text = upload.csv_text
    column_mapping = upload.column_mapping_json
    db.commit()

    outcome = prepare_and_train(
        tenant_id=tenant_id,
        csv_text=csv_text,
        column_mapping=column_mapping,
    )
    return _apply_training_outcome(db, tenant_id, outcome)


def _apply_training_outcome(db: Session, tenant_id: str, outcome: TrainingOutcome) -> dict[str, Any]:
    """Each of the five flags below (business_impact_core + the four Stage
    2b modules - survival/segments/anomalies/clv) is set independently, on
    that module's OWN real gate result - a borderline classifier doesn't
    suppress a genuinely passing segmentation, and a passing classifier
    doesn't paper over a failing survival fit. model_dir/data_path are
    written whenever ANY flag ends up enabled (a self-registered tenant
    whose classifier missed its bar but whose anomaly detection genuinely
    passed still needs Company.model_dir set, or /api/anomalies has
    nothing to find) - Company stays completely untouched, same as before,
    only when NOTHING at all passed (see Company's docstring in
    database/models.py for why model_dir/data_path/feature_flags_json
    always move together).

    Stage 2c: priority_ranking/backtest/scenario_simulator/customer_timeline
    now flow through this exact same generic loop too - see
    src/models/tenant_training.py's module docstring and
    _run_optional_modules() for each one's own real sanity gate.
    scenario_simulator's module_result is "skipped" (not attempted) rather
    than "failed" for a tenant whose segments didn't pass first, since
    scenario.py has a hard structural dependency on segment_model.pkl (see
    that module's scenario_simulator_sanity_check() docstring) - the loop
    above already treats both the same way (feature_flags only gets a key
    when passed=True), so no special-casing is needed here. Stage 2d adds
    budget_optimizer to this same generic loop too, on its own real gate
    (src/models/budget_optimizer.py's budget_optimizer_sanity_check()).

    Also populates the customers/predictions tables for this tenant here -
    real self-registered tenants used to never get either populated at
    all (database/migrate_csv_to_db.py only ever ran once, by hand, for
    Telco/Banking at initial setup), so GET /customers - and every UI
    surface built on it (Survival Likelihood, What-If Simulator, Customer
    360's picker) - silently showed "No customers found" for every
    self-registered tenant regardless of what actually trained. Re-training
    the same tenant must not duplicate rows, so existing ones are cleared
    first - this makes the tables always reflect the MOST RECENT training
    run's real population, the same "re-training replaces, never appends"
    semantics model_dir's own artifacts already have on disk."""
    feature_flags: dict[str, bool] = {}
    if outcome.is_sane:
        feature_flags["business_impact_core"] = True
    for module_name, module_result in outcome.module_results.items():
        if module_result["passed"]:
            feature_flags[module_name] = True

    if feature_flags:
        company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
        if company is not None:
            company.model_dir = outcome.model_dir
            company.data_path = outcome.data_path
            company.feature_flags_json = feature_flags
            db.commit()

            tenant_config = load_tenant_config(ROOT / outcome.model_dir)
            # ORM-level delete (load-then-db.delete(), not Query.delete()'s
            # bulk SQL DELETE) on purpose: re-training the SAME tenant with a
            # bulk delete silently dropped rows on the second run - SQLite
            # reuses a plain (non-AUTOINCREMENT) INTEGER PRIMARY KEY's now-
            # free row ids for the fresh INSERTs below, and Query.delete()
            # doesn't remove the now-gone rows from THIS session's identity
            # map, so migrate_customers()/migrate_predictions() collide with
            # stale entries mid-flush (observed directly: a SAWarning per
            # collision, and a real, silent short row count). db.delete(obj)
            # keeps the identity map itself consistent as each object is
            # actually removed, so a reused row id never collides. job/company
            # (still referenced by the caller after this function returns)
            # are untouched either way - only Customer/Prediction rows are
            # ever loaded and deleted here.
            for row in db.query(Prediction).filter(Prediction.tenant_id == tenant_id).all():
                db.delete(row)
            for row in db.query(Customer).filter(Customer.tenant_id == tenant_id).all():
                db.delete(row)
            db.commit()
            migrate_customers(db, ROOT / outcome.data_path, tenant_id, tenant_config["id_column"])
            if "business_impact_core" in feature_flags:
                migrate_predictions(db, ROOT / outcome.model_dir, tenant_id, ROOT / outcome.data_path, tenant_config)

    target_positive_value = outcome.inferred_target_positive_value
    if hasattr(target_positive_value, "item"):  # e.g. numpy.int64 from pandas - JSON can't serialize it directly
        target_positive_value = target_positive_value.item()

    result_metadata: dict[str, Any] = {
        "roc_auc": outcome.roc_auc,
        "pr_auc": outcome.pr_auc,
        "n_train": outcome.n_train,
        "n_test": outcome.n_test,
        "churn_rate_train": outcome.churn_rate_train,
        "churn_rate_test": outcome.churn_rate_test,
        "model_dir": outcome.model_dir,
        "data_path": outcome.data_path,
        "inferred_target_positive_value": target_positive_value,
        "is_sane": outcome.is_sane,
        "features_auto_enabled": outcome.is_sane,
        # Hyperparameter search results (src/models/train.py) - two
        # distinct numbers, never conflated: cv_score_during_tuning is the
        # training-split-only CV estimate used to pick hyperparameters;
        # final_test_roc_auc (== roc_auc above) is the real, once-computed
        # held-out score.
        "cv_score_during_tuning": outcome.cv_score_during_tuning,
        "final_test_roc_auc": outcome.final_test_roc_auc,
        "best_hyperparameters": outcome.best_hyperparameters,
        "tuning_seconds": outcome.tuning_seconds,
        **({"tuning_skipped_reason": outcome.tuning_skipped_reason} if outcome.tuning_skipped_reason else {}),
        # Real result + gate verdict for each of the four Stage 2b modules,
        # whether or not it passed - see TrainingOutcome.module_results'
        # docstring (src/models/tenant_training.py) for the exact shape.
        "modules": outcome.module_results,
    }
    if outcome.sanity_warning:
        result_metadata["sanity_warning"] = outcome.sanity_warning
    return result_metadata


def _run_training_job(job_id: int, tenant_id: str) -> None:
    """Runs in a detached background thread - opens its OWN db session (the
    request's session is closed the instant the HTTP response is sent) and
    is solely responsible for this job row's status transitions from here
    on. tenant_id is re-checked on every lookup even though job_id alone is
    already unique, as defense in depth against a future bug elsewhere ever
    handing this function a mismatched pair."""
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id, TrainingJob.tenant_id == tenant_id).first()
        if job is None:
            return  # shouldn't happen - the row is committed before this thread is started

        job.status = "running"
        job.started_at = _utcnow()
        db.commit()

        try:
            result_metadata = _run_training(db, tenant_id)
        except Exception as exc:
            job.status = "failed"
            job.completed_at = _utcnow()
            job.error_message = str(exc)
            db.commit()
            return

        job.status = "succeeded"
        job.completed_at = _utcnow()
        job.result_metadata_json = result_metadata
        db.commit()
    finally:
        db.close()


@router.post(
    "/start",
    response_model=TrainingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a background training job for a previously validated upload",
    description="Creates a training_jobs row (status=queued) for `upload_id` - which must "
    "belong to your tenant - and immediately starts the job in a background thread, returning "
    "its id right away without waiting for it to finish. Intended to be called with an upload "
    "reference that has already been through POST /api/onboarding/validate, though this "
    "endpoint itself only checks tenant ownership of upload_id, not your onboarding-status step "
    "- see GET /api/training/status/{job_id} to poll for completion. Runs real training "
    "(src.models.train.train_model()) against your tenant's most recently validated upload. "
    "Returns 404 if upload_id doesn't belong to your tenant.",
)
def start_training(
    request: TrainingStartRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TrainingStartResponse:
    upload = (
        get_tenant_scoped_query(db, OnboardingUpload, tenant_id=current_user.tenant_id)
        .filter(OnboardingUpload.id == request.upload_id)
        .first()
    )
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")

    job = TrainingJob(tenant_id=current_user.tenant_id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)

    # Daemon + fire-and-forget (never .join()-ed): the endpoint returns as
    # soon as the thread is started, not when it finishes - see this
    # module's docstring for why a raw thread makes that guarantee direct.
    threading.Thread(target=_run_training_job, args=(job.id, current_user.tenant_id), daemon=True).start()

    return TrainingStartResponse(job_id=job.id, status=job.status, created_at=job.created_at.isoformat())


@router.get(
    "/status/{job_id}",
    summary="Get a training job's current status",
    description="Returns a training job's status (queued/running/succeeded/failed) and "
    "timestamps, tenant-scoped - a tenant can only ever see their own jobs, never another "
    "tenant's, even by guessing a valid id. Meant to be polled periodically while status is "
    "queued or running. error_message is populated only once status is failed; "
    "result_metadata_json only once status is succeeded. Returns 404 if job_id doesn't belong "
    "to your tenant.",
)
def training_status(
    job_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    job = (
        get_tenant_scoped_query(db, TrainingJob, tenant_id=current_user.tenant_id)
        .filter(TrainingJob.id == job_id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")

    return {
        "job_id": job.id,
        "tenant_id": job.tenant_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message,
        "result_metadata_json": job.result_metadata_json,
    }
