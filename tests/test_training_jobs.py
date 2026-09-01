"""Async training-job infrastructure (queued -> running -> succeeded/failed,
tenant isolation) plus Stage 2b's real training wiring: a training job now
calls the actual src.models.train.train_model() against a tenant's
confirmed, validated upload, and only auto-enables feature_flags for a
sane (non-degenerate, non-leaky) result."""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.training as training_module
from api.main import app
from database.db import get_db
from database.models import Base, Company

_ORIGINAL_SESSION_LOCAL = training_module.SessionLocal

ROOT = Path(__file__).resolve().parents[1]
MERIDIAN_SAMPLE_CSV = ROOT / "data" / "test_onboarding_samples" / "company_meridian_wireless.csv"

THIN_CSV = (
    "id,label,tenure\n"
    "1,No,5\n"
    "2,Yes,10\n"
    "3,No,15\n"
)


def _make_noisy_csv(n_rows: int, seed: int) -> str:
    """A structurally valid but genuinely uninformative dataset: the
    feature column carries no real relationship to the target, so a
    correctly-behaving trainer should land near-random (ROC-AUC close to
    0.5), not crash. Balanced classes and a deterministic seed keep this
    reproducible - no flakiness from an unlucky split."""
    rng = random.Random(seed)
    lines = ["customer_id,target,revenue,noise_feature"]
    for i in range(n_rows):
        target = "Yes" if i % 2 == 0 else "No"
        revenue = round(rng.uniform(20, 120), 2)
        noise_feature = round(rng.uniform(0, 100), 2)
        lines.append(f"C{i},{target},{revenue},{noise_feature}")
    return "\n".join(lines) + "\n"


def _make_signal_csv(n_rows: int, seed: int) -> str:
    """A small but genuinely trainable dataset: a feature that's actually
    correlated with the target (with enough overlap/noise to stay well
    short of a perfect separator), so training should land a sane,
    middling ROC-AUC rather than a crash or a leakage-flagged near-1.0."""
    rng = random.Random(seed)
    lines = ["customer_id,target,revenue,tenure_like_feature"]
    for i in range(n_rows):
        churned = i % 2 == 0
        target = "Yes" if churned else "No"
        revenue = round(rng.uniform(20, 120), 2)
        # Churned customers skew toward a lower value; substantial overlap
        # between the two distributions keeps this from being a near-
        # perfect separator.
        base = rng.gauss(20, 15) if churned else rng.gauss(45, 15)
        feature = round(max(0.0, base), 2)
        lines.append(f"C{i},{target},{revenue},{feature}")
    return "\n".join(lines) + "\n"


def _cleanup_tenant_artifacts(tenant_id: str) -> None:
    shutil.rmtree(ROOT / "models" / tenant_id, ignore_errors=True)
    (ROOT / "data" / "tenant_uploads" / f"{tenant_id}.csv").unlink(missing_ok=True)


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    # The background thread runs outside FastAPI's request/DI lifecycle, so
    # the dependency_overrides above never reaches it - it must be pointed
    # at this SAME in-memory database directly, or it would silently write
    # to the real app.db instead.
    training_module.SessionLocal = TestingSessionLocal

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    training_module.SessionLocal = _ORIGINAL_SESSION_LOCAL


def _get_token(client, email="training-tester@example.com", tenant_id="telco"):
    client.post("/auth/register", json={"email": email, "password": "s3cret-pw", "tenant_id": tenant_id, "role": "analyst"})
    response = client.post("/auth/login", json={"email": email, "password": "s3cret-pw"})
    return response.json()["access_token"]


def _register_company(client, company_name, email, password="s3cret-pw1"):
    """Full self-registration flow (POST /api/companies/register) - unlike
    _get_token's bare /auth/register, this creates an actual Company row,
    which is what _apply_training_outcome (api/training.py) looks up to
    decide whether to flip feature_flags. Returns (tenant_id, token)."""
    response = client.post(
        "/api/companies/register",
        json={"company_name": company_name, "email": email, "password": password},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return body["tenant_id"], body["access_token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _upload(client, token, content: str = THIN_CSV, filename: str = "upload.csv"):
    return client.post(
        "/api/onboarding/upload",
        headers=_auth_headers(token),
        files={"file": (filename, content.encode("utf-8"), "text/csv")},
    )


def _upload_and_validate(client, token, content: str) -> int:
    """Uploads `content` and confirms whatever column mapping the backend
    suggests for it - correct here only because both _make_noisy_csv and
    _make_signal_csv use the exact role-bearing header names
    (customer_id/target/revenue) suggest_mapping already recognizes."""
    upload = _upload(client, token, content).json()
    suggested = upload["suggested_mapping"]
    column_mapping = {column: info["suggested_role"] for column, info in suggested.items()}
    validate_response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": column_mapping},
    )
    assert validate_response.status_code == 200, validate_response.text
    return upload["upload_id"]


def _poll_until_terminal(client, token, job_id, timeout_seconds=15.0, poll_interval=0.02):
    """Polls GET /api/training/status/{job_id} until status is a terminal
    state (succeeded/failed), returning every distinct status observed
    along the way plus the final body. Fails the test outright (rather than
    silently returning a non-terminal state) if the job never finishes
    within timeout_seconds - a hang here means the job infrastructure
    itself is broken, not a flaky assertion downstream."""
    seen_statuses = []
    last_status = None
    deadline = time.monotonic() + timeout_seconds
    body = None
    while time.monotonic() < deadline:
        body = client.get(f"/api/training/status/{job_id}", headers=_auth_headers(token)).json()
        if body["status"] != last_status:
            seen_statuses.append(body["status"])
            last_status = body["status"]
        if body["status"] in ("succeeded", "failed"):
            return seen_statuses, body
        time.sleep(poll_interval)
    raise AssertionError(f"Job {job_id} never reached a terminal status within {timeout_seconds}s (last seen: {body})")


def test_start_training_returns_immediately_without_waiting_for_completion(client):
    """The whole point of the background-thread design: POST /start must
    return long before the background thread (real training now) finishes."""
    token = _get_token(client, email="immediate-return@example.com")
    upload = _upload(client, token).json()

    started_at = time.monotonic()
    response = client.post(
        "/api/training/start",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"]},
    )
    elapsed = time.monotonic() - started_at

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert isinstance(body["job_id"], int)
    assert elapsed < 1.0, f"POST /api/training/start took {elapsed:.2f}s - it should return immediately"


def test_status_transitions_queued_running_then_succeeded(client):
    tenant_id, token = _register_company(client, "Status Transition Co", "status-transitions@example.com")
    try:
        upload_id = _upload_and_validate(client, token, _make_signal_csv(n_rows=300, seed=7))

        start_response = client.post(
            "/api/training/start",
            headers=_auth_headers(token),
            json={"upload_id": upload_id},
        )
        assert start_response.status_code == 202
        # The row is created with status=queued and committed BEFORE the
        # background thread is even started, so this is deterministic - no
        # race with the polling loop below is possible here.
        assert start_response.json()["status"] == "queued"
        job_id = start_response.json()["job_id"]

        seen_statuses, final = _poll_until_terminal(client, token, job_id, timeout_seconds=30.0)

        assert "running" in seen_statuses
        assert seen_statuses[-1] == "succeeded"
        assert final["status"] == "succeeded"
        assert final["started_at"] is not None
        assert final["completed_at"] is not None
        assert final["error_message"] is None

        metadata = final["result_metadata_json"]
        assert 0.55 < metadata["roc_auc"] < 0.98
        assert 0.0 < metadata["pr_auc"] <= 1.0
        assert metadata["is_sane"] is True
        assert metadata["features_auto_enabled"] is True
    finally:
        _cleanup_tenant_artifacts(tenant_id)


def test_failed_job_stores_error_message(client, monkeypatch):
    """Simulates a failure by monkeypatching just the "do the work" step
    (not the whole thread/status machinery) - _run_training_job's
    try/except around it is what's actually being tested here."""

    def _boom(db, tenant_id):
        raise RuntimeError("training backend unavailable")

    monkeypatch.setattr(training_module, "_run_training", _boom)

    token = _get_token(client, email="failed-job@example.com")
    upload = _upload(client, token).json()
    start_response = client.post(
        "/api/training/start",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"]},
    )
    job_id = start_response.json()["job_id"]

    # timeout_seconds above the 15s default: _boom() itself raises
    # instantly, but this project's background training-job threads
    # (api/training.py) are daemon threads pytest never joins between
    # tests, and other tests in a full-suite run start real, CPU-bound
    # hyperparameter searches (src/models/train.py, n_jobs=4) in their own
    # background threads - one of those still finishing can starve this
    # instant operation of CPU for a few seconds under a full-suite load.
    # A genuinely broken job infrastructure would still time out well
    # before this.
    seen_statuses, final = _poll_until_terminal(client, token, job_id, timeout_seconds=45.0)

    assert seen_statuses[-1] == "failed"
    assert final["status"] == "failed"
    assert final["error_message"] == "training backend unavailable"
    assert final["result_metadata_json"] is None
    assert final["started_at"] is not None
    assert final["completed_at"] is not None


def test_start_training_without_validated_upload_fails_with_real_error_message(client):
    """No column_mapping_json ever got set (upload was never validated) -
    real training must fail honestly, not silently fall back to anything."""
    token = _get_token(client, email="unvalidated-upload@example.com")
    upload = _upload(client, token, THIN_CSV).json()
    start_response = client.post(
        "/api/training/start",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"]},
    )
    job_id = start_response.json()["job_id"]

    seen_statuses, final = _poll_until_terminal(client, token, job_id)

    assert seen_statuses[-1] == "failed"
    assert "validated upload" in final["error_message"]
    assert final["result_metadata_json"] is None


def test_start_training_unknown_upload_returns_404(client):
    token = _get_token(client, email="unknown-upload@example.com")
    response = client.post(
        "/api/training/start",
        headers=_auth_headers(token),
        json={"upload_id": 999999},
    )
    assert response.status_code == 404


def test_status_unknown_job_returns_404(client):
    token = _get_token(client, email="unknown-job@example.com")
    response = client.get("/api/training/status/999999", headers=_auth_headers(token))
    assert response.status_code == 404


def test_tenant_cannot_view_another_tenants_job_status(client):
    """A banking-tenant user must never be able to check a telco tenant's
    training job status, even by guessing a valid job id."""
    telco_token = _get_token(client, email="telco-owner@example.com", tenant_id="telco")
    banking_token = _get_token(client, email="banking-other@example.com", tenant_id="banking")

    upload = _upload(client, telco_token).json()
    start_response = client.post(
        "/api/training/start",
        headers=_auth_headers(telco_token),
        json={"upload_id": upload["upload_id"]},
    )
    job_id = start_response.json()["job_id"]

    cross_tenant_response = client.get(f"/api/training/status/{job_id}", headers=_auth_headers(banking_token))
    assert cross_tenant_response.status_code == 404

    same_tenant_response = client.get(f"/api/training/status/{job_id}", headers=_auth_headers(telco_token))
    assert same_tenant_response.status_code == 200

    # Let the background thread finish before the test (and its in-memory
    # DB/engine) tears down, so it isn't left writing to a closed engine.
    _poll_until_terminal(client, telco_token, job_id)


def test_start_training_requires_upload_to_belong_to_caller_tenant(client):
    """A banking-tenant user must never be able to start a training job
    against a telco tenant's upload_id, even by guessing a valid id."""
    telco_token = _get_token(client, email="telco-owner-2@example.com", tenant_id="telco")
    banking_token = _get_token(client, email="banking-other-2@example.com", tenant_id="banking")

    upload = _upload(client, telco_token).json()

    cross_tenant_response = client.post(
        "/api/training/start",
        headers=_auth_headers(banking_token),
        json={"upload_id": upload["upload_id"]},
    )
    assert cross_tenant_response.status_code == 404


def test_real_training_on_known_good_sample_produces_sane_metrics_and_enables_features(client):
    """End-to-end against Meridian Wireless's real validated sample CSV
    (the same shape/columns as the fully-validated Telco reference model) -
    real training must produce a real, sane ROC-AUC/PR-AUC and actually
    flip the tenant's feature_flags on, not a simulated placeholder.

    Uses the PURE auto-suggested mapping (no manual role override) - a
    "tenure"-named column now suggests role="duration" on its own (see
    src/onboarding/schema_fields.py's "tenure" entry comment: this used to
    require a human to manually re-pick "duration" every time, which is
    exactly what cost cascade-retails Survival Analysis for real), so
    survival now genuinely RUNS AND PASSES under pure auto-suggestion here
    too (asserted below) - not skipped the way it used to be. clv still
    correctly stays skipped: Meridian's real sample has no CLV-equivalent
    column at all (Telco's own real CLV column lives in a separately-
    enriched dataset, not the base schema this sample shares). segments/
    anomalies need no such role and always run regardless."""
    tenant_id, token = _register_company(client, "Known Good Sample Co", "known-good-sample@example.com")
    try:
        csv_text = MERIDIAN_SAMPLE_CSV.read_text(encoding="utf-8")
        upload_id = _upload_and_validate(client, token, csv_text)

        start_response = client.post(
            "/api/training/start",
            headers=_auth_headers(token),
            json={"upload_id": upload_id},
        )
        assert start_response.status_code == 202
        job_id = start_response.json()["job_id"]

        _, final = _poll_until_terminal(client, token, job_id, timeout_seconds=120.0)

        assert final["status"] == "succeeded"
        metadata = final["result_metadata_json"]
        assert 0.55 < metadata["roc_auc"] < 0.98, metadata
        assert 0.0 < metadata["pr_auc"] <= 1.0
        assert metadata["is_sane"] is True
        assert "sanity_warning" not in metadata
        assert metadata["features_auto_enabled"] is True

        modules = metadata["modules"]
        # "tenure" now auto-suggests role="duration" (see this test's own
        # docstring) - survival genuinely runs and passes here, not skipped.
        assert modules["survival"]["ran"] is True and modules["survival"]["passed"] is True, modules["survival"]
        assert 0.55 < modules["survival"]["metrics"]["c_index"] < 0.95
        assert modules["clv"]["ran"] is False  # Meridian's sample has no CLV-equivalent column at all
        # clv_estimated is the formula-based proxy this exact gap unlocks
        # (src/models/clv.py's estimate_clv_bulk()) - genuinely attempted
        # and passed since a real duration_column and a passing survival
        # model both exist for this sample.
        assert modules["clv_estimated"]["ran"] is True and modules["clv_estimated"]["passed"] is True, modules["clv_estimated"]
        assert modules["segments"]["ran"] is True and modules["segments"]["passed"] is True, modules["segments"]
        assert modules["anomalies"]["ran"] is True and modules["anomalies"]["passed"] is True, modules["anomalies"]
        # Stage 2c: priority_ranking/backtest/customer_timeline have no role-
        # mapping prerequisite (unlike survival/clv above) and always run;
        # scenario_simulator's structural prerequisite is segments (not
        # survival/clv), which passed above, so it runs too.
        assert modules["priority_ranking"]["ran"] is True and modules["priority_ranking"]["passed"] is True, modules["priority_ranking"]
        assert modules["backtest"]["ran"] is True and modules["backtest"]["passed"] is True, modules["backtest"]
        assert modules["customer_timeline"]["ran"] is True and modules["customer_timeline"]["passed"] is True, modules["customer_timeline"]
        assert modules["scenario_simulator"]["ran"] is True and modules["scenario_simulator"]["passed"] is True, modules["scenario_simulator"]
        # Stage 2d: budget_optimizer has no structural prerequisite - always runs, always passes.
        assert modules["budget_optimizer"]["ran"] is True and modules["budget_optimizer"]["passed"] is True, modules["budget_optimizer"]

        db = training_module.SessionLocal()
        try:
            company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
            assert company is not None
            assert company.model_dir == f"models/{tenant_id}"
            assert company.data_path == f"data/tenant_uploads/{tenant_id}.csv"
            # business_impact_core (classifier passed) + segments/anomalies/
            # survival (each independently passed its own gate) - clv absent
            # since it never ran at all (not "ran but failed"). Stage 2c adds
            # priority_ranking/backtest/customer_timeline/scenario_simulator,
            # Stage 2d adds budget_optimizer, each independently passed on
            # its own real gate (see modules assertions above).
            assert company.feature_flags_json == {
                "business_impact_core": True,
                "survival": True,
                "segments": True,
                "anomalies": True,
                "priority_ranking": True,
                "backtest": True,
                "customer_timeline": True,
                "scenario_simulator": True,
                "budget_optimizer": True,
                "clv_estimated": True,
            }
        finally:
            db.close()
    finally:
        _cleanup_tenant_artifacts(tenant_id)


def test_real_training_with_duration_and_clv_roles_mapped_enables_survival_and_honestly_skips_clv(client):
    """Same Meridian sample, but with "tenure" manually remapped to the new
    "duration" onboarding role (see src/onboarding/schema_fields.py) - the
    exact scenario the guided upload flow's mapping dropdown now enables a
    human to choose. Proves survival.py's tenant_config-driven generalization
    actually produces a real, gate-passing result through the full HTTP
    pipeline, not just when called directly in Python. clv still has no
    real column to map (Meridian's sample has no CLV-equivalent field) -
    correctly reported as unavailable, not fabricated."""
    tenant_id, token = _register_company(
        client, "Duration Mapped Sample Co", "duration-mapped-sample@example.com"
    )
    try:
        csv_text = MERIDIAN_SAMPLE_CSV.read_text(encoding="utf-8")
        upload = _upload(client, token, csv_text, filename="m.csv").json()
        mapping = {column: info["suggested_role"] for column, info in upload["suggested_mapping"].items()}
        mapping["tenure"] = "duration"
        validate_response = client.post(
            "/api/onboarding/validate",
            headers=_auth_headers(token),
            json={"upload_id": upload["upload_id"], "column_mapping": mapping},
        )
        assert validate_response.status_code == 200, validate_response.text

        start_response = client.post(
            "/api/training/start",
            headers=_auth_headers(token),
            json={"upload_id": upload["upload_id"]},
        )
        job_id = start_response.json()["job_id"]

        _, final = _poll_until_terminal(client, token, job_id, timeout_seconds=120.0)

        assert final["status"] == "succeeded"
        modules = final["result_metadata_json"]["modules"]
        assert modules["survival"]["ran"] is True, modules["survival"]
        assert modules["survival"]["passed"] is True, modules["survival"]
        assert 0.55 < modules["survival"]["metrics"]["c_index"] < 0.95
        assert "Contract" in modules["survival"]["metrics"]["coefficients"]
        assert modules["clv"] == {
            "ran": False,
            "passed": False,
            "metrics": None,
            "reason": (
                "unavailable: no CLV-equivalent column provided (map a column to the 'clv' role to "
                "enable CLV modeling)"
            ),
        }
        # clv_estimated: the formula-based proxy for exactly this gap - a
        # real duration_column (mapped above) and a passing survival model
        # both exist, so it genuinely runs and passes.
        assert modules["clv_estimated"]["ran"] is True and modules["clv_estimated"]["passed"] is True, modules["clv_estimated"]

        db = training_module.SessionLocal()
        try:
            company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
            assert company is not None
            assert company.feature_flags_json["survival"] is True
            assert company.feature_flags_json["clv_estimated"] is True
            assert "clv" not in company.feature_flags_json
        finally:
            db.close()
    finally:
        _cleanup_tenant_artifacts(tenant_id)


def test_real_training_on_bad_tiny_dataset_produces_low_auc_and_does_not_enable_classifier_features(client):
    """A genuinely uninformative feature column must land a low (near-
    random) ROC-AUC, not crash - and that low score must NOT auto-enable
    business_impact_core. Same caution this project applies to its own
    metrics (see src/models/tenant_training.py's MIN_SANE_ROC_AUC).

    segments/anomalies are independent, unsupervised modules that don't use
    the target at all - they're gated on their OWN real result (see
    src/models/tenant_training.py's module docstring), not suppressed just
    because the classifier came out unsane. This tiny dataset's only mapped
    feature (noise_feature) doesn't clear suggest_mapping's confidence
    threshold on its own generic name, so it's suggested "ignore" - leaving
    revenue as segment.py/anomaly.py's only real feature, which they still
    handle correctly (not a crash, not a silently-inflated pass)."""
    tenant_id, token = _register_company(client, "Bad Tiny Dataset Co", "bad-tiny-dataset@example.com")
    try:
        upload_id = _upload_and_validate(client, token, _make_noisy_csv(n_rows=200, seed=99))

        start_response = client.post(
            "/api/training/start",
            headers=_auth_headers(token),
            json={"upload_id": upload_id},
        )
        job_id = start_response.json()["job_id"]

        _, final = _poll_until_terminal(client, token, job_id, timeout_seconds=30.0)

        assert final["status"] == "succeeded"  # the pipeline itself ran fine - it's the result that's bad
        metadata = final["result_metadata_json"]
        assert metadata["roc_auc"] <= 0.55, metadata
        assert metadata["is_sane"] is False
        assert "sanity_warning" in metadata
        assert metadata["features_auto_enabled"] is False

        db = training_module.SessionLocal()
        try:
            company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
            assert company is not None
            # business_impact_core specifically must stay off - the
            # classifier itself is unsane - regardless of what segments/
            # anomalies independently did.
            assert "business_impact_core" not in (company.feature_flags_json or {})
        finally:
            db.close()
    finally:
        _cleanup_tenant_artifacts(tenant_id)


def test_real_training_model_artifact_structure_matches_existing_tenants(client):
    """The core classifier artifact files a real training job writes for a
    self-registered tenant must be the exact same four files already used
    for Telco/Banking (models/v1, models/banking_v1) - nothing new
    invented for the classifier itself. Since Stage 2b, segment_model.pkl/
    anomaly_model.pkl are ALSO written unconditionally (segments/anomalies
    need no optional role to run - see src/models/tenant_training.py's
    _run_optional_modules()); survival_model.pkl/clv_model.pkl are absent
    here since this sample maps neither "duration" nor "clv"."""
    tenant_id, token = _register_company(client, "Artifact Structure Co", "artifact-structure@example.com")
    try:
        upload_id = _upload_and_validate(client, token, _make_signal_csv(n_rows=200, seed=13))

        start_response = client.post(
            "/api/training/start",
            headers=_auth_headers(token),
            json={"upload_id": upload_id},
        )
        job_id = start_response.json()["job_id"]
        _, final = _poll_until_terminal(client, token, job_id, timeout_seconds=30.0)
        assert final["status"] == "succeeded", final

        model_dir = ROOT / "models" / tenant_id
        core_artifact_files = {"model.pkl", "encoders.pkl", "metadata.json", "split_indices.json"}
        always_on_module_files = {"segment_model.pkl", "anomaly_model.pkl"}
        existing_tenant_files = {p.name for p in (ROOT / "models" / "v1").iterdir() if p.is_file()}
        new_tenant_files = {p.name for p in model_dir.iterdir() if p.is_file()}
        # models/v1 (Telco) additionally has survival_model.pkl/clv_model.pkl
        # (this sample maps neither "duration" nor "clv", so those two stay
        # absent here) - the files below are the ones that must match.
        assert new_tenant_files == core_artifact_files | always_on_module_files
        assert core_artifact_files | always_on_module_files <= existing_tenant_files
    finally:
        _cleanup_tenant_artifacts(tenant_id)
