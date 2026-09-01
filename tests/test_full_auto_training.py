"""Stage 2b (full auto-training): end-to-end coverage of the real HTTP
pipeline (register -> upload -> map -> validate -> train -> poll) wiring
survival.py/segment.py/anomaly.py/clv.py into training, each gated on its
own real sanity check (src/models/tenant_training.py's
_run_optional_modules()) - not just the classifier tested by
test_training_jobs.py.

Two kinds of coverage, mirroring this project's existing "known good" vs.
"deliberately bad" pattern (test_training_jobs.py's Meridian sample vs.
_make_noisy_csv/_make_signal_csv):
  1. Meridian Wireless's real validated sample, with "tenure" manually
     mapped to the new "duration" role - every module produces a real,
     gate-passing result (clv correctly stays unavailable: Meridian's
     sample has no CLV-equivalent column at all, not a failure).
  2. Two deliberately-bad synthetic datasets, each engineered to make a
     SPECIFIC module's own sanity gate genuinely fail (not just produce a
     mediocre classifier) - survival/clv via a leaking/noise dataset,
     segmentation via a degenerate near-single-cluster one - confirming
     each failing module is trained (its real, honest metrics are
     reported) but NOT enabled.

Anomaly detection's gate isn't exercised as a failure case here: unlike
c-index/silhouette/R² (fitted metrics that can genuinely come out bad),
IsolationForest's flagged fraction is parameterized directly by
`contamination` and computed via predict() on the very data it was fit on,
so it lands within a hair of that chosen rate (5% here) by construction
for any reasonably-sized dataset - the 1%-20% gate exists to catch a
pathological/tiny-dataset rounding failure, not a normal "bad data"
scenario. Its PASSING behavior is still exercised in every scenario below.
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Any

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


def _make_leaky_and_noisy_csv(n_rows: int, seed: int) -> str:
    """Engineered to exercise survival's AUTOMATIC leakage exclusion (no
    duration_leakage_column role is mapped for this tenant below) AND
    clv's floor, while leaving the classifier/segmentation/anomaly
    unaffected:
      - leak_feature is (duration_val + tiny noise) - a near-exact copy of
        the survival duration, the same TotalCharges-shaped leak this
        project has hit before (see src/models/survival.py's docstring).
        With no duration_leakage_column mapped, train_survival_model()'s
        own automatic correlation-based check (DURATION_LEAKAGE_AUTO_
        DETECT_THRESHOLD) catches and excludes it BEFORE fitting - it
        never reaches the model, so c-index stays honest (and, since this
        synthetic duration_val carries no other real signal, lands near
        random and correctly fails survival's FLOOR gate instead of its
        leakage ceiling - see the test below).
      - clv_val is pure uniform noise, uncorrelated with anything -
        genuinely unpredictable, so R² should land at or below 0.
    """
    rng = random.Random(seed)
    lines = ["customer_id,target,revenue,duration_val,leak_feature,other_feature,clv_val"]
    for i in range(n_rows):
        target = "Yes" if i % 2 == 0 else "No"
        revenue = round(rng.uniform(20, 120), 2)
        duration_val = round(rng.uniform(1, 72), 2)
        leak_feature = round(duration_val + rng.uniform(-0.01, 0.01), 4)
        other_feature = round(rng.uniform(0, 1), 4)
        clv_val = round(rng.uniform(0, 1000), 2)
        lines.append(f"C{i},{target},{revenue},{duration_val},{leak_feature},{other_feature},{clv_val}")
    return "\n".join(lines) + "\n"


def _make_degenerate_segmentation_csv(n_rows: int, seed: int) -> str:
    """Engineered to fail segmentation's dominant-cluster-share gate: every
    feature (including revenue) takes only two possible values, sharing
    the SAME ~97%/3% majority/minority split across every column at once -
    the whole feature space is genuinely two point masses, so no k finds a
    better split than "everyone" vs. "the same tiny minority", regardless
    of how many clusters KMeans is allowed to search over."""
    rng = random.Random(seed)
    lines = ["customer_id,target,revenue,duration_val,degenerate_feature"]
    for i in range(n_rows):
        target = "Yes" if i % 2 == 0 else "No"
        is_minority = rng.random() < 0.03
        revenue = round((70.0 if not is_minority else 20.0) + rng.uniform(-0.05, 0.05), 2)
        duration_val = round((50.0 if not is_minority else 5.0) + rng.uniform(-0.05, 0.05), 2)
        degenerate_feature = "A" if not is_minority else "B"
        lines.append(f"C{i},{target},{revenue},{duration_val},{degenerate_feature}")
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
    training_module.SessionLocal = TestingSessionLocal

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    training_module.SessionLocal = _ORIGINAL_SESSION_LOCAL


def _register_company(client, company_name, email, password="s3cret-pw1"):
    response = client.post(
        "/api/companies/register",
        json={"company_name": company_name, "email": email, "password": password},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return body["tenant_id"], body["access_token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _upload_map_validate_train(client, token, csv_text: str, mapping_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Full onboarding wizard flow against `csv_text`, starting from the
    backend's own suggested mapping (mapping_overrides applied on top, so
    only the roles that matter for a given test deviate from
    auto-suggestion) - then starts training and polls to a terminal state.
    Returns the final GET /api/training/status/{job_id} body."""
    headers = _auth_headers(token)
    upload = client.post(
        "/api/onboarding/upload",
        headers=headers,
        files={"file": ("upload.csv", csv_text.encode("utf-8"), "text/csv")},
    ).json()

    mapping = {column: info["suggested_role"] for column, info in upload["suggested_mapping"].items()}
    mapping.update(mapping_overrides or {})

    validate_response = client.post(
        "/api/onboarding/validate",
        headers=headers,
        json={"upload_id": upload["upload_id"], "column_mapping": mapping},
    )
    assert validate_response.status_code == 200, validate_response.text

    start_response = client.post("/api/training/start", headers=headers, json={"upload_id": upload["upload_id"]})
    assert start_response.status_code == 202, start_response.text
    job_id = start_response.json()["job_id"]

    deadline = time.monotonic() + 120.0
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/training/status/{job_id}", headers=headers).json()
        if status["status"] in ("succeeded", "failed"):
            return status
        time.sleep(0.5)
    raise AssertionError(f"training job {job_id} did not reach a terminal state in time: {status}")


# --- Scenario 1: Meridian's real data, every module genuinely passes (or is honestly unavailable) ---


def test_meridian_full_pipeline_every_module_gates_correctly(client):
    tenant_id, token = _register_company(client, "Full Pipeline Meridian Co", "full-pipeline-meridian@example.com")
    try:
        csv_text = MERIDIAN_SAMPLE_CSV.read_text(encoding="utf-8")
        final = _upload_map_validate_train(client, token, csv_text, mapping_overrides={"tenure": "duration"})

        assert final["status"] == "succeeded", final
        metadata = final["result_metadata_json"]

        # Classifier: real, sane result.
        assert 0.55 < metadata["roc_auc"] < 0.98, metadata
        assert metadata["is_sane"] is True

        modules = metadata["modules"]

        # Survival: duration role mapped -> genuinely trained and passing.
        survival = modules["survival"]
        assert survival["ran"] is True and survival["passed"] is True, survival
        assert 0.55 < survival["metrics"]["c_index"] < 0.95
        assert "Contract" in survival["metrics"]["coefficients"]  # Meridian's own real top hazard factor

        # Segmentation: no role needed, always attempted - real, passing result.
        segments = modules["segments"]
        assert segments["ran"] is True and segments["passed"] is True, segments
        assert segments["metrics"]["chosen_k"] >= 2
        assert segments["metrics"]["silhouette"] > 0.05
        assert segments["metrics"]["dominant_cluster_share"] <= 0.90

        # Anomaly detection: no role needed, always attempted - real, passing result.
        anomalies = modules["anomalies"]
        assert anomalies["ran"] is True and anomalies["passed"] is True, anomalies
        assert 0.01 <= anomalies["metrics"]["flagged_fraction"] <= 0.20

        # CLV: honestly unavailable - Meridian's real sample has no CLV-equivalent column.
        clv = modules["clv"]
        assert clv["ran"] is False
        assert clv["passed"] is False
        assert clv["metrics"] is None
        assert "unavailable" in clv["reason"]

        # clv_estimated: the formula-based proxy for exactly this gap - a
        # real duration_column (mapped above) and a passing survival model
        # both exist, so it genuinely runs and passes.
        clv_estimated = modules["clv_estimated"]
        assert clv_estimated["ran"] is True and clv_estimated["passed"] is True, clv_estimated
        assert clv_estimated["metrics"]["coefficient_of_variation"] > 0.05

        # Stage 2c: priority_ranking/backtest/customer_timeline need no role
        # mapping and always run; scenario_simulator's only structural
        # prerequisite is segments (passed above), not survival/clv, so it
        # runs and passes here too even though clv didn't run at all.
        priority_ranking = modules["priority_ranking"]
        assert priority_ranking["ran"] is True and priority_ranking["passed"] is True, priority_ranking

        backtest = modules["backtest"]
        assert backtest["ran"] is True and backtest["passed"] is True, backtest
        assert backtest["metrics"]["n_churned_test"] >= 50

        customer_timeline = modules["customer_timeline"]
        assert customer_timeline["ran"] is True and customer_timeline["passed"] is True, customer_timeline

        scenario_simulator = modules["scenario_simulator"]
        assert scenario_simulator["ran"] is True and scenario_simulator["passed"] is True, scenario_simulator

        # Stage 2d: budget_optimizer has no structural prerequisite of its
        # own (see src/models/tenant_training.py's module docstring) -
        # always runs, always passes when business_impact_core does.
        budget_optimizer = modules["budget_optimizer"]
        assert budget_optimizer["ran"] is True and budget_optimizer["passed"] is True, budget_optimizer

        db = training_module.SessionLocal()
        try:
            company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
            assert company is not None
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
            assert "clv" not in company.feature_flags_json
        finally:
            db.close()
    finally:
        _cleanup_tenant_artifacts(tenant_id)


# --- Scenario 2: deliberately bad data, the modules that SHOULD fail their gates correctly don't get enabled ---


def test_leaky_and_noisy_dataset_auto_excludes_the_leak_and_still_fails_survival_and_clv_gates_honestly(client):
    """leak_feature IS a real TotalCharges-shaped leak, but with no
    duration_leakage_column role mapped, train_survival_model()'s own
    automatic correlation check (src/models/survival.py's
    DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD) catches and excludes it before
    fitting - so unlike the leakage-ceiling scenario this test used to
    describe, the fitted c-index here is honest, not inflated. Since this
    synthetic duration_val carries no other real relationship to anything,
    the honest result correctly fails survival's FLOOR gate instead - a
    different, but equally real, "don't enable it" outcome. See
    tests/test_survival.py's test_planted_totalcharges_shaped_leak_is_
    automatically_caught_and_excluded for the module-level regression test
    that the exclusion itself happens."""
    tenant_id, token = _register_company(client, "Leaky Noisy Co", "leaky-noisy@example.com")
    try:
        csv_text = _make_leaky_and_noisy_csv(n_rows=600, seed=1)
        final = _upload_map_validate_train(
            client,
            token,
            csv_text,
            # leak_feature/other_feature don't clear suggest_mapping's
            # confidence threshold on their own generic names (suggested
            # "ignore" by default) - explicit overrides so both actually
            # reach the classifier/survival covariates, same as a human
            # manually confirming a low-confidence suggestion would.
            mapping_overrides={
                "duration_val": "duration",
                "leak_feature": "feature",
                "other_feature": "feature",
                "clv_val": "clv",
            },
        )

        assert final["status"] == "succeeded", final
        modules = final["result_metadata_json"]["modules"]

        # Survival: leak_feature is automatically excluded before fitting
        # (no duration_leakage_column was mapped) - the real, honest
        # result has no leftover leak inflating it, so it correctly fails
        # the FLOOR gate (this synthetic duration_val has no other real
        # signal), not the leakage ceiling.
        survival = modules["survival"]
        assert survival["ran"] is True, survival
        assert survival["passed"] is False, survival
        assert survival["metrics"]["c_index"] <= 0.55
        assert "leak_feature" in survival["metrics"]["duration_leakage_warning"]
        assert "leak_feature" not in survival["metrics"]["coefficients"]
        assert "barely better than random" in survival["reason"]

        # CLV: clv_val is pure noise - a real, non-crashing fit that
        # correctly can't beat predicting the average.
        clv = modules["clv"]
        assert clv["ran"] is True, clv
        assert clv["passed"] is False, clv
        assert clv["metrics"]["r2"] <= 0.05
        assert "R²" in clv["reason"]

        # clv_estimated is skipped outright (not attempted-and-failed) since
        # this tenant DOES have a real, mapped CLV column (clv_val) - the
        # formula-based proxy is only ever offered when no real CLV column
        # exists at all, regardless of whether the real one passed its gate.
        assert modules["clv_estimated"]["ran"] is False
        assert "real, mapped CLV column" in modules["clv_estimated"]["reason"]

        # Segmentation/anomaly detection don't use duration_val/leak_feature/
        # clv_val's target-relationship at all - this dataset was engineered
        # to break survival/clv specifically, not these two, and they
        # should still produce real, passing, non-degenerate results.
        assert modules["segments"]["ran"] is True and modules["segments"]["passed"] is True, modules["segments"]
        assert modules["anomalies"]["ran"] is True and modules["anomalies"]["passed"] is True, modules["anomalies"]

        db = training_module.SessionLocal()
        try:
            company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
            assert company is not None
            assert "survival" not in company.feature_flags_json
            assert "clv" not in company.feature_flags_json
            assert company.feature_flags_json["segments"] is True
            assert company.feature_flags_json["anomalies"] is True
        finally:
            db.close()
    finally:
        _cleanup_tenant_artifacts(tenant_id)


def test_degenerate_dataset_fails_segmentation_gate(client):
    tenant_id, token = _register_company(client, "Degenerate Segments Co", "degenerate-segments@example.com")
    try:
        csv_text = _make_degenerate_segmentation_csv(n_rows=600, seed=1)
        final = _upload_map_validate_train(
            client,
            token,
            csv_text,
            # degenerate_feature doesn't clear suggest_mapping's confidence
            # threshold on its own generic name (suggested "ignore") -
            # explicit override so it actually reaches segment.py/anomaly.py
            # as a feature, same as a human confirming a low-confidence
            # suggestion would.
            mapping_overrides={"duration_val": "duration", "degenerate_feature": "feature"},
        )

        assert final["status"] == "succeeded", final
        modules = final["result_metadata_json"]["modules"]

        # Segmentation: every feature collapses to the same ~97%/3% point
        # masses at once - real result, correctly rejected as degenerate.
        segments = modules["segments"]
        assert segments["ran"] is True, segments
        assert segments["passed"] is False, segments
        assert segments["metrics"]["dominant_cluster_share"] > 0.90
        assert "single cluster" in segments["reason"]

        db = training_module.SessionLocal()
        try:
            company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
            assert company is not None
            assert "segments" not in company.feature_flags_json
        finally:
            db.close()
    finally:
        _cleanup_tenant_artifacts(tenant_id)
