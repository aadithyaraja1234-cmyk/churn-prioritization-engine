"""Regression coverage for the missing-db= bug class found across
priority_ranking/backtest/budget_optimizer/scenario_simulator/model-metrics/
model-importance in api/main.py, and for the copilot tool layer's own
identical bug (src/copilot/tools.py never threading db= to
feature_enabled()/model_dir_for()/etc. at all).

Without db=, get_tenant_profile() (src/tenant_registry.py) can never fall
through to a Company-backed profile for a self-registered tenant - every
one of these endpoints/tools 100%-of-the-time reported "unavailable" for
such a tenant regardless of their real feature_flags_json, even though the
underlying src/models/*.py functions themselves were already fully
tenant-generic. Registers and trains one real self-registered tenant (same
pattern as tests/test_full_auto_training.py) and confirms every one of
these actually returns real data now, not the old always-unavailable shape.
"""

from __future__ import annotations

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
from database.models import Base

_ORIGINAL_SESSION_LOCAL = training_module.SessionLocal

ROOT = Path(__file__).resolve().parents[1]
MERIDIAN_SAMPLE_CSV = ROOT / "data" / "test_onboarding_samples" / "company_meridian_wireless.csv"


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
        test_client.db_sessionmaker = TestingSessionLocal
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


@pytest.fixture()
def trained_tenant(client):
    """Registers and fully trains one real self-registered tenant (Meridian's
    validated sample, with duration mapped so survival/scenario_simulator
    both pass too) - every downstream test in this file reuses it rather
    than re-training per test."""
    tenant_id, token = _register_company(client, "Endpoint Coverage Co", "endpoint-coverage@example.com")
    final = _upload_map_validate_train(client, token, MERIDIAN_SAMPLE_CSV.read_text(encoding="utf-8"), mapping_overrides={"tenure": "duration"})
    assert final["status"] == "succeeded", final
    return tenant_id, token


def test_priority_and_backtest_return_real_data_not_unavailable(client, trained_tenant):
    tenant_id, token = trained_tenant
    headers = _auth_headers(token)

    priority_response = client.get("/api/priority", headers=headers)
    assert priority_response.status_code == 200
    priority_body = priority_response.json()
    assert isinstance(priority_body, list) and len(priority_body) > 0, priority_body
    assert "customerID" in priority_body[0]

    backtest_response = client.get("/api/backtest", headers=headers)
    assert backtest_response.status_code == 200
    backtest_body = backtest_response.json()
    assert "revenue_weighted" in backtest_body, backtest_body

    curve_response = client.get("/api/backtest/curve", params={"min_pct": 10, "max_pct": 20, "step_pct": 10}, headers=headers)
    assert curve_response.status_code == 200
    assert len(curve_response.json()) == 2


def test_budget_optimizer_returns_real_allocation_not_unavailable(client, trained_tenant):
    tenant_id, token = trained_tenant
    response = client.post("/api/budget-optimizer", json={"budget_amount": 1000}, headers=_auth_headers(token))
    assert response.status_code == 200
    body = response.json()
    assert body.get("available") is not False, body
    assert body["n_customers_covered"] > 0


def test_scenario_endpoints_return_real_data_not_unavailable(client, trained_tenant):
    tenant_id, token = trained_tenant
    headers = _auth_headers(token)

    run_response = client.post(
        "/api/scenario",
        json={"scenario_type": "uniform_charge_change", "params": {"percent_change": -10}, "scenario_name": "regression-check"},
        headers=headers,
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body.get("available") is not False, run_body
    assert "id" in run_body

    history_response = client.get("/api/scenario/history", headers=headers)
    assert history_response.status_code == 200
    history_body = history_response.json()
    assert isinstance(history_body, list) and len(history_body) == 1, history_body

    detail_response = client.get(f"/api/scenario/{run_body['id']}", headers=headers)
    assert detail_response.status_code == 200
    assert detail_response.json().get("available") is not False


def test_model_metrics_and_importance_return_real_data_not_unavailable(client, trained_tenant):
    tenant_id, token = trained_tenant
    headers = _auth_headers(token)

    metrics_response = client.get("/api/model/metrics", headers=headers)
    assert metrics_response.status_code == 200
    metrics_body = metrics_response.json()
    assert metrics_body.get("available") is not False, metrics_body
    assert "roc_auc" in metrics_body

    importance_response = client.get("/api/model/importance", headers=headers)
    assert importance_response.status_code == 200
    importance_body = importance_response.json()
    assert isinstance(importance_body, list) and len(importance_body) > 0, importance_body
    assert "feature" in importance_body[0]


def test_whatif_works_for_non_telco_columns(client, trained_tenant):
    tenant_id, token = trained_tenant
    headers = _auth_headers(token)

    priority_body = client.get("/api/priority", headers=headers).json()
    customer_id = priority_body[0]["customerID"]

    response = client.post(
        "/api/whatif",
        json={"customer_id": customer_id, "overrides": {"Contract": "Two year"}},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "original_probability" in body
    assert "new_probability" in body


def test_copilot_tools_resolve_a_self_registered_tenant_without_db_bug(client, trained_tenant):
    """Unit-level (no live Gemini call needed): exercises the exact
    src/copilot/tools.py functions the agent calls, with a real db session
    for this real self-registered tenant - each must return real data, not
    the old "not yet trained for this tenant" every one of these used to
    report 100% of the time regardless of feature_flags_json."""
    tenant_id, token = trained_tenant
    headers = _auth_headers(token)
    priority_body = client.get("/api/priority", headers=headers).json()
    customer_id = priority_body[0]["customerID"]

    from src.copilot import tools

    db = client.db_sessionmaker()
    try:
        summary = tools.get_business_impact_summary(tenant_id, db=db)
        assert "total_revenue_at_risk" in summary, summary

        opportunities = tools.get_top_opportunities(tenant_id, limit=3, db=db)
        assert len(opportunities["customers"]) > 0, opportunities

        detail = tools.get_customer_detail(customer_id, tenant_id, db=db)
        assert "churn_probability" in detail, detail

        budget = tools.run_budget_optimization(tenant_id, budget=500, db=db)
        assert "error" not in budget and budget.get("available") is not False, budget

        scenario = tools.run_scenario(tenant_id, "uniform_charge_change", {"percent_change": -5}, db=db)
        assert "error" not in scenario and scenario.get("available") is not False, scenario

        segments = tools.get_segment_summary(tenant_id, db=db)
        assert len(segments["segments"]) > 0, segments

        alerts = tools.get_alerts(tenant_id, db=db)
        assert "alerts" in alerts, alerts

        survival = tools.get_survival_likelihood(customer_id, tenant_id, db=db)
        assert "churn_likelihood" in survival, survival
    finally:
        db.close()


def test_training_populates_customers_and_predictions_tables_without_duplicating_on_retrain(client):
    """Regression test: self-registered tenants used to never get rows in
    the customers/predictions tables at all (database/migrate_csv_to_db.py
    only ever ran once, by hand, for Telco/Banking at initial setup) - GET
    /customers, and every UI surface built on it (Survival Likelihood,
    What-If Simulator, Customer 360's picker), silently showed "No
    customers found" for every self-registered tenant regardless of what
    actually trained. Also confirms re-training the same tenant doesn't
    duplicate rows.

    Deliberately does NOT go through the HTTP-triggered background-thread
    training path (unlike this file's other tests) - calls
    training_module._run_training() directly instead, with one fresh
    session per call, matching the real production shape exactly
    (api/training.py's _run_training_job() always opens its own brand-new
    SessionLocal() per job - see that function's own docstring). Two
    real trainings back-to-back through the actual background-thread path
    against this test file's StaticPool-shared-single-connection-across-
    threads fixture produced non-deterministic row-count flakiness that
    does NOT reproduce against the real file-based database/app.db
    (confirmed directly, twice) - a test-infrastructure-only artifact of
    that combination, not a real bug, and irrelevant to what this test
    actually needs to prove."""
    tenant_id, token = _register_company(client, "Retrain Coverage Co", "retrain-coverage@example.com")
    headers = _auth_headers(token)
    csv_text = MERIDIAN_SAMPLE_CSV.read_text(encoding="utf-8")

    def upload_and_validate():
        upload = client.post(
            "/api/onboarding/upload",
            headers=headers,
            files={"file": ("upload.csv", csv_text.encode("utf-8"), "text/csv")},
        ).json()
        mapping = {column: info["suggested_role"] for column, info in upload["suggested_mapping"].items()}
        mapping["tenure"] = "duration"
        response = client.post(
            "/api/onboarding/validate",
            headers=headers,
            json={"upload_id": upload["upload_id"], "column_mapping": mapping},
        )
        assert response.status_code == 200, response.text

    from database.models import Customer, Prediction

    upload_and_validate()
    db = client.db_sessionmaker()
    try:
        result_metadata = training_module._run_training(db, tenant_id)
        assert result_metadata["is_sane"] is True, result_metadata
    finally:
        db.close()

    customers_response = client.get("/customers", headers=headers)
    assert customers_response.status_code == 200
    customers = customers_response.json()
    assert len(customers) > 0, "customers table was never populated for this self-registered tenant"
    assert all(c["raw_features"] for c in customers)

    db = client.db_sessionmaker()
    try:
        customer_count = db.query(Customer).filter(Customer.tenant_id == tenant_id).count()
        prediction_count = db.query(Prediction).filter(Prediction.tenant_id == tenant_id).count()
        assert customer_count == len(customers)
        assert prediction_count > 0  # business_impact_core passed for this tenant - real predictions must exist too
    finally:
        db.close()

    # Re-train the exact same tenant (same upload, same mapping) and confirm
    # the tables reflect the latest run's population, not a duplicated one.
    upload_and_validate()
    db = client.db_sessionmaker()
    try:
        result_metadata = training_module._run_training(db, tenant_id)
        assert result_metadata["is_sane"] is True, result_metadata
    finally:
        db.close()

    db = client.db_sessionmaker()
    try:
        assert db.query(Customer).filter(Customer.tenant_id == tenant_id).count() == customer_count
        assert db.query(Prediction).filter(Prediction.tenant_id == tenant_id).count() == prediction_count
    finally:
        db.close()
