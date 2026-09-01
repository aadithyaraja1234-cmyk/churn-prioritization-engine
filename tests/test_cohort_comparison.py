import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base


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
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _get_token(client, tenant_id):
    email = f"cohort-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


def test_cohort_comparison_returns_real_stored_metrics_for_both_tenants(client):
    telco_metadata = json.loads(Path("models/v1/metadata.json").read_text())
    banking_metadata = json.loads(Path("models/banking_v1/metadata.json").read_text())

    token = _get_token(client, "telco")
    response = client.get("/api/cohort-comparison", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()

    assert payload["telco"]["roc_auc"] == telco_metadata["roc_auc"]
    assert payload["telco"]["pr_auc"] == telco_metadata["pr_auc"]
    assert payload["telco"]["n_train"] == telco_metadata["n_train"]
    assert payload["telco"]["n_test"] == telco_metadata["n_test"]

    assert payload["banking"]["roc_auc"] == banking_metadata["roc_auc"]
    assert payload["banking"]["pr_auc"] == banking_metadata["pr_auc"]
    assert payload["banking"]["n_train"] == banking_metadata["n_train"]
    assert payload["banking"]["n_test"] == banking_metadata["n_test"]


def test_cohort_comparison_accessible_regardless_of_caller_tenant(client):
    # This is an intentional cross-tenant aggregate view - a banking user
    # should still see both cohorts, not just their own.
    token = _get_token(client, "banking")
    response = client.get("/api/cohort-comparison", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["telco"] is not None
    assert payload["banking"] is not None


def test_cohort_comparison_includes_top_5_features_per_tenant(client):
    token = _get_token(client, "telco")
    response = client.get("/api/cohort-comparison", headers={"Authorization": f"Bearer {token}"})
    payload = response.json()

    assert len(payload["telco"]["top_features"]) == 5
    assert len(payload["banking"]["top_features"]) == 5
    assert payload["telco"]["top_features"][0]["feature"] == "Contract"
    assert payload["banking"]["top_features"][0]["feature"] == "Age"


def test_cohort_comparison_requires_authentication(client):
    response = client.get("/api/cohort-comparison")
    assert response.status_code == 401


def test_cohort_comparison_contains_no_customer_level_data(client):
    # Aggregate model stats only - it must be structurally impossible for a
    # banking user to see individual Telco (or Banking) customer records
    # through this endpoint, only model-level statistics.
    token = _get_token(client, "banking")
    response = client.get("/api/cohort-comparison", headers={"Authorization": f"Bearer {token}"})
    payload = response.json()

    for cohort in (payload["telco"], payload["banking"]):
        assert set(cohort.keys()) == {
            "roc_auc",
            "pr_auc",
            "n_train",
            "n_test",
            "churn_rate_train",
            "churn_rate_test",
            "top_features",
        }
        for feature_row in cohort["top_features"]:
            assert set(feature_row.keys()) == {"feature", "importance"}
