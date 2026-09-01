"""API-key auth tests (Part 2 of the API-docs/API-key feature).

The single most important test here is
test_api_key_tenant_isolation_matches_jwt_isolation_rigor - it must prove an
API key can never retrieve another tenant's data, with the same rigor as
tests/test_tenant_isolation.py proves it for the DB-query layer and
tests/test_api_endpoints.py proves it for JWT auth.
"""

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import api_keys as api_keys_module
from api.main import app
from database.db import get_db
from database.models import Base
from src.models.business_impact import compute_business_impact_bulk

MODEL_DIR = "models/v1"
DATA_PATH = "data/raw/telco.csv"
CLV_DATA_PATH = "data/raw/telco_enriched.csv"


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
    api_keys_module.reset_rate_limits_for_testing()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    api_keys_module.reset_rate_limits_for_testing()


def _get_token(client, email, password, tenant_id, role="analyst"):
    client.post("/auth/register", json={"email": email, "password": password, "tenant_id": tenant_id, "role": role})
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _api_key_header(key):
    return {"X-API-Key": key}


@pytest.fixture()
def telco_token(client):
    return _get_token(client, "telco-keyowner@example.com", "s3cret-pw", "telco")


@pytest.fixture()
def banking_token(client):
    return _get_token(client, "banking-keyowner@example.com", "s3cret-pw", "banking")


def _create_key(client, token, name="test-integration", rate_limit_per_minute=None):
    body = {"name": name}
    if rate_limit_per_minute is not None:
        body["rate_limit_per_minute"] = rate_limit_per_minute
    response = client.post("/api/api-keys", json=body, headers=_bearer(token))
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def telco_api_key(client, telco_token):
    return _create_key(client, telco_token, name="telco-integration")


@pytest.fixture()
def banking_api_key(client, banking_token):
    return _create_key(client, banking_token, name="banking-integration")


# --- Key lifecycle ---


def test_create_api_key_returns_raw_key_once_and_never_again(client, telco_token):
    created = _create_key(client, telco_token)
    assert created["api_key"].startswith("sk_live_")
    assert created["masked_key"].startswith("sk_live_....")
    assert created["masked_key"] != created["api_key"]
    assert created["masked_key"].endswith(created["api_key"][-4:])
    assert created["rate_limit_per_minute"] == 60

    listed = client.get("/api/api-keys", headers=_bearer(telco_token)).json()
    assert len(listed) == 1
    assert listed[0]["masked_key"] == created["masked_key"]
    # The raw key must never appear anywhere in the list response.
    assert "api_key" not in listed[0]
    assert created["api_key"] not in str(listed)


def test_new_api_key_works_immediately_against_a_protected_endpoint(client, telco_api_key):
    response = client.get("/api/business-impact", headers=_api_key_header(telco_api_key["api_key"]))
    assert response.status_code == 200
    assert response.json()["metadata"] is not None


def test_revoke_api_key_soft_deletes_and_rejects_future_use(client, telco_token, telco_api_key):
    key_id = telco_api_key["id"]
    raw_key = telco_api_key["api_key"]

    # Works before revocation.
    assert client.get("/api/business-impact", headers=_api_key_header(raw_key)).status_code == 200

    revoke_response = client.delete(f"/api/api-keys/{key_id}", headers=_bearer(telco_token))
    assert revoke_response.status_code == 200
    assert revoke_response.json()["revoked_at"] is not None

    # The row must still exist (soft delete, for audit purposes), just revoked.
    listed = client.get("/api/api-keys", headers=_bearer(telco_token)).json()
    assert listed[0]["revoked"] is True
    assert listed[0]["revoked_at"] is not None

    # Immediately rejected afterward.
    rejected = client.get("/api/business-impact", headers=_api_key_header(raw_key))
    assert rejected.status_code == 401


def test_unknown_api_key_is_rejected(client):
    response = client.get("/api/business-impact", headers=_api_key_header("sk_live_not-a-real-key"))
    assert response.status_code == 401


def test_revoking_someone_elses_key_returns_404(client, telco_token, banking_api_key):
    response = client.delete(f"/api/api-keys/{banking_api_key['id']}", headers=_bearer(telco_token))
    assert response.status_code == 404

    # And the banking key must still work - the failed cross-tenant revoke
    # attempt must not have silently affected it.
    still_works = client.get(
        "/api/business-impact", headers=_api_key_header(banking_api_key["api_key"])
    )
    assert still_works.status_code == 200


# --- Rate limiting ---


def test_rate_limit_triggers_429_once_exceeded(client, telco_token):
    created = _create_key(client, telco_token, name="rate-limited", rate_limit_per_minute=2)
    raw_key = created["api_key"]

    first = client.get("/api/business-impact", headers=_api_key_header(raw_key))
    second = client.get("/api/business-impact", headers=_api_key_header(raw_key))
    third = client.get("/api/business-impact", headers=_api_key_header(raw_key))

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "rate limit" in third.json()["detail"].lower()


def test_rate_limit_is_per_key_not_shared_across_keys(client, telco_token):
    key_a = _create_key(client, telco_token, name="key-a", rate_limit_per_minute=1)
    key_b = _create_key(client, telco_token, name="key-b", rate_limit_per_minute=1)

    assert client.get("/api/business-impact", headers=_api_key_header(key_a["api_key"])).status_code == 200
    # key_a is now at its limit, but key_b has its own independent window.
    assert client.get("/api/business-impact", headers=_api_key_header(key_a["api_key"])).status_code == 429
    assert client.get("/api/business-impact", headers=_api_key_header(key_b["api_key"])).status_code == 200


# --- Excluded endpoints: must reject API-key auth even if one is supplied ---


def test_copilot_chat_rejects_api_key_auth(client, telco_api_key):
    response = client.post(
        "/api/copilot/chat",
        json={"message": "hello"},
        headers=_api_key_header(telco_api_key["api_key"]),
    )
    assert response.status_code == 401


def test_morning_brief_rejects_api_key_auth(client, telco_api_key):
    response = client.get("/api/copilot/morning-brief", headers=_api_key_header(telco_api_key["api_key"]))
    assert response.status_code == 401


def test_api_key_management_endpoints_reject_api_key_auth(client, telco_api_key):
    headers = _api_key_header(telco_api_key["api_key"])
    assert client.get("/api/api-keys", headers=headers).status_code == 401
    assert client.post("/api/api-keys", json={"name": "x"}, headers=headers).status_code == 401
    assert client.delete(f"/api/api-keys/{telco_api_key['id']}", headers=headers).status_code == 401


def test_auth_endpoints_are_unaffected_by_a_stray_api_key_header(client, telco_api_key):
    """/auth/* never consults X-API-Key at all - a key present alongside a
    login/register call must be silently ignored, not granted any effect."""
    response = client.post(
        "/auth/login",
        json={"email": "telco-keyowner@example.com", "password": "s3cret-pw"},
        headers=_api_key_header(telco_api_key["api_key"]),
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


# --- CRITICAL: API-key auth must enforce the exact same tenant isolation as JWT ---


def test_api_key_tenant_isolation_matches_jwt_isolation_rigor(
    client, telco_token, banking_token, telco_api_key, banking_api_key
):
    """The single most important test in this feature. Mirrors the rigor of
    tests/test_tenant_isolation.py (DB layer) and
    tests/test_api_endpoints.py's banking-vs-telco assertions (JWT layer),
    but for the new API-key auth path.

    Banking was fully trained (see config/config.yaml's banking entry) -
    this test used to lean on banking's disabled feature_flags as its
    "must never see telco's data" signal (an unavailable placeholder can't
    leak anything). Now both tenants have real, independently-scored data,
    which is actually a STRONGER isolation proof: a banking key must
    return its OWN real Balance-based numbers, genuinely different from
    telco's real MonthlyCharges-based ones, never telco's."""
    telco_key_headers = _api_key_header(telco_api_key["api_key"])
    banking_key_headers = _api_key_header(banking_api_key["api_key"])
    telco_jwt_headers = _bearer(telco_token)
    banking_jwt_headers = _bearer(banking_token)

    # 1. A telco API key gets exactly the same real data a telco JWT gets -
    #    proves the API-key path routes through the identical tenant-scoped
    #    code, not a parallel/looser path.
    telco_via_key = client.get("/api/business-impact", headers=telco_key_headers)
    telco_via_jwt = client.get("/api/business-impact", headers=telco_jwt_headers)
    assert telco_via_key.status_code == 200
    assert telco_via_jwt.status_code == 200
    assert telco_via_key.json() == telco_via_jwt.json()
    assert len(telco_via_key.json()["customers"]) > 0

    # 2. A banking API key must NEVER retrieve telco's data on the same
    #    endpoint - it must get exactly what a banking JWT gets: its OWN
    #    real, independently-scored customer list, never telco's.
    banking_via_key = client.get("/api/business-impact", headers=banking_key_headers)
    banking_via_jwt = client.get("/api/business-impact", headers=banking_jwt_headers)
    assert banking_via_key.status_code == 200
    assert banking_via_key.json() == banking_via_jwt.json()
    banking_customers = banking_via_key.json()["customers"]
    assert len(banking_customers) > 0
    telco_customer_ids = {c["customerID"] for c in telco_via_key.json()["customers"]}
    banking_customer_ids = {c["customerID"] for c in banking_customers}
    assert telco_customer_ids.isdisjoint(banking_customer_ids)

    # 3. Priority ranking: both keys get real, tenant-shaped rankings, but
    #    over disjoint customer populations - never leaking into each other.
    telco_priority = client.get("/api/priority", headers=telco_key_headers)
    banking_priority = client.get("/api/priority", headers=banking_key_headers)
    assert telco_priority.status_code == 200
    assert isinstance(telco_priority.json(), list) and len(telco_priority.json()) > 0
    assert banking_priority.status_code == 200
    assert isinstance(banking_priority.json(), list) and len(banking_priority.json()) > 0
    telco_priority_ids = {row["customerID"] for row in telco_priority.json()}
    banking_priority_ids = {row["customerID"] for row in banking_priority.json()}
    assert telco_priority_ids.isdisjoint(banking_priority_ids)

    # 4. Customer-detail lookup: a real telco customer ID looked up with the
    #    BANKING key must not leak telco's data - banking's business_impact_core
    #    is enabled now, so it genuinely attempts the lookup against its OWN
    #    population and correctly 404s (that ID doesn't exist there), never
    #    silently falling through to telco's real churn_probability/health_score.
    impact_df = compute_business_impact_bulk(model_dir=MODEL_DIR, data_path=DATA_PATH, clv_data_path=CLV_DATA_PATH)
    real_telco_customer_id = impact_df.iloc[0]["customerID"]

    detail_via_telco_key = client.get(
        f"/api/customer/{real_telco_customer_id}/detail", headers=telco_key_headers
    )
    detail_via_banking_key = client.get(
        f"/api/customer/{real_telco_customer_id}/detail", headers=banking_key_headers
    )
    assert detail_via_telco_key.status_code == 200
    assert "churn_probability" in detail_via_telco_key.json()

    assert detail_via_banking_key.status_code == 404
    assert "churn_probability" not in detail_via_banking_key.json()

    # 5. Alerts: same shape-level isolation check, one more endpoint - both
    #    real now, structurally isolated from each other.
    telco_alerts = client.get("/api/alerts", headers=telco_key_headers)
    banking_alerts = client.get("/api/alerts", headers=banking_key_headers)
    assert telco_alerts.status_code == 200 and "available" not in telco_alerts.json()
    assert banking_alerts.status_code == 200 and "available" not in banking_alerts.json()

    # 6. Cross-check the reverse direction too: a banking API key must never
    #    be able to impersonate telco by any means available to it - there is
    #    no tenant_id parameter anywhere in these requests, so this is also a
    #    structural guarantee, not just a behavioral one.
    for endpoint in ("/api/business-impact", "/api/priority", "/api/alerts"):
        assert client.get(endpoint, headers=banking_key_headers).json() != client.get(
            endpoint, headers=telco_key_headers
        ).json()
