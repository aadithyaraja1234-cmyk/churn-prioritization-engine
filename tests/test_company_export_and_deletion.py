"""GET /api/companies/export and DELETE /api/companies/account: the
data-portability/right-to-erasure endpoints added for self-registered
tenants. Reference tenants (Telco/Banking) have no Company row and must be
rejected by deletion; export must never leak a password hash, an API key's
raw value, or its key_hash.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base

VALID_PASSWORD = "Str0ng!Passw0rd"


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


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _register(client, company_name="Acme Retail", email="founder@acme-retail.example", password=VALID_PASSWORD):
    return client.post(
        "/api/companies/register",
        json={"company_name": company_name, "email": email, "password": password},
    )


def _get_telco_token(client, role="analyst"):
    client.post(
        "/auth/register",
        json={"email": f"telco-{role}@example.com", "password": "s3cret-pw", "tenant_id": "telco", "role": role},
    )
    response = client.post("/auth/login", json={"email": f"telco-{role}@example.com", "password": "s3cret-pw"})
    return response.json()["access_token"]


def test_export_returns_company_and_never_leaks_password_or_key_hash(client):
    token = _register(client).json()["access_token"]
    headers = _auth_headers(token)

    client.post("/api/api-keys", headers=headers, json={"name": "integration"})

    body = client.get("/api/companies/export", headers=headers).json()
    assert body["company"]["company_name"] == "Acme Retail"
    assert body["users"] == [{"email": "founder@acme-retail.example", "role": "admin", "created_at": body["users"][0]["created_at"]}]
    assert len(body["api_keys"]) == 1
    assert body["api_keys"][0]["masked_key"].startswith("sk_live_")

    export_text = str(body)
    assert VALID_PASSWORD not in export_text
    assert "key_hash" not in export_text
    assert "sk_live_" in export_text  # masked_key is fine
    assert not any(k["masked_key"].count("sk_live_") and len(k["masked_key"]) > 20 for k in body["api_keys"])


def test_delete_account_requires_admin_role(client):
    _register(client, company_name="NonAdmin Co", email="founder@nonadmin.example")
    client.post(
        "/auth/register",
        json={"email": "analyst@nonadmin.example", "password": "s3cret-pw", "tenant_id": "nonadmin-co", "role": "analyst"},
    )
    analyst_token = client.post(
        "/auth/login", json={"email": "analyst@nonadmin.example", "password": "s3cret-pw"}
    ).json()["access_token"]

    response = client.delete("/api/companies/account", headers=_auth_headers(analyst_token))
    assert response.status_code == 403


def test_delete_account_rejects_reference_tenant_without_company_row(client):
    telco_admin_token = _get_telco_token(client, role="admin")
    response = client.delete("/api/companies/account", headers=_auth_headers(telco_admin_token))
    assert response.status_code == 404


def test_delete_account_removes_company_and_associated_data(client):
    token = _register(client, company_name="Doomed Co", email="founder@doomed.example").json()["access_token"]
    headers = _auth_headers(token)

    client.put("/api/tracking/cust-1", headers=headers, json={"status": "called"})
    client.post("/api/api-keys", headers=headers, json={"name": "integration"})

    response = client.delete("/api/companies/account", headers=headers)
    assert response.status_code == 204

    # The tenant no longer exists at all - even read-only lookups now 404,
    # not just "empty" (registering the exact same company name again
    # succeeds, which would fail if the old row were still there).
    assert client.get("/api/companies/onboarding-status", headers=headers).status_code == 404
    reregister = _register(client, company_name="Doomed Co", email="founder2@doomed.example")
    assert reregister.status_code == 201
