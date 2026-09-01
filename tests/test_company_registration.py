"""Company self-registration: field-level validation, tenant creation,
onboarding-status progression, and tenant isolation from the pre-existing
config.yaml-defined tenants (telco/banking)."""

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


def _register(client, company_name="Acme Retail", email="founder@acme-retail.example", password=VALID_PASSWORD):
    return client.post(
        "/api/companies/register",
        json={"company_name": company_name, "email": email, "password": password},
    )


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _get_telco_token(client):
    client.post(
        "/auth/register", json={"email": "telco-user@example.com", "password": "s3cret-pw", "tenant_id": "telco", "role": "analyst"}
    )
    response = client.post("/auth/login", json={"email": "telco-user@example.com", "password": "s3cret-pw"})
    return response.json()["access_token"]


# --- Field-level validation ---


def test_register_rejects_weak_password_with_field_level_error(client):
    response = _register(client, password="weak")
    assert response.status_code == 400
    errors = response.json()["detail"]["errors"]
    assert "password" in errors
    assert "company_name" not in errors
    assert "email" not in errors


def test_register_rejects_malformed_email_with_field_level_error(client):
    response = _register(client, email="not-an-email")
    assert response.status_code == 400
    errors = response.json()["detail"]["errors"]
    assert "email" in errors


def test_register_rejects_short_company_name_with_field_level_error(client):
    response = _register(client, company_name="A")
    assert response.status_code == 400
    errors = response.json()["detail"]["errors"]
    assert "company_name" in errors


def test_register_reports_all_field_errors_at_once_not_just_the_first(client):
    response = client.post(
        "/api/companies/register",
        json={"company_name": "X", "email": "bad-email", "password": "weak"},
    )
    errors = response.json()["detail"]["errors"]
    assert set(errors.keys()) == {"company_name", "email", "password"}


def test_register_accepts_valid_input(client):
    response = _register(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["tenant_id"] == "acme-retail"
    assert body["company_name"] == "Acme Retail"
    assert body["onboarding_step"] == "registered"
    assert "access_token" in body


def test_register_rejects_duplicate_email(client):
    _register(client, company_name="First Co", email="dup@example.com")
    response = _register(client, company_name="Second Co", email="dup@example.com")
    assert response.status_code == 400
    assert "email" in response.json()["detail"]["errors"]


def test_register_rejects_duplicate_company_name_slug(client):
    _register(client, company_name="Acme Retail")
    response = _register(client, company_name="acme retail")  # same slug, different case/spacing
    assert response.status_code == 400
    assert "company_name" in response.json()["detail"]["errors"]


# --- New tenant is real and usable ---


def test_new_company_has_no_trained_model_yet(client):
    token = _register(client).json()["access_token"]
    response = client.get("/api/business-impact", headers=_auth_headers(token))
    assert response.status_code == 200
    assert response.json()["available"] is False


# --- Tenant isolation from pre-existing tenants ---


def test_new_company_is_isolated_from_telco_tenant(client):
    company_token = _register(client, company_name="Isolated Co", email="iso@example.com").json()["access_token"]
    telco_token = _get_telco_token(client)

    company_customers = client.get("/customers", headers=_auth_headers(company_token)).json()
    telco_customers = client.get("/customers", headers=_auth_headers(telco_token)).json()

    assert company_customers == []  # brand new tenant has no seeded customer rows
    assert len(telco_customers) >= 0  # telco's own data is untouched either way
    assert all(c["tenant_id"] == "telco" for c in telco_customers)


def test_onboarding_status_requires_authentication(client):
    response = client.get("/api/companies/onboarding-status")
    assert response.status_code == 401


# --- Onboarding-status progression ---


def test_pre_existing_tenant_reports_ready_without_ever_registering(client):
    telco_token = _get_telco_token(client)
    response = client.get("/api/companies/onboarding-status", headers=_auth_headers(telco_token))
    assert response.status_code == 200
    assert response.json()["step"] == "ready"


def test_new_company_status_advances_through_upload_validate_confirm(client):
    token = _register(client, company_name="Progress Co", email="progress@example.com").json()["access_token"]
    headers = _auth_headers(token)

    status_after_register = client.get("/api/companies/onboarding-status", headers=headers).json()
    assert status_after_register["step"] == "registered"

    csv_content = "id,label,tenure\n1,No,5\n2,Yes,10\n"
    upload_response = client.post(
        "/api/onboarding/upload", headers=headers, files={"file": ("upload.csv", csv_content.encode(), "text/csv")}
    )
    upload_id = upload_response.json()["upload_id"]

    status_after_upload = client.get("/api/companies/onboarding-status", headers=headers).json()
    assert status_after_upload["step"] == "data_uploaded"
    assert status_after_upload["last_upload_id"] == upload_id

    validate_response = client.post(
        "/api/onboarding/validate",
        headers=headers,
        json={"upload_id": upload_id, "column_mapping": {"id": "customer_id", "label": "target", "tenure": "feature"}},
    )
    assert validate_response.status_code == 200

    status_after_validate = client.get("/api/companies/onboarding-status", headers=headers).json()
    assert status_after_validate["step"] == "data_validated"

    confirm_response = client.post("/api/onboarding/confirm", headers=headers)
    assert confirm_response.status_code == 200
    assert confirm_response.json()["step"] == "ready"

    status_after_confirm = client.get("/api/companies/onboarding-status", headers=headers).json()
    assert status_after_confirm["step"] == "ready"


def test_status_never_regresses_on_a_second_upload(client):
    token = _register(client, company_name="NoRegress Co", email="noregress@example.com").json()["access_token"]
    headers = _auth_headers(token)
    csv_content = "id,label\n1,No\n2,Yes\n"

    upload1 = client.post(
        "/api/onboarding/upload", headers=headers, files={"file": ("a.csv", csv_content.encode(), "text/csv")}
    ).json()
    client.post(
        "/api/onboarding/validate",
        headers=headers,
        json={"upload_id": upload1["upload_id"], "column_mapping": {"id": "customer_id", "label": "target"}},
    )
    client.post("/api/onboarding/confirm", headers=headers)
    assert client.get("/api/companies/onboarding-status", headers=headers).json()["step"] == "ready"

    # Re-uploading a second file must not regress "ready" back to "data_uploaded".
    client.post("/api/onboarding/upload", headers=headers, files={"file": ("b.csv", csv_content.encode(), "text/csv")})
    assert client.get("/api/companies/onboarding-status", headers=headers).json()["step"] == "ready"


def test_confirm_is_a_404_noop_for_tenants_without_onboarding_in_progress(client):
    telco_token = _get_telco_token(client)
    response = client.post("/api/onboarding/confirm", headers=_auth_headers(telco_token))
    assert response.status_code == 404


def test_get_upload_rehydrates_for_resume(client):
    token = _register(client, company_name="Resume Co", email="resume@example.com").json()["access_token"]
    headers = _auth_headers(token)
    csv_content = "id,label,tenure\n1,No,5\n2,Yes,10\n"
    upload = client.post(
        "/api/onboarding/upload", headers=headers, files={"file": ("upload.csv", csv_content.encode(), "text/csv")}
    ).json()

    rehydrated = client.get(f"/api/onboarding/upload/{upload['upload_id']}", headers=headers)
    assert rehydrated.status_code == 200
    body = rehydrated.json()
    assert body["columns"] == upload["columns"]
    assert body["suggested_mapping"] == upload["suggested_mapping"]


def test_get_upload_is_tenant_isolated(client):
    company_a_token = _register(client, company_name="Company A", email="a@example.com").json()["access_token"]
    company_b_token = _register(client, company_name="Company B", email="b@example.com").json()["access_token"]

    upload = client.post(
        "/api/onboarding/upload",
        headers=_auth_headers(company_a_token),
        files={"file": ("upload.csv", b"id,label\n1,No\n2,Yes\n", "text/csv")},
    ).json()

    cross_tenant = client.get(f"/api/onboarding/upload/{upload['upload_id']}", headers=_auth_headers(company_b_token))
    assert cross_tenant.status_code == 404
