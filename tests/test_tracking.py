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


def _get_token(client, tenant_id="telco"):
    email = f"tracking-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


def test_get_tracking_returns_default_not_started_state_when_no_row_exists(client):
    token = _get_token(client)
    response = client.get("/api/tracking/1234-ABCDE", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["customer_id"] == "1234-ABCDE"
    assert payload["status"] == "not_started"
    assert payload["assigned_manager"] is None
    assert payload["call_scheduled_date"] is None


def test_put_tracking_creates_and_persists_fields(client):
    token = _get_token(client)
    put_response = client.put(
        "/api/tracking/1234-ABCDE",
        json={"assigned_manager": "Jordan", "status": "called", "call_scheduled_date": "2026-08-01"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert put_response.status_code == 200
    payload = put_response.json()
    assert payload["assigned_manager"] == "Jordan"
    assert payload["status"] == "called"
    assert payload["call_scheduled_date"].startswith("2026-08-01")
    assert payload["updated_at"] is not None

    get_response = client.get("/api/tracking/1234-ABCDE", headers={"Authorization": f"Bearer {token}"})
    assert get_response.status_code == 200
    assert get_response.json()["assigned_manager"] == "Jordan"
    assert get_response.json()["status"] == "called"


def test_put_tracking_updates_existing_row_rather_than_duplicating(client):
    token = _get_token(client)
    client.put(
        "/api/tracking/1234-ABCDE",
        json={"assigned_manager": "Jordan", "status": "called"},
        headers={"Authorization": f"Bearer {token}"},
    )
    second_response = client.put(
        "/api/tracking/1234-ABCDE",
        json={"assigned_manager": "Alex", "status": "resolved"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert second_response.status_code == 200
    payload = second_response.json()
    assert payload["assigned_manager"] == "Alex"
    assert payload["status"] == "resolved"


@pytest.mark.parametrize("status", ["not_started", "called", "emailed", "resolved"])
def test_put_tracking_accepts_all_valid_statuses(client, status):
    token = _get_token(client)
    response = client.put(
        "/api/tracking/1234-ABCDE",
        json={"status": status},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == status


def test_put_tracking_rejects_invalid_status(client):
    token = _get_token(client)
    response = client.put(
        "/api/tracking/1234-ABCDE",
        json={"status": "not_a_real_status"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


def test_put_tracking_rejects_invalid_call_scheduled_date(client):
    token = _get_token(client)
    response = client.put(
        "/api/tracking/1234-ABCDE",
        json={"status": "called", "call_scheduled_date": "not-a-date"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


def test_tracking_is_tenant_scoped(client):
    telco_token = _get_token(client, tenant_id="telco")
    banking_token = _get_token(client, tenant_id="banking")

    client.put(
        "/api/tracking/SHARED-ID",
        json={"assigned_manager": "Telco Manager", "status": "called"},
        headers={"Authorization": f"Bearer {telco_token}"},
    )

    telco_view = client.get("/api/tracking/SHARED-ID", headers={"Authorization": f"Bearer {telco_token}"})
    assert telco_view.json()["assigned_manager"] == "Telco Manager"
    assert telco_view.json()["status"] == "called"

    # Same customer_id string, but a different tenant - must not see the
    # telco tenant's tracking row (tenant isolation is enforced by
    # get_tenant_scoped_query(), not by customer_id uniqueness).
    banking_view = client.get("/api/tracking/SHARED-ID", headers={"Authorization": f"Bearer {banking_token}"})
    assert banking_view.json()["assigned_manager"] is None
    assert banking_view.json()["status"] == "not_started"


def test_tracking_requires_authentication(client):
    get_response = client.get("/api/tracking/1234-ABCDE")
    assert get_response.status_code == 401

    put_response = client.put("/api/tracking/1234-ABCDE", json={"status": "called"})
    assert put_response.status_code == 401
