import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base, Customer


@pytest.fixture()
def session_factory():
    # StaticPool forces every connection from this engine to share the same
    # underlying SQLite :memory: database, so data written in one session is
    # visible to the next - without it, each session would see an empty DB.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)


@pytest.fixture()
def client(session_factory):
    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _get_token(client, email, password, tenant_id, role="analyst"):
    client.post("/auth/register", json={"email": email, "password": password, "tenant_id": tenant_id, "role": role})
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_unauthenticated_request_to_protected_endpoint_returns_401(client):
    response = client.get("/customers")
    assert response.status_code == 401


def test_login_wrong_password_fails_correct_password_succeeds(client):
    client.post(
        "/auth/register",
        json={"email": "user@telco.com", "password": "correct-horse-battery-staple", "tenant_id": "telco", "role": "analyst"},
    )

    wrong = client.post("/auth/login", json={"email": "user@telco.com", "password": "wrong-password"})
    assert wrong.status_code == 401

    correct = client.post("/auth/login", json={"email": "user@telco.com", "password": "correct-horse-battery-staple"})
    assert correct.status_code == 200
    assert "access_token" in correct.json()


def test_telco_token_cannot_retrieve_banking_data_via_tenant_id_query_param(client, session_factory):
    session = session_factory()
    session.add_all(
        [
            Customer(tenant_id="telco", customer_id="T-100", raw_features={"x": 1}),
            Customer(tenant_id="banking", customer_id="B-100", raw_features={"y": 1}),
        ]
    )
    session.commit()
    session.close()

    token = _get_token(client, "telco-user@example.com", "s3cret-pw", "telco")

    response = client.get(
        "/customers",
        params={"tenant_id": "banking"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["tenant_id"] == "telco"
    assert all(row["tenant_id"] != "banking" for row in payload)
