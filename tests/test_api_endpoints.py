import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base

PRIORITY_BACKTEST_NOT_VALIDATED = {
    "available": False,
    "reason": (
        "revenue-weighted prioritization was only validated for the Telco tenant; "
        "banking's revenue-proxy column (Balance) has not been backtested"
    ),
}
NOT_TRAINED = {"available": False, "reason": "not yet trained for this tenant"}


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


def _get_token(client, email, password, tenant_id, role="analyst"):
    client.post("/auth/register", json={"email": email, "password": password, "tenant_id": tenant_id, "role": role})
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture()
def telco_token(client):
    return _get_token(client, "telco-user@example.com", "s3cret-pw", "telco")


@pytest.fixture()
def banking_token(client):
    return _get_token(client, "banking-user@example.com", "s3cret-pw", "banking")


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


TELCO_ENDPOINTS = [
    ("/api/model/metrics", {}),
    ("/api/model/importance", {}),
    ("/api/priority", {"strategy": "revenue_weighted", "limit": 10}),
    ("/api/backtest", {"top_pct": 0.2}),
    ("/api/survival/segments", {}),
    ("/api/segments", {}),
    ("/api/anomalies", {"limit": 5}),
    ("/api/clv/importance", {}),
    ("/customers", {}),
]


@pytest.mark.parametrize("path,params", TELCO_ENDPOINTS)
def test_telco_endpoints_return_200_when_authenticated(client, telco_token, path, params):
    response = client.get(path, params=params, headers=_auth_headers(telco_token))
    assert response.status_code == 200, response.text


def test_health_endpoint_returns_200_unauthenticated():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_banking_model_metrics_and_importance_are_real_and_return_200(client, banking_token):
    metrics_response = client.get("/api/model/metrics", headers=_auth_headers(banking_token))
    assert metrics_response.status_code == 200
    assert "roc_auc" in metrics_response.json()

    importance_response = client.get("/api/model/importance", headers=_auth_headers(banking_token))
    assert importance_response.status_code == 200
    assert len(importance_response.json()) > 0


def test_banking_priority_and_backtest_return_not_validated(client, banking_token):
    priority_response = client.get("/api/priority", headers=_auth_headers(banking_token))
    assert priority_response.status_code == 200
    assert priority_response.json() == PRIORITY_BACKTEST_NOT_VALIDATED

    backtest_response = client.get("/api/backtest", headers=_auth_headers(banking_token))
    assert backtest_response.status_code == 200
    assert backtest_response.json() == PRIORITY_BACKTEST_NOT_VALIDATED


def test_banking_telco_only_analyses_return_not_trained(client, banking_token):
    for path in ("/api/survival/segments", "/api/segments", "/api/anomalies", "/api/clv/importance"):
        response = client.get(path, headers=_auth_headers(banking_token))
        assert response.status_code == 200, response.text
        assert response.json() == NOT_TRAINED
