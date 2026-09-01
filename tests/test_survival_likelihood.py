import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base
from src.data.clean import clean_data
from src.data.load import load_raw
from src.models.survival import CustomerNotFoundError, churn_likelihood_for_customer, churn_likelihood_within_window

MODEL_DIR = "models/v1"
DATA_PATH = "data/raw/telco.csv"


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
    email = f"survival-likelihood-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


# --- Pure module-level checks ---


@pytest.mark.parametrize("window_days", [7, 30, 90])
@pytest.mark.parametrize("contract", ["Month-to-month", "One year", "Two year"])
def test_churn_likelihood_within_window_is_a_valid_probability(contract, window_days):
    likelihood = churn_likelihood_within_window(contract, 18, window_days, model_dir=MODEL_DIR, data_path=DATA_PATH)
    assert 0.0 <= likelihood <= 1.0


def test_churn_likelihood_within_window_increases_with_window_size():
    likelihoods = [
        churn_likelihood_within_window("Month-to-month", 18, window_days, model_dir=MODEL_DIR, data_path=DATA_PATH)
        for window_days in (7, 30, 90)
    ]
    assert likelihoods == sorted(likelihoods)


def test_churn_likelihood_for_customer_returns_all_three_real_windows():
    df = clean_data(load_raw(DATA_PATH))
    customer_id = df.iloc[0]["customerID"]

    result = churn_likelihood_for_customer(customer_id, model_dir=MODEL_DIR, data_path=DATA_PATH)

    assert set(result.keys()) == {"7d", "30d", "90d"}
    for likelihood in result.values():
        assert 0.0 <= likelihood <= 1.0


def test_churn_likelihood_for_customer_raises_for_unknown_customer():
    with pytest.raises(CustomerNotFoundError):
        churn_likelihood_for_customer("NONEXISTENT-ID-0000", model_dir=MODEL_DIR, data_path=DATA_PATH)


# --- API-level checks ---


def test_survival_likelihood_endpoint_returns_real_probabilities(client):
    token = _get_token(client)
    df = clean_data(load_raw(DATA_PATH))
    customer_id = df.iloc[0]["customerID"]

    response = client.get(f"/api/survival/likelihood/{customer_id}", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"7d", "30d", "90d"}
    for likelihood in payload.values():
        assert 0.0 <= likelihood <= 1.0


def test_survival_likelihood_endpoint_returns_404_for_unknown_customer(client):
    token = _get_token(client)
    response = client.get(
        "/api/survival/likelihood/NONEXISTENT-ID-0000", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404


def test_survival_likelihood_endpoint_returns_real_data_for_banking_tenant(client):
    """Banking was fully trained (see config/config.yaml's banking entry) -
    survival genuinely passed its own real gate (c-index 0.716, using the
    real Tenure column). A telco-shaped fake id correctly 404s now (not
    found in banking's own, differently-shaped CustomerId population) -
    proved with a REAL banking customer id instead."""
    import pandas as pd

    df = pd.read_csv("data/raw/bank_churn.csv")
    real_banking_customer_id = str(df.iloc[0]["CustomerId"])

    token = _get_token(client, tenant_id="banking")
    response = client.get(
        f"/api/survival/likelihood/{real_banking_customer_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("available") is not False
    for value in body.values():
        assert 0.0 <= value <= 1.0


def test_survival_likelihood_endpoint_requires_authentication(client):
    response = client.get("/api/survival/likelihood/1234-ABCDE")
    assert response.status_code == 401
