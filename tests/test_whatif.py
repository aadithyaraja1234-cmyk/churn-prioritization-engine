import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base
from src.models.whatif import simulate_whatif


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


def _get_token(client):
    client.post(
        "/auth/register",
        json={"email": "whatif-user@example.com", "password": "pw-123456", "tenant_id": "telco", "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": "whatif-user@example.com", "password": "pw-123456"})
    return response.json()["access_token"]


def _month_to_month_customer_id():
    df = pd.read_csv("data/raw/telco.csv")
    return df[df["Contract"] == "Month-to-month"].iloc[0]["customerID"]


def test_two_year_contract_meaningfully_decreases_churn_probability(client):
    token = _get_token(client)
    customer_id = _month_to_month_customer_id()

    response = client.post(
        "/api/whatif",
        json={"customer_id": customer_id, "overrides": {"Contract": "Two year"}},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["new_probability"] < payload["original_probability"]
    assert payload["delta"] < -0.05


def test_invalid_feature_key_returns_400(client):
    token = _get_token(client)
    customer_id = _month_to_month_customer_id()

    response = client.post(
        "/api/whatif",
        json={"customer_id": customer_id, "overrides": {"NotARealColumn": "whatever"}},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400


def test_invalid_categorical_value_returns_400_with_valid_options(client):
    token = _get_token(client)
    customer_id = _month_to_month_customer_id()

    response = client.post(
        "/api/whatif",
        json={"customer_id": customer_id, "overrides": {"Contract": "Century-long"}},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "valid_values" in detail
    assert "Two year" in detail["valid_values"]
    assert "Month-to-month" in detail["valid_values"]


# --- Regression: simulate_whatif() used to be hardcoded to Telco's literal
# "customerID"/"Churn" column names and assumed predict_proba()'s index 1
# was always the positive class - both crash/misbehave for any tenant whose
# columns or label spelling differ (the same recurring bug class this
# project has hit at least three times before - see docs/ADDING_A_TENANT.md).
# Self-contained (trains its own tiny synthetic tenant in tmp_path) so this
# runs anywhere, CI included - mirrors tests/test_recommend.py's equivalent
# synthetic regression test. ---


def test_whatif_for_a_freshly_trained_tenant_with_non_telco_column_names(tmp_path):
    import numpy as np

    from src.config import load_config
    from src.models.train import train_model

    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config" / "config.yaml"

    rng = np.random.default_rng(13)
    n = 320
    tenure_like = rng.integers(0, 60, size=n)
    fee = np.round(rng.normal(20, 5, size=n).clip(5, 40), 2)
    plan = rng.choice(["Alpha", "Beta"], size=n)
    logit = -0.5 - 0.02 * tenure_like + 0.9 * (plan == "Alpha") + rng.normal(0, 0.6, size=n)
    left = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame(
        {
            "acct_id": [f"A{i:04d}" for i in range(n)],
            "months_active": tenure_like,
            "fee_amount": fee,
            "plan_name": plan,
            "left": left,
        }
    )
    data_path = tmp_path / "synthetic.csv"
    df.to_csv(data_path, index=False)

    tenant_config = {
        "id_column": "acct_id",
        "target_column": "left",
        "target_positive_value": "Yes",
        "revenue_column": "fee_amount",
        "tuning_enabled": False,  # keep this test fast - not what's under test here
    }
    model_dir = tmp_path / "model"
    train_model(data_path, CONFIG_PATH, output_dir=model_dir, tenant_config=tenant_config)

    split_info = json.loads((model_dir / "split_indices.json").read_text(encoding="utf-8"))
    test_customer_id = df.loc[split_info["test_idx"][0], "acct_id"]

    result = simulate_whatif(
        test_customer_id,
        {"plan_name": "Beta"},  # "Alpha" was designed to be the higher-risk plan
        model_dir=model_dir,
        data_path=data_path,
    )
    assert result["customer_id"] == test_customer_id
    assert 0.0 <= result["original_probability"] <= 1.0
    assert 0.0 <= result["new_probability"] <= 1.0
    assert result["delta"] == pytest.approx(result["new_probability"] - result["original_probability"])
