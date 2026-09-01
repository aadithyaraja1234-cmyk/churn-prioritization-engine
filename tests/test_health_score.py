import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base
from src.models.health_score import compute_health_score
from src.models.prioritize import get_priority_ranking

EXPECTED_COMPONENT_KEYS = {
    "churn_component",
    "survival_component",
    "segment_component",
    "anomaly_component",
    "clv_component",
}

# Aurora Streaming: a real self-registered tenant with NO survival model
# (no duration role was ever mapped) and a genuinely different id_column
# ("customer_id", not "customerID") and clv_column ("lifetime_value_estimate",
# not "CLTV") - the real regression case for both bugs fixed in this pass:
# (1) compute_health_score() used to hardcode "customerID"/"CLTV"/index-1
# predict_proba() regardless of tenant_config, and (2) it unconditionally
# loaded survival_model.pkl, crashing outright for any tenant missing one
# instead of dropping that component the way business_impact.py/recommend.py
# already do for their own optional signals.
AURORA_MODEL_DIR = "models/aurora-streaming"
AURORA_DATA_PATH = "data/tenant_uploads/aurora-streaming.csv"


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
        json={"email": "health-user@example.com", "password": "pw-123456", "tenant_id": "telco", "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": "health-user@example.com", "password": "pw-123456"})
    return response.json()["access_token"]


def _extreme_customer_ids():
    ranking = get_priority_ranking(strategy="probability_only", model_dir="models/v1", data_path="data/raw/telco.csv")
    high_risk_id = ranking.iloc[0]["customerID"]
    low_risk_id = ranking.iloc[-1]["customerID"]
    return high_risk_id, low_risk_id


def test_health_score_always_between_0_and_100(client):
    token = _get_token(client)
    high_risk_id, low_risk_id = _extreme_customer_ids()

    for customer_id in (high_risk_id, low_risk_id):
        response = client.get(f"/api/health-score/{customer_id}", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        score = response.json()["health_score"]
        assert 0 <= score <= 100


def test_high_risk_customer_scores_lower_than_low_risk_customer(client):
    token = _get_token(client)
    high_risk_id, low_risk_id = _extreme_customer_ids()

    high_response = client.get(f"/api/health-score/{high_risk_id}", headers={"Authorization": f"Bearer {token}"})
    low_response = client.get(f"/api/health-score/{low_risk_id}", headers={"Authorization": f"Bearer {token}"})

    assert high_response.json()["health_score"] < low_response.json()["health_score"]


def test_response_includes_all_5_components_not_just_total(client):
    token = _get_token(client)
    high_risk_id, _ = _extreme_customer_ids()

    response = client.get(f"/api/health-score/{high_risk_id}", headers={"Authorization": f"Bearer {token}"})
    payload = response.json()

    assert "health_score" in payload
    assert set(payload["components"].keys()) == EXPECTED_COMPONENT_KEYS
    assert "weights" in payload


def test_health_score_for_a_tenant_with_no_survival_model_and_different_columns():
    """See AURORA_MODEL_DIR's comment above for exactly which two bugs this
    guards against."""
    import pandas as pd

    df = pd.read_csv(AURORA_DATA_PATH)
    customer_id = df.iloc[0]["customer_id"]

    result = compute_health_score(customer_id, model_dir=AURORA_MODEL_DIR, data_path=AURORA_DATA_PATH, clv_model_dir=AURORA_MODEL_DIR, clv_data_path=AURORA_DATA_PATH)

    assert 0 <= result["health_score"] <= 100
    # survival_component must be genuinely ABSENT (not a faked/neutral 50) -
    # Aurora has no survival_model.pkl at all.
    assert "survival_component" not in result["components"]
    assert "survival_component" not in result["components_used"]
    assert set(result["components_used"]) == {"churn_component", "segment_component", "anomaly_component", "clv_component"}
    # The reported weights must actually reconcile with health_score - see
    # compute_health_score()'s renormalization comment.
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    manual_recompute = sum(result["weights"][name] * result["components"][name] for name in result["components_used"])
    assert manual_recompute == pytest.approx(result["health_score"], abs=0.01)
