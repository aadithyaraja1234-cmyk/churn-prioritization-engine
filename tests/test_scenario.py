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
from src.models.scenario import InvalidScenarioError, run_scenario

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
    email = f"scenario-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


# --- Pure module-level checks (no API/DB involved) ---


def test_price_increase_increases_total_revenue_at_risk():
    result = run_scenario(
        "uniform_charge_change",
        {"percent_change": 10},
        "10pct price increase",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    assert result["total_revenue_at_risk"]["after"] > result["total_revenue_at_risk"]["before"]


def test_price_decrease_decreases_total_revenue_at_risk():
    result = run_scenario(
        "uniform_charge_change",
        {"percent_change": -10},
        "10pct price decrease",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    assert result["total_revenue_at_risk"]["after"] < result["total_revenue_at_risk"]["before"]


def test_favorable_contract_migration_decreases_high_risk_count():
    result = run_scenario(
        "contract_migration",
        {"from_contract": "Month-to-month", "to_contract": "Two year", "migration_rate": 0.5},
        "M2M to Two-year migration",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    assert result["customers_high_risk"]["after"] < result["customers_high_risk"]["before"]
    assert result["customers_high_risk"]["delta"] < 0
    assert result["affected_customer_count"] > 0


def test_per_segment_breakdown_sums_to_portfolio_total():
    result = run_scenario(
        "uniform_charge_change",
        {"percent_change": 15},
        "sum check",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    segment_before_sum = sum(row["before_revenue_at_risk"] for row in result["per_segment_breakdown"])
    segment_after_sum = sum(row["after_revenue_at_risk"] for row in result["per_segment_breakdown"])
    segment_customer_count_sum = sum(row["customer_count"] for row in result["per_segment_breakdown"])

    assert segment_before_sum == pytest.approx(result["total_revenue_at_risk"]["before"])
    assert segment_after_sum == pytest.approx(result["total_revenue_at_risk"]["after"])

    high_risk_before_sum = sum(row["before_high_risk_count"] for row in result["per_segment_breakdown"])
    high_risk_after_sum = sum(row["after_high_risk_count"] for row in result["per_segment_breakdown"])
    assert high_risk_before_sum == result["customers_high_risk"]["before"]
    assert high_risk_after_sum == result["customers_high_risk"]["after"]
    assert segment_customer_count_sum > 0


def test_discount_offer_decreases_average_monthly_charges():
    result = run_scenario(
        "discount_offer",
        {"discount_percent": 20, "target_segment": None},
        "20pct discount to everyone",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    # A blanket discount lowers billed revenue even where it also lowers
    # churn risk, so revenue-at-risk (probability * charges) should fall.
    assert result["total_revenue_at_risk"]["after"] < result["total_revenue_at_risk"]["before"]
    assert result["affected_customer_count"] > 0


def test_discount_offer_targeted_at_segment_affects_fewer_customers_than_all():
    all_customers_result = run_scenario(
        "discount_offer",
        {"discount_percent": 20, "target_segment": None},
        "discount all",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    segment_result = run_scenario(
        "discount_offer",
        {"discount_percent": 20, "target_segment": 2},
        "discount segment 2",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    assert 0 < segment_result["affected_customer_count"] < all_customers_result["affected_customer_count"]


def test_discount_offer_invalid_target_segment_raises():
    with pytest.raises(InvalidScenarioError):
        run_scenario(
            "discount_offer",
            {"discount_percent": 10, "target_segment": 999},
            "bad segment",
            model_dir=MODEL_DIR,
            data_path=DATA_PATH,
        )


def test_loyalty_program_decreases_average_churn_probability():
    result = run_scenario(
        "loyalty_program",
        {"adoption_rate": 1.0},
        "full adoption loyalty program",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    # 100% adoption at a flat 0.85x churn_probability multiplier for
    # everyone should strictly decrease total revenue-at-risk (probability
    # * unchanged charges), since no feature/charges changed.
    assert result["total_revenue_at_risk"]["after"] < result["total_revenue_at_risk"]["before"]
    assert result["affected_customer_count"] > 0


def test_loyalty_program_metadata_includes_illustrative_label():
    result = run_scenario(
        "loyalty_program",
        {"adoption_rate": 0.5},
        "half adoption",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    note = result["metadata"].get("loyalty_program_note", "")
    assert "illustrative" in note.lower()
    assert "no real loyalty-program outcome data" in note.lower()
    assert "15%" in note


def test_loyalty_program_zero_adoption_leaves_probabilities_unchanged():
    result = run_scenario(
        "loyalty_program",
        {"adoption_rate": 0.0},
        "no adoption",
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    assert result["affected_customer_count"] == 0
    assert result["total_revenue_at_risk"]["after"] == pytest.approx(result["total_revenue_at_risk"]["before"])


def test_invalid_scenario_type_raises():
    with pytest.raises(InvalidScenarioError):
        run_scenario("not_a_real_type", {}, "bad", model_dir=MODEL_DIR, data_path=DATA_PATH)


def test_missing_params_raises():
    with pytest.raises(InvalidScenarioError):
        run_scenario("uniform_charge_change", {}, "missing params", model_dir=MODEL_DIR, data_path=DATA_PATH)


# --- API-level checks ---


def test_post_scenario_persists_and_returns_full_result(client):
    token = client and _get_token(client)
    response = client.post(
        "/api/scenario",
        json={
            "scenario_type": "uniform_charge_change",
            "params": {"percent_change": 5},
            "scenario_name": "5pct increase",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "id" in payload
    assert payload["scenario_name"] == "5pct increase"
    assert "summary" in payload
    assert "metadata" in payload


def test_post_discount_offer_scenario_via_api(client):
    token = _get_token(client)
    response = client.post(
        "/api/scenario",
        json={
            "scenario_type": "discount_offer",
            "params": {"discount_percent": 15, "target_segment": None},
            "scenario_name": "api discount test",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["scenario_type"] == "discount_offer"


def test_post_loyalty_program_scenario_via_api(client):
    token = _get_token(client)
    response = client.post(
        "/api/scenario",
        json={
            "scenario_type": "loyalty_program",
            "params": {"adoption_rate": 0.4},
            "scenario_name": "api loyalty test",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario_type"] == "loyalty_program"
    assert "loyalty_program_note" in payload["metadata"]


def test_post_scenario_invalid_params_returns_400(client):
    token = _get_token(client)
    response = client.post(
        "/api/scenario",
        json={"scenario_type": "uniform_charge_change", "params": {}, "scenario_name": "bad"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


def test_scenario_history_is_tenant_scoped(client):
    telco_token = _get_token(client, tenant_id="telco")
    banking_token = _get_token(client, tenant_id="banking")

    client.post(
        "/api/scenario",
        json={
            "scenario_type": "uniform_charge_change",
            "params": {"percent_change": 8},
            "scenario_name": "telco-only scenario",
        },
        headers={"Authorization": f"Bearer {telco_token}"},
    )

    telco_history = client.get("/api/scenario/history", headers={"Authorization": f"Bearer {telco_token}"})
    assert telco_history.status_code == 200
    assert len(telco_history.json()) == 1
    assert telco_history.json()[0]["scenario_name"] == "telco-only scenario"

    # Banking was fully trained (see config/config.yaml's banking entry) -
    # scenario_simulator genuinely passed its own gate, so this is now a
    # real tenant-scoped query, not an "unavailable" shape - proves banking
    # sees ZERO of telco's scenarios (real isolation), not "not trained".
    banking_response = client.get("/api/scenario/history", headers={"Authorization": f"Bearer {banking_token}"})
    assert banking_response.status_code == 200
    assert banking_response.json() == []


def test_scenario_detail_not_accessible_across_tenants(client):
    telco_token = _get_token(client, tenant_id="telco")
    banking_token = _get_token(client, tenant_id="banking")

    create_response = client.post(
        "/api/scenario",
        json={
            "scenario_type": "uniform_charge_change",
            "params": {"percent_change": 12},
            "scenario_name": "cross-tenant check",
        },
        headers={"Authorization": f"Bearer {telco_token}"},
    )
    scenario_id = create_response.json()["id"]

    own_tenant_response = client.get(
        f"/api/scenario/{scenario_id}", headers={"Authorization": f"Bearer {telco_token}"}
    )
    assert own_tenant_response.status_code == 200

    # Banking was fully trained (see config/config.yaml's banking entry) -
    # scenario_simulator genuinely passed its own gate, so this now reaches
    # the real tenant-scoped DB lookup and correctly 404s (telco's scenario
    # id genuinely doesn't exist in banking's own scenarios table), rather
    # than being gated off before it ever got there.
    other_tenant_response = client.get(
        f"/api/scenario/{scenario_id}", headers={"Authorization": f"Bearer {banking_token}"}
    )
    assert other_tenant_response.status_code == 404


def test_scenario_detail_requires_authentication(client):
    response = client.get("/api/scenario/1")
    assert response.status_code == 401


# --- Regression: generalized to accept tenant_config (same recurring
# hardcoded-Telco-column bug this session already found in business_impact.py/
# recommend.py/backtest.py). ---

AURORA_MODEL_DIR = "models/aurora-streaming"
AURORA_DATA_PATH = "data/tenant_uploads/aurora-streaming.csv"


def test_aurora_uniform_charge_change_loads_with_genuinely_different_column_names():
    """uniform_charge_change only needs id_column/revenue_column, both of
    which Aurora has - id/revenue columns (customer_id/monthly_fee) share
    NOT ONE literal name with Telco's (customerID/MonthlyCharges). Used to
    crash outright; must now return a real result."""
    result = run_scenario(
        "uniform_charge_change",
        {"percent_change": 10},
        "aurora regression check",
        model_dir=AURORA_MODEL_DIR,
        data_path=AURORA_DATA_PATH,
    )
    assert result["customers_scored"] > 0
    assert result["total_revenue_at_risk"]["after"] != result["total_revenue_at_risk"]["before"]


def test_aurora_contract_migration_fails_honestly_with_no_segment_column():
    """Aurora never mapped a 'duration' role during onboarding (see
    tests/test_business_impact.py's module comment for the full story), so
    survival never ran and no segment/contract-equivalent column was ever
    resolved. contract_migration structurally cannot work without one -
    must raise a clear, specific error, not a KeyError deep in pandas."""
    from src.models.scenario import NoSegmentFeatureColumnError

    with pytest.raises(NoSegmentFeatureColumnError):
        run_scenario(
            "contract_migration",
            {"from_contract": "x", "to_contract": "y", "migration_rate": 0.1},
            "aurora regression check",
            model_dir=AURORA_MODEL_DIR,
            data_path=AURORA_DATA_PATH,
        )


def test_scenario_for_a_freshly_trained_tenant_with_non_telco_column_names(tmp_path):
    """Self-contained variant (trains its own tiny synthetic tenant, WITH a
    resolved segment_feature_column via a real survival fit, in tmp_path)
    so this specific guard - including contract_migration's positive path,
    which no currently-real self-registered tenant exercises (none has
    mapped a duration column yet) - runs anywhere, CI included."""
    import numpy as np

    from src.config import load_config
    from src.models.survival import train_survival_model
    from src.models.train import train_model

    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config" / "config.yaml"

    rng = np.random.default_rng(17)
    n = 400
    tenure_like = rng.integers(0, 60, size=n)
    fee = np.round(rng.normal(20, 5, size=n).clip(5, 40), 2)
    # Alphabetical-order-aligned labels (same trick as this session's
    # live-demo dataset generators) so the fitted Cox model's hazard ratio
    # for this column is real and dominant, not washed out by revenue
    # collinearity - see scripts/generate_live_demo_batch.py's own
    # docstring for the full mechanism.
    plan = rng.choice(["Alpha", "Beta"], size=n, p=[0.5, 0.5])
    logit = -0.3 - 0.02 * tenure_like + 0.9 * (plan == "Alpha") + rng.normal(0, 0.6, size=n)
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
        "duration_column": "months_active",
        "tuning_enabled": False,
    }
    model_dir = tmp_path / "model"
    train_model(data_path, CONFIG_PATH, output_dir=model_dir, tenant_config=tenant_config)
    from src.models.segment import run_segmentation

    run_segmentation(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    survival_metrics = train_survival_model(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    assert "coefficients" in survival_metrics  # confirms the fit actually happened

    uniform_result = run_scenario(
        "uniform_charge_change", {"percent_change": -10}, "synthetic uniform", model_dir=model_dir, data_path=data_path
    )
    assert uniform_result["customers_scored"] > 0

    migration_result = run_scenario(
        "contract_migration",
        {"from_contract": "Alpha", "to_contract": "Beta", "migration_rate": 0.5},
        "synthetic migration",
        model_dir=model_dir,
        data_path=data_path,
    )
    assert migration_result["affected_customer_count"] > 0
