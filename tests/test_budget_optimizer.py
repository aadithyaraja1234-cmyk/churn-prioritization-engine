import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base
from src.models.business_impact import compute_business_impact_bulk
from src.models.budget_optimizer import DEFAULT_COST_PER_INTERVENTION, InvalidBudgetError, optimize_budget

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
    email = f"budget-optimizer-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


# --- Pure module-level checks ---


def test_selected_customers_are_correct_topn_by_opportunity_score():
    n = 5
    cost = 100.0
    result = optimize_budget(budget_amount=n * cost, cost_per_intervention=cost, model_dir=MODEL_DIR, data_path=DATA_PATH)

    impact_df = compute_business_impact_bulk(model_dir=MODEL_DIR, data_path=DATA_PATH)
    expected_top_n_ids = impact_df.head(n)["customerID"].tolist()

    assert [row["customer_id"] for row in result["selected_customers"]] == expected_top_n_ids
    assert result["n_customers_covered"] == n
    # Confirm descending order by opportunity_score within the selection too.
    scores = [row["opportunity_score"] for row in result["selected_customers"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("budget_amount", [1.0, 74.0, 75.0, 1000.0, 50000.0])
def test_total_cost_never_exceeds_budget_amount(budget_amount):
    result = optimize_budget(budget_amount=budget_amount, model_dir=MODEL_DIR, data_path=DATA_PATH)
    assert result["total_cost"] <= budget_amount


def test_roi_calculation_correct_given_known_inputs():
    n = 10
    cost = 50.0
    result = optimize_budget(budget_amount=n * cost, cost_per_intervention=cost, model_dir=MODEL_DIR, data_path=DATA_PATH)

    total_recoverable = sum(row["recoverable_revenue"] for row in result["selected_customers"])
    expected_total_cost = n * cost
    expected_roi = (total_recoverable - expected_total_cost) / expected_total_cost

    assert result["total_cost"] == pytest.approx(expected_total_cost)
    assert result["total_recoverable_revenue"] == pytest.approx(total_recoverable)
    assert result["roi"] == pytest.approx(expected_roi)


def test_default_cost_per_intervention_used_when_not_provided():
    result = optimize_budget(budget_amount=1000.0, model_dir=MODEL_DIR, data_path=DATA_PATH)
    assert result["cost_per_intervention"] == DEFAULT_COST_PER_INTERVENTION
    assert result["cost_per_intervention_is_default"] is True


def test_custom_cost_per_intervention_overrides_default():
    result = optimize_budget(budget_amount=1000.0, cost_per_intervention=200.0, model_dir=MODEL_DIR, data_path=DATA_PATH)
    assert result["cost_per_intervention"] == 200.0
    assert result["cost_per_intervention_is_default"] is False


def test_pct_of_at_risk_revenue_covered_in_valid_range():
    result = optimize_budget(budget_amount=10000.0, model_dir=MODEL_DIR, data_path=DATA_PATH)
    assert 0.0 <= result["pct_of_at_risk_revenue_covered"] <= 1.0


def test_metadata_includes_cost_assumption_and_hillstrom_citation():
    result = optimize_budget(budget_amount=1000.0, model_dir=MODEL_DIR, data_path=DATA_PATH)
    metadata = result["metadata"]
    assert "cost_per_intervention_note" in metadata
    assert "no real cost data exists" in metadata["cost_per_intervention_note"]
    assert "intervention_success_rate" in metadata
    assert "hillstrom_benchmark" in metadata
    assert "minethatdata.com" in metadata["hillstrom_benchmark"]["source_url"]


@pytest.mark.parametrize("budget_amount", [0.0, -100.0])
def test_invalid_budget_amount_raises(budget_amount):
    with pytest.raises(InvalidBudgetError):
        optimize_budget(budget_amount=budget_amount, model_dir=MODEL_DIR, data_path=DATA_PATH)


@pytest.mark.parametrize("cost", [0.0, -50.0])
def test_invalid_cost_per_intervention_raises(cost):
    with pytest.raises(InvalidBudgetError):
        optimize_budget(budget_amount=1000.0, cost_per_intervention=cost, model_dir=MODEL_DIR, data_path=DATA_PATH)


def test_budget_too_small_for_even_one_intervention_returns_empty_selection():
    result = optimize_budget(budget_amount=10.0, cost_per_intervention=75.0, model_dir=MODEL_DIR, data_path=DATA_PATH)
    assert result["selected_customers"] == []
    assert result["n_customers_covered"] == 0
    assert result["total_cost"] == 0.0
    assert result["roi"] == 0.0


def test_stop_at_positive_roi_skips_early_low_value_customer_instead_of_halting(monkeypatch):
    # Regression test for the "stop at first failure" bug: opportunity_score
    # ranks by revenue_at_risk * ease_of_saving * clv_percentile_weight, not
    # by recoverable_revenue, so an uneconomical customer can rank ahead of
    # several economically-fine ones. A synthetic ranked list makes this
    # deterministic: LOW1 outranks three HIGH customers but doesn't clear the
    # cost bar. The correct behavior is to skip LOW1 and select the three
    # HIGH customers, not to halt and return nothing.
    synthetic_df = pd.DataFrame(
        [
            {
                "customerID": "LOW1",
                "opportunity_score": 100.0,
                "recoverable_revenue": 5.0,
                "churn_probability": 0.9,
                "revenue_at_risk": 50.0,
            },
            {
                "customerID": "HIGH1",
                "opportunity_score": 90.0,
                "recoverable_revenue": 50.0,
                "churn_probability": 0.8,
                "revenue_at_risk": 200.0,
            },
            {
                "customerID": "HIGH2",
                "opportunity_score": 80.0,
                "recoverable_revenue": 60.0,
                "churn_probability": 0.7,
                "revenue_at_risk": 220.0,
            },
            {
                "customerID": "HIGH3",
                "opportunity_score": 70.0,
                "recoverable_revenue": 40.0,
                "churn_probability": 0.6,
                "revenue_at_risk": 150.0,
            },
        ]
    )
    monkeypatch.setattr("src.models.budget_optimizer.compute_business_impact_bulk", lambda **kwargs: synthetic_df)
    monkeypatch.setattr(
        "src.models.budget_optimizer.recommend_action_for_customer",
        lambda customer_id, **kwargs: {"recommended_action": "stub action"},
    )

    cost = 10.0
    result = optimize_budget(budget_amount=3 * cost, cost_per_intervention=cost, stop_at_positive_roi=True)

    selected_ids = [row["customer_id"] for row in result["selected_customers"]]
    assert selected_ids == ["HIGH1", "HIGH2", "HIGH3"]
    assert "LOW1" not in selected_ids
    assert result["n_customers_covered"] == 3
    assert result["total_cost"] == pytest.approx(30.0)
    assert result["total_recoverable_revenue"] == pytest.approx(150.0)
    assert result["budget_fully_utilized"] is True
    assert result["unspent_budget"] == pytest.approx(0.0)


def test_stop_at_positive_roi_never_includes_uneconomical_customer():
    # $10, not the $75 default: opportunity_score ranks by revenue_at_risk *
    # ease_of_saving * clv_percentile_weight, not by recoverable_revenue, so
    # the two orderings aren't monotonic with each other - some customer
    # ranked ahead of many economically-fine ones dips below even a low $10
    # bar. A low cost keeps the qualifying pool non-trivial so this test
    # exercises real, non-mocked data rather than vacuously passing.
    cost = 10.0
    result = optimize_budget(
        budget_amount=50000.0,
        cost_per_intervention=cost,
        stop_at_positive_roi=True,
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    assert result["n_customers_covered"] > 0
    for row in result["selected_customers"]:
        assert row["recoverable_revenue"] > cost


def test_stop_at_positive_roi_leaves_budget_unspent_and_flags_it():
    cost = 10.0
    result = optimize_budget(
        budget_amount=50000.0,
        cost_per_intervention=cost,
        stop_at_positive_roi=True,
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
    )
    assert result["budget_fully_utilized"] is False
    assert result["unspent_budget"] == pytest.approx(50000.0 - result["total_cost"])
    assert result["unspent_budget"] > 0


def test_stop_at_positive_roi_false_by_default_and_can_spend_uneconomical_customers():
    cost = 75.0
    result = optimize_budget(budget_amount=50000.0, cost_per_intervention=cost, model_dir=MODEL_DIR, data_path=DATA_PATH)
    assert result["stop_at_positive_roi"] is False
    uneconomical = [row for row in result["selected_customers"] if row["recoverable_revenue"] <= cost]
    assert len(uneconomical) > 0
    assert result["budget_fully_utilized"] is True


# --- API-level checks ---


def test_budget_optimizer_endpoint_returns_real_results(client):
    token = _get_token(client)
    response = client.post(
        "/api/budget-optimizer",
        json={"budget_amount": 10000, "cost_per_intervention": None},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_customers_covered"] > 0
    assert len(payload["selected_customers"]) == payload["n_customers_covered"]
    assert payload["selected_customers"][0]["recommended_action"]


def test_budget_optimizer_endpoint_returns_400_for_invalid_budget(client):
    token = _get_token(client)
    response = client.post(
        "/api/budget-optimizer",
        json={"budget_amount": -5},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


def test_budget_optimizer_returns_real_allocation_for_banking_tenant(client):
    """Banking was fully trained (see config/config.yaml's banking entry) -
    budget_optimizer genuinely passed its own real sanity gate against
    Balance as the revenue proxy. Real allocation, not "not yet trained"."""
    token = _get_token(client, tenant_id="banking")
    response = client.post(
        "/api/budget-optimizer",
        json={"budget_amount": 1000},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("available") is not False
    assert "n_customers_covered" in body


def test_budget_optimizer_requires_authentication(client):
    response = client.post("/api/budget-optimizer", json={"budget_amount": 1000})
    assert response.status_code == 401
