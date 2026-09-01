"""Stage 1 integration tests: every copilot tool must return exactly what
calling its underlying src/models/*.py function directly returns - these
are thin wrappers, not new logic."""

import pytest

from src.copilot import tools
from src.models.business_impact import compute_business_impact_bulk
from src.models.alerts import scan_for_alerts
from src.models.budget_optimizer import optimize_budget
from src.models.explain import explain_customer
from src.models.health_score import compute_health_score
from src.models.recommend import recommend_action_for_customer
from src.models.scenario import run_scenario as run_scenario_module
from src.models.segment import load_cluster_profiles
from src.models.survival import churn_likelihood_for_customer

MODEL_DIR = "models/v1"
DATA_PATH = "data/raw/telco.csv"
CLV_MODEL_DIR = "models/v1_enriched"
CLV_DATA_PATH = "data/raw/telco_enriched.csv"

# Banking (config/config.yaml) used to be this test file's "everything
# disabled" reference tenant for the *_not_trained_for_banking tests below
# - it no longer is, now that Banking has been fully trained (business_
# impact_core/survival/segments/anomalies/priority_ranking/backtest/
# customer_timeline/scenario_simulator/budget_optimizer all genuinely
# passed their own real gates; only clv stays honestly off, and no tool
# here is clv-specific). UNREGISTERED_TENANT_ID has no matching profile at
# all (not in config.yaml's static tenants, and db=None on every one of
# these calls means no Company-backed fallback either), so every tool
# reports "unavailable" the same way a genuinely never-trained tenant
# would - a cleaner "nothing enabled" case than Banking ever was, since it
# doesn't rely on a specific tenant's config staying frozen forever.
UNREGISTERED_TENANT_ID = "totally-unregistered-tenant"


@pytest.fixture(scope="module")
def sample_customer_id():
    impact_df = compute_business_impact_bulk(model_dir=MODEL_DIR, data_path=DATA_PATH, clv_data_path=CLV_DATA_PATH)
    return impact_df.iloc[0]["customerID"]


# --- get_business_impact_summary ---


def test_get_business_impact_summary_matches_direct_call():
    result = tools.get_business_impact_summary("telco")
    impact_df = compute_business_impact_bulk(model_dir=MODEL_DIR, data_path=DATA_PATH, clv_data_path=CLV_DATA_PATH)

    assert result["total_revenue_at_risk"] == pytest.approx(float(impact_df["revenue_at_risk"].sum()))
    assert result["total_recoverable_revenue"] == pytest.approx(float(impact_df["recoverable_revenue"].sum()))
    assert result["source"] == "business_impact.py:compute_business_impact_bulk"


def test_get_business_impact_summary_not_trained_for_an_unregistered_tenant():
    result = tools.get_business_impact_summary(UNREGISTERED_TENANT_ID)
    assert result["available"] is False
    assert result["reason"] == "not yet trained for this tenant"


# --- get_top_opportunities ---


def test_get_top_opportunities_matches_direct_call():
    result = tools.get_top_opportunities("telco", limit=5)
    impact_df = compute_business_impact_bulk(model_dir=MODEL_DIR, data_path=DATA_PATH, clv_data_path=CLV_DATA_PATH)
    expected_ids = impact_df.head(5)["customerID"].tolist()

    assert len(result["customers"]) == 5
    assert [c["customerID"] for c in result["customers"]] == expected_ids
    assert result["customers"][0]["opportunity_score"] == pytest.approx(float(impact_df.iloc[0]["opportunity_score"]))


def test_get_top_opportunities_not_trained_for_an_unregistered_tenant():
    result = tools.get_top_opportunities(UNREGISTERED_TENANT_ID)
    assert result["available"] is False


# --- get_customer_detail ---


def test_get_customer_detail_matches_direct_calls(sample_customer_id):
    result = tools.get_customer_detail(sample_customer_id, "telco")

    explanation = explain_customer(sample_customer_id, model_dir=MODEL_DIR, data_path=DATA_PATH)
    health = compute_health_score(
        sample_customer_id,
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
        clv_model_dir=CLV_MODEL_DIR,
        clv_data_path=CLV_DATA_PATH,
    )
    recommendation = recommend_action_for_customer(
        sample_customer_id, model_dir=MODEL_DIR, data_path=DATA_PATH, clv_data_path=CLV_DATA_PATH
    )

    assert result["churn_probability"] == pytest.approx(explanation["churn_probability"])
    assert result["top_features"] == explanation["top_features"]
    assert result["health_score"] == pytest.approx(health["health_score"])
    assert result["recommended_action"] == recommendation["recommended_action"]
    assert result["triggered_by"] == recommendation["triggered_by"]

    # clv_component is deliberately renamed (not just passed through) - see
    # get_customer_detail()'s docstring: a bare "clv_component" invites an
    # LLM to misreport a 0-100 percentile rank as a dollar CLV figure.
    assert "clv_component" not in result["health_score_components"]
    assert result["health_score_components"]["clv_percentile_rank_component"] == pytest.approx(
        health["components"]["clv_component"]
    )
    for key in ("churn_component", "survival_component", "segment_component", "anomaly_component"):
        assert result["health_score_components"][key] == pytest.approx(health["components"][key])


def test_get_customer_detail_clv_field_name_is_unambiguous_about_units(sample_customer_id):
    """Guards against the specific fabrication risk Stage 2's LLM would
    otherwise be exposed to: a field literally named "clv" without a unit
    qualifier reads as a dollar amount. The percentile-rank value must
    also fall in [0, 100], not some unbounded/dollar-scale range."""
    result = tools.get_customer_detail(sample_customer_id, "telco")
    components = result["health_score_components"]

    assert "clv_value" not in components
    assert "predicted_clv" not in components
    assert "clv_component" not in components
    assert "clv_percentile_rank_component" in components
    assert 0.0 <= components["clv_percentile_rank_component"] <= 100.0


def test_get_customer_detail_unknown_customer_returns_error_not_fabrication():
    result = tools.get_customer_detail("NONEXISTENT-ID-0000", "telco")
    assert "error" in result
    assert "churn_probability" not in result


def test_get_customer_detail_not_trained_for_an_unregistered_tenant(sample_customer_id):
    result = tools.get_customer_detail(sample_customer_id, UNREGISTERED_TENANT_ID)
    assert result["available"] is False


# --- run_budget_optimization ---


def test_run_budget_optimization_matches_direct_call():
    result = tools.run_budget_optimization("telco", budget=1000.0, cost_override=50.0)
    direct = optimize_budget(
        budget_amount=1000.0,
        cost_per_intervention=50.0,
        model_dir=MODEL_DIR,
        data_path=DATA_PATH,
        clv_data_path=CLV_DATA_PATH,
    )

    assert result["n_customers_covered"] == direct["n_customers_covered"]
    assert result["total_cost"] == pytest.approx(direct["total_cost"])
    assert result["roi"] == pytest.approx(direct["roi"])
    assert [c["customer_id"] for c in result["selected_customers"]] == [
        c["customer_id"] for c in direct["selected_customers"]
    ]


def test_run_budget_optimization_not_trained_for_an_unregistered_tenant():
    result = tools.run_budget_optimization(UNREGISTERED_TENANT_ID, budget=1000.0)
    assert result["available"] is False


def test_run_budget_optimization_invalid_budget_returns_error_not_fabrication():
    result = tools.run_budget_optimization("telco", budget=-100.0)
    assert "error" in result


# --- run_scenario ---


def test_run_scenario_matches_direct_call():
    params = {"percent_change": 10}
    result = tools.run_scenario("telco", "uniform_charge_change", params)
    direct = run_scenario_module(
        "uniform_charge_change", params, scenario_name="direct test", model_dir=MODEL_DIR, data_path=DATA_PATH
    )

    assert result["total_revenue_at_risk"] == direct["total_revenue_at_risk"]
    assert result["customers_high_risk"] == direct["customers_high_risk"]
    assert result["net_change_in_recoverable_revenue"] == pytest.approx(direct["net_change_in_recoverable_revenue"])


def test_run_scenario_not_trained_for_an_unregistered_tenant():
    result = tools.run_scenario(UNREGISTERED_TENANT_ID, "uniform_charge_change", {"percent_change": 10})
    assert result["available"] is False


def test_run_scenario_invalid_type_returns_error_not_fabrication():
    result = tools.run_scenario("telco", "not_a_real_scenario_type", {})
    assert "error" in result


# --- get_segment_summary ---


def test_get_segment_summary_matches_direct_call():
    result = tools.get_segment_summary("telco")
    profiles = load_cluster_profiles(model_dir=MODEL_DIR, data_path=DATA_PATH)

    assert len(result["segments"]) == len(profiles)
    assert result["segments"] == profiles.to_dict(orient="records")


def test_get_segment_summary_not_trained_for_an_unregistered_tenant():
    result = tools.get_segment_summary(UNREGISTERED_TENANT_ID)
    assert result["available"] is False


# --- get_alerts ---


def test_get_alerts_matches_direct_call():
    result = tools.get_alerts("telco")
    direct = scan_for_alerts(model_dir=MODEL_DIR, data_path=DATA_PATH, clv_data_path=CLV_DATA_PATH)

    assert len(result["alerts"]) == len(direct)
    assert result["alerts"] == direct


def test_get_alerts_not_trained_for_an_unregistered_tenant():
    result = tools.get_alerts(UNREGISTERED_TENANT_ID)
    assert result["available"] is False


# --- get_survival_likelihood ---


def test_get_survival_likelihood_matches_direct_call(sample_customer_id):
    result = tools.get_survival_likelihood(sample_customer_id, "telco")
    direct = churn_likelihood_for_customer(sample_customer_id, model_dir=MODEL_DIR, data_path=DATA_PATH)

    assert result["churn_likelihood"] == direct
    for value in result["churn_likelihood"].values():
        assert 0.0 <= value <= 1.0


def test_get_survival_likelihood_unknown_customer_returns_error_not_fabrication():
    result = tools.get_survival_likelihood("NONEXISTENT-ID-0000", "telco")
    assert "error" in result
    assert "churn_likelihood" not in result


def test_get_survival_likelihood_not_trained_for_an_unregistered_tenant(sample_customer_id):
    result = tools.get_survival_likelihood(sample_customer_id, UNREGISTERED_TENANT_ID)
    assert result["available"] is False


# --- Every tool must self-identify its source ---


@pytest.mark.parametrize(
    "call,expected_source",
    [
        (lambda: tools.get_business_impact_summary("telco"), "business_impact.py:compute_business_impact_bulk"),
        (lambda: tools.get_top_opportunities("telco"), "business_impact.py:compute_business_impact_bulk"),
        (lambda: tools.get_segment_summary("telco"), "segment.py:load_cluster_profiles"),
        (lambda: tools.get_alerts("telco"), "alerts.py:scan_for_alerts"),
        (lambda: tools.run_budget_optimization("telco", 1000.0), "budget_optimizer.py:optimize_budget"),
        (
            lambda: tools.run_scenario("telco", "uniform_charge_change", {"percent_change": 5}),
            "scenario.py:run_scenario",
        ),
    ],
)
def test_tool_reports_its_real_source(call, expected_source):
    result = call()
    assert result["source"] == expected_source
