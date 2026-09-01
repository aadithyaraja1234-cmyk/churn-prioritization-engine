from src.models.alerts import (
    HIGHEST_RISK_CLUSTER,
    evaluate_customer_alerts,
    evaluate_portfolio_alert,
    scan_for_alerts,
)

REVENUE_AT_RISK_THRESHOLD = 30.0

# Meridian Wireless: a real self-registered tenant. Since Stage 2b
# (src/models/tenant_training.py's _run_optional_modules()) segment_model.pkl/
# anomaly_model.pkl are now trained unconditionally for every tenant - only
# CLV genuinely stays missing here (no clv_column was mapped for Meridian's
# real upload). See docs/ADDING_A_TENANT.md.
MERIDIAN_MODEL_DIR = "models/meridian-wireless"
MERIDIAN_DATA_PATH = "data/tenant_uploads/meridian-wireless.csv"

# Fernwood Retail Collective: unlike Meridian (whose real id_column happens
# to be "customerID", a coincidence that meant the tests above never
# actually exercised scan_for_alerts()'s old hardcoded "customerID"/"CLTV"
# literals), Fernwood's confirmed mapping uses "cust_ref" as its id_column -
# genuinely different from Telco's literal, so this is a real regression
# guard for the bug fixed in this pass (KeyError: 'customerID' for any
# tenant whose id column isn't spelled that way).
FERNWOOD_MODEL_DIR = "models/fernwood-retail-collective"
FERNWOOD_DATA_PATH = "data/tenant_uploads/fernwood-retail-collective.csv"


def test_high_value_high_risk_rule_fires_on_matching_synthetic_customer():
    alerts = evaluate_customer_alerts(
        customer_id="SYNTH-001",
        churn_probability=0.85,  # > 0.7
        revenue_at_risk=5.0,  # below quartile threshold - irrelevant to this rule
        clv_percentile=95.0,  # top 20%
        cluster=0,  # not the highest-risk cluster - irrelevant to this rule
        is_anomaly=False,
        revenue_at_risk_top_quartile_threshold=REVENUE_AT_RISK_THRESHOLD,
    )

    types = {a["type"] for a in alerts}
    assert "high_value_high_risk" in types


def test_highest_risk_segment_high_revenue_rule_fires_on_matching_synthetic_customer():
    alerts = evaluate_customer_alerts(
        customer_id="SYNTH-002",
        churn_probability=0.1,  # low - irrelevant to this rule
        revenue_at_risk=100.0,  # >= threshold
        clv_percentile=10.0,  # not top 20% - irrelevant to this rule
        cluster=HIGHEST_RISK_CLUSTER,
        is_anomaly=False,
        revenue_at_risk_top_quartile_threshold=REVENUE_AT_RISK_THRESHOLD,
    )

    types = {a["type"] for a in alerts}
    assert "highest_risk_segment_high_revenue" in types


def test_anomaly_high_value_rule_fires_on_matching_synthetic_customer():
    alerts = evaluate_customer_alerts(
        customer_id="SYNTH-003",
        churn_probability=0.1,  # low - irrelevant to this rule
        revenue_at_risk=5.0,  # below threshold - irrelevant to this rule
        clv_percentile=90.0,  # top 20%
        cluster=0,  # not the highest-risk cluster - irrelevant to this rule
        is_anomaly=True,
        revenue_at_risk_top_quartile_threshold=REVENUE_AT_RISK_THRESHOLD,
    )

    types = {a["type"] for a in alerts}
    assert "anomaly_high_value" in types


def test_customer_matching_no_conditions_produces_no_alerts():
    alerts = evaluate_customer_alerts(
        customer_id="SYNTH-004",
        churn_probability=0.1,  # not > 0.7
        revenue_at_risk=1.0,  # below threshold
        clv_percentile=10.0,  # not top 20%
        cluster=0,  # not the highest-risk cluster
        is_anomaly=False,
        revenue_at_risk_top_quartile_threshold=REVENUE_AT_RISK_THRESHOLD,
    )

    assert alerts == []


def test_portfolio_alert_fires_above_threshold():
    alert = evaluate_portfolio_alert(total_revenue_at_risk=15_995.30, threshold=10_000.0)
    assert alert is not None
    assert alert["type"] == "portfolio_revenue_at_risk"
    assert alert["customer_id"] is None


def test_portfolio_alert_does_not_fire_below_threshold():
    alert = evaluate_portfolio_alert(total_revenue_at_risk=5_000.0, threshold=10_000.0)
    assert alert is None


def test_clv_dependent_rules_never_fire_when_clv_percentile_is_none():
    """clv_percentile=None means this tenant has no CLV data at all (not a
    genuine per-customer edge case) - high_value_high_risk and
    anomaly_high_value both require is_clv_top20, which must be False
    (never crash) when the signal simply doesn't exist for this tenant."""
    alerts = evaluate_customer_alerts(
        customer_id="SYNTH-005",
        churn_probability=0.95,  # would trigger high_value_high_risk if CLV were known
        revenue_at_risk=5.0,
        clv_percentile=None,
        cluster=0,
        is_anomaly=True,  # would trigger anomaly_high_value if CLV were known
        revenue_at_risk_top_quartile_threshold=REVENUE_AT_RISK_THRESHOLD,
    )
    types = {a["type"] for a in alerts}
    assert "high_value_high_risk" not in types
    assert "anomaly_high_value" not in types


def test_segment_rule_still_fires_when_clv_percentile_is_none():
    """A rule that doesn't depend on CLV (highest_risk_segment_high_revenue
    needs only cluster + revenue) must still fire even when this tenant has
    no CLV data - missing ONE signal shouldn't silently suppress alerts
    that never needed it."""
    alerts = evaluate_customer_alerts(
        customer_id="SYNTH-006",
        churn_probability=0.1,
        revenue_at_risk=100.0,  # >= threshold
        clv_percentile=None,
        cluster=HIGHEST_RISK_CLUSTER,
        is_anomaly=False,
        revenue_at_risk_top_quartile_threshold=REVENUE_AT_RISK_THRESHOLD,
    )
    types = {a["type"] for a in alerts}
    assert "highest_risk_segment_high_revenue" in types


def test_scan_for_alerts_on_clv_only_missing_tenant_does_not_crash():
    """End-to-end against Meridian's real artifacts (clv_data_path=None -
    the same value src.tenant_registry's Company-backed profile resolution
    actually passes for a self-registered tenant with no CLV data): must
    return a real list (not crash), restricted to alert types that don't
    need CLV specifically.

    NOTE: "highest_risk_segment_high_revenue" firing here relies on
    alerts.py's own HIGHEST_RISK_CLUSTER=2 constant, which is Telco's
    finalized k=4 clustering's own label (see alerts.py's comment) - it is
    NOT validated against Meridian's independently-fit segment_model.pkl
    (its own k=3 clustering, chosen by silhouette search - see
    src/models/tenant_training.py). Cluster index "2" carries no
    guaranteed meaning across two separately-fit KMeans models; this alert
    firing for Meridian is a real, reproducible behavior of the current
    code, not a claim that cluster 2 is actually Meridian's highest-risk
    segment. Generalizing HIGHEST_RISK_CLUSTER the way survival.py's
    segment_feature_column was auto-derived (Stage 2b) is real,
    unfinished follow-up work, same class of gap as business_impact.py/
    scenario.py's still-Telco-hardcoded logic - flagged, not silently
    fixed or silently hidden behind a narrower assertion."""
    alerts = scan_for_alerts(
        model_dir=MERIDIAN_MODEL_DIR,
        data_path=MERIDIAN_DATA_PATH,
        clv_data_path=None,
    )
    assert isinstance(alerts, list)
    fired_types = {a["type"] for a in alerts}
    assert fired_types <= {"portfolio_revenue_at_risk", "highest_risk_segment_high_revenue", "anomaly_high_value"}
    # high_value_high_risk specifically DOES need CLV (clv_percentile) - it
    # must never fire without real CLV data behind it.
    assert "high_value_high_risk" not in fired_types


def test_scan_for_alerts_loads_with_genuinely_different_column_names():
    """Regression test: scan_for_alerts() used to hardcode "customerID"/
    "CLTV" regardless of this tenant's own tenant_config, crashing with
    KeyError for any tenant (like Fernwood, id_column="cust_ref") whose
    columns aren't spelled that way. Must now return a real, non-empty
    alert list."""
    alerts = scan_for_alerts(model_dir=FERNWOOD_MODEL_DIR, data_path=FERNWOOD_DATA_PATH, clv_data_path=None)
    assert isinstance(alerts, list)
    assert len(alerts) > 0
    for alert in alerts:
        if alert["customer_id"] is not None:
            assert alert["customer_id"].startswith("FR-")
