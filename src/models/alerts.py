"""Alerts — pure conditional rules over already-computed outputs.

No new modeling: every signal here (churn probability, CLV, segment
assignment, anomaly flag, revenue-at-risk) is read from existing saved
model artifacts via functions already built elsewhere in this project.
Scanning is restricted to the test set, since that's the population
churn_probability/revenue_at_risk are actually validated on (see
prioritize.py). CLV percentile is computed against the full customer
population (same convention as health_score.py).

The rule logic itself (evaluate_customer_alerts/evaluate_portfolio_alert) is
kept separate from scan_for_alerts()'s real-data orchestration, so the rules
can be tested against synthetic, fully-controlled inputs rather than real
model output (which can't be dictated to hit a specific condition on demand).

Generalized to accept tenant_config (same pattern as prioritize.py/
business_impact.py/recommend.py/health_score.py - read back from
model_dir/split_indices.json via src/data/split.py's load_tenant_config()
when omitted). segment_model.pkl/anomaly_model.pkl/CLV data may not exist
for a tenant that's only had the core classifier trained (same graceful-
degradation pattern as business_impact.py's compute_business_impact_bulk()) -
each missing signal is skipped rather than crashing, exactly as this
module's own docstring on scan_for_alerts() already described in principle
(it just wasn't actually true yet: every id/CLV-column lookup below used to
be hardcoded to Telco's literal "customerID"/"CLTV" regardless of what this
tenant's own tenant_config said, which crashes for any tenant whose columns
are named differently - the third confirmed instance of this project's
recurring hardcoded-column bug, alongside health_score.py/whatif.py in this
same pass).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_tenant_config
from src.features.encode import transform_categorical_features
from src.models.anomaly import load_flagged_anomalies
from src.models.prioritize import get_priority_ranking

# Cluster 2 = "New, high-spend, flexible contract — highest risk" (45.9%
# churn rate) from segment.py's CLUSTER_LABELS - the finalized k=4 model.
HIGHEST_RISK_CLUSTER = 2

CLV_TOP_PERCENTILE_CUTOFF = 80.0  # "top 20%" -> percentile rank >= 80
REVENUE_AT_RISK_TOP_QUARTILE = 0.75

# Observed total top-20% revenue-at-risk on the real test set is ~$15,995;
# $10,000 is below that on purpose, so this alert is expected to fire under
# real conditions rather than sit dormant - adjust upward if that changes.
DEFAULT_PORTFOLIO_THRESHOLD = 10_000.0


def evaluate_customer_alerts(
    customer_id: str,
    churn_probability: float,
    revenue_at_risk: float,
    clv_percentile: float | None,
    cluster: int | None,
    is_anomaly: bool,
    revenue_at_risk_top_quartile_threshold: float,
) -> list[dict[str, Any]]:
    """Apply the three per-customer rules to already-computed signals.

    Pure conditional logic - no model inference happens here. clv_percentile/
    cluster may be None for a tenant missing CLV data or a trained segment
    model (see scan_for_alerts) - an alert rule that depends on either
    simply never fires for that customer, rather than erroring; rules that
    don't depend on the missing signal (e.g. highest_risk_segment_high_revenue
    when only CLV is missing) still evaluate normally.
    """
    alerts: list[dict[str, Any]] = []
    is_clv_top20 = clv_percentile is not None and clv_percentile >= CLV_TOP_PERCENTILE_CUTOFF

    if is_clv_top20 and churn_probability > 0.7:
        alerts.append(
            {
                "type": "high_value_high_risk",
                "severity": "high",
                "customer_id": customer_id,
                "description": (
                    f"Customer {customer_id} is in the top 20% by lifetime value "
                    f"(CLTV percentile {clv_percentile:.0f}) and has a "
                    f"{churn_probability * 100:.0f}% predicted churn probability — "
                    "a high-value customer at serious risk."
                ),
            }
        )

    if cluster == HIGHEST_RISK_CLUSTER and revenue_at_risk >= revenue_at_risk_top_quartile_threshold:
        alerts.append(
            {
                "type": "highest_risk_segment_high_revenue",
                "severity": "high",
                "customer_id": customer_id,
                "description": (
                    f"Customer {customer_id} belongs to the highest-risk segment "
                    f"(Cluster {HIGHEST_RISK_CLUSTER}, ~45.9% churn rate) and is in the top "
                    f"quartile of revenue at risk (${revenue_at_risk:.2f})."
                ),
            }
        )

    if is_anomaly and is_clv_top20:
        alerts.append(
            {
                "type": "anomaly_high_value",
                "severity": "medium",
                "customer_id": customer_id,
                "description": (
                    f"Customer {customer_id} was flagged as a statistical anomaly and is in "
                    f"the top 20% by lifetime value (CLTV percentile {clv_percentile:.0f}) — "
                    "worth a manual look, though being flagged is not inherently high-risk."
                ),
            }
        )

    return alerts


def evaluate_portfolio_alert(total_revenue_at_risk: float, threshold: float) -> dict[str, Any] | None:
    """Single portfolio-level rule: total top-20% revenue-at-risk vs threshold."""
    if total_revenue_at_risk <= threshold:
        return None
    return {
        "type": "portfolio_revenue_at_risk",
        "severity": "high",
        "customer_id": None,
        "description": (
            f"Total revenue at risk across the top 20% priority list is "
            f"${total_revenue_at_risk:,.2f}, exceeding the ${threshold:,.2f} threshold."
        ),
    }


def scan_for_alerts(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    clv_data_path: str | Path | None = "data/raw/telco_enriched.csv",
    portfolio_threshold: float = DEFAULT_PORTFOLIO_THRESHOLD,
    tenant_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """clv_data_path may be None, and model_dir may not have a
    segment_model.pkl/anomaly_model.pkl - all true for a tenant that's only
    had the core classifier trained (same graceful-degradation pattern as
    business_impact.py's compute_business_impact_bulk()). Each missing
    signal is skipped rather than crashing; evaluate_customer_alerts()
    already treats a None clv_percentile/cluster as "that rule doesn't
    fire" rather than "skip this customer entirely" - a customer missing
    only from the CLV file (a real per-tenant capability gap, not a
    genuine per-customer data error) still gets evaluated against the
    rules that don't need CLV."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)
    clv_data_path = Path(clv_data_path) if clv_data_path is not None else None

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    clv_column = tenant_config.get("clv_column")

    ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path, tenant_config=tenant_config
    )

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    encoders = joblib.load(model_dir / "encoders.pkl")

    # Per-customer cluster assignment - score with the already-fitted model,
    # same pattern as segment.load_cluster_profiles()/health_score.py.
    if (model_dir / "segment_model.pkl").exists():
        segment_saved = joblib.load(model_dir / "segment_model.pkl")
        segment_feature_columns = segment_saved["feature_columns"]
        X_segment = transform_categorical_features(df[segment_feature_columns], encoders)
        cluster_labels = segment_saved["model"].predict(segment_saved["scaler"].transform(X_segment))
        cluster_by_id = dict(zip(df[id_column], cluster_labels))
    else:
        # No segment model trained for this tenant - highest_risk_segment_
        # high_revenue below naturally never fires (cluster_by_id.get()
        # returns None, never equal to HIGHEST_RISK_CLUSTER).
        cluster_by_id = {}

    # Anomaly flags - score with the already-fitted model, no refit.
    if (model_dir / "anomaly_model.pkl").exists():
        anomaly_results = load_flagged_anomalies(model_dir=model_dir, data_path=data_path)
        anomaly_flag_by_id = dict(zip(anomaly_results[id_column], anomaly_results["anomaly_flag"]))
    else:
        # No anomaly model trained for this tenant - anomaly_high_value
        # below naturally never fires (is_anomaly is always False).
        anomaly_flag_by_id = {}

    # CLV percentile against the full population (same convention as health_score.py).
    if clv_data_path is not None and clv_column is not None:
        clv_by_id = pd.read_csv(clv_data_path).set_index(id_column)[clv_column]
        clv_percentile_by_id = clv_by_id.rank(pct=True) * 100.0
    else:
        # No CLV data for this tenant - high_value_high_risk/anomaly_high_value
        # below naturally never fire (clv_percentile is always None).
        clv_percentile_by_id = pd.Series(dtype=float)

    revenue_at_risk_q75 = ranking["expected_revenue_at_risk"].quantile(REVENUE_AT_RISK_TOP_QUARTILE)

    alerts: list[dict[str, Any]] = []

    for _, row in ranking.iterrows():
        customer_id = row["customerID"]
        clv_percentile = clv_percentile_by_id.get(customer_id)

        alerts.extend(
            evaluate_customer_alerts(
                customer_id=customer_id,
                churn_probability=float(row["churn_probability"]),
                revenue_at_risk=float(row["expected_revenue_at_risk"]),
                clv_percentile=float(clv_percentile) if clv_percentile is not None else None,
                cluster=cluster_by_id.get(customer_id),
                is_anomaly=anomaly_flag_by_id.get(customer_id) == -1,
                revenue_at_risk_top_quartile_threshold=revenue_at_risk_q75,
            )
        )

    top_n = round(0.2 * len(ranking))
    total_revenue_at_risk = float(
        ranking.sort_values("expected_revenue_at_risk", ascending=False).head(top_n)["expected_revenue_at_risk"].sum()
    )
    portfolio_alert = evaluate_portfolio_alert(total_revenue_at_risk, portfolio_threshold)
    if portfolio_alert is not None:
        alerts.append(portfolio_alert)

    return alerts
