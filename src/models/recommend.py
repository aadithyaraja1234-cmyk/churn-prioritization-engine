"""Retention action recommender — a rules layer over already-computed outputs.

DESIGNED HEURISTIC: the rule mapping below is a judgment call about which
action plausibly fits which driver, not a model. It has NOT been validated
against real retention outcomes (no A/B test, no historical intervention
data) - same honesty requirement as the Phase 5 feedback loop's synthetic
disclosure. Treat this as a starting playbook, not a proven prescription.

No new modeling: churn probability + top drivers come from explain_customer()
(the existing explainability module), CLV percentile from the real CLTV
column, and the anomaly/segment flags from the already-fitted models. This
module only reuses them.

The rule logic (determine_recommendation) is kept separate from the
real-data orchestration (recommend_action_for_customer), so the rules can be
tested against synthetic, fully-controlled inputs - matching the pattern
already used in alerts.py.

Generalized to accept tenant_config (same pattern as prioritize.py/
business_impact.py/explain.py - read back from model_dir/split_indices.json
via src/data/split.py's load_tenant_config() when omitted): the third
confirmed real instance of this project's recurring hardcoded-Telco-column
bug (Meridian's action-queue/alerts, then Aurora's business-impact, now
this). Only the DATA ACCESS is generalized here, not determine_recommendation()'s
actual rule logic - "Contract"/"tenure" as literal top-driver names are
Telco-specific playbook text, the same kind of hand-tuned prose as
segment.py's CLUSTER_LABELS, not something a column-name lookup can make
tenant-generic on its own. A tenant whose top driver is genuinely named
something else just never matches those two specific rules and falls
through to the churn-probability/anomaly-based ones below them - a real,
honest degrade, not a crash.
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
from src.models.alerts import CLV_TOP_PERCENTILE_CUTOFF, HIGHEST_RISK_CLUSTER
from src.models.anomaly import load_flagged_anomalies
from src.models.explain import explain_customer

_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


class CustomerNotFoundError(ValueError):
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        super().__init__(f"Customer '{customer_id}' not found")


def determine_recommendation(
    top_feature_names: list[str],
    monthly_charges: float,
    monthly_charges_median: float,
    churn_probability: float,
    is_anomaly: bool,
    is_highest_risk_segment: bool,
) -> dict[str, Any]:
    """Pure rule evaluation - no model inference happens here.

    Rules are checked in priority order; the first match wins.
    """
    contract_is_top_driver = "Contract" in top_feature_names
    tenure_is_top_driver = "tenure" in top_feature_names

    if contract_is_top_driver and monthly_charges > monthly_charges_median:
        return {
            "action": "Offer a discount or loyalty pricing to reduce price sensitivity",
            "triggered_by": ["Contract", "MonthlyCharges (above median)"],
        }

    if contract_is_top_driver:
        return {
            "action": "Offer a contract upgrade incentive (e.g. waived fee for switching to a longer contract)",
            "triggered_by": ["Contract", "MonthlyCharges (at or below median)"],
        }

    if tenure_is_top_driver and churn_probability > 0.6:
        return {
            "action": "Assign priority onboarding support",
            "triggered_by": ["tenure", "churn_probability"],
        }

    if is_anomaly and not is_highest_risk_segment:
        return {
            "action": "Review manually — unusual profile, standard playbook may not apply",
            "triggered_by": ["anomaly_flag"],
        }

    if churn_probability < 0.3:
        return {
            "action": "No action needed — monitor as part of regular review",
            "triggered_by": ["churn_probability (low)"],
        }

    return {
        "action": "Standard retention outreach — no dominant driver identified",
        "triggered_by": [],
    }


def apply_priority_prefix(action: str, is_clv_top20: bool) -> str:
    """Prefix an action with 'PRIORITY: ' when CLV percentile is top 20% - pure, no model inference."""
    return f"PRIORITY: {action}" if is_clv_top20 else action


def _population_context(
    model_dir: Path, data_path: Path, clv_data_path: Path | None, tenant_config: dict[str, Any]
) -> dict[str, Any]:
    """Population-wide lookups reused across customers - cached in-memory,
    since these don't depend on which customer is being scored.

    segment_model.pkl, anomaly_model.pkl, and clv_data_path may not exist
    for a tenant that's only had the core classifier trained (same
    graceful-degradation pattern as business_impact.py's
    compute_business_impact_bulk() for clv_data_path=None/no
    survival_model.pkl) - each missing signal is skipped rather than
    crashing. determine_recommendation()'s is_highest_risk_segment/
    is_anomaly checks already evaluate False for a customer missing from
    an empty {} lookup, and clv_percentile_by_id.get(id, 0.0) already
    falls back to 0.0 for an empty Series - no special-casing needed
    downstream beyond returning empty/None here."""
    cache_key = (str(model_dir), str(data_path), str(clv_data_path))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    id_column = tenant_config["id_column"]
    revenue_column = tenant_config["revenue_column"]
    clv_column = tenant_config.get("clv_column")

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    encoders = joblib.load(model_dir / "encoders.pkl")

    has_segment = (model_dir / "segment_model.pkl").exists()
    if has_segment:
        segment_saved = joblib.load(model_dir / "segment_model.pkl")
        segment_feature_columns = segment_saved["feature_columns"]
        X_segment = transform_categorical_features(df[segment_feature_columns], encoders)
        cluster_labels = segment_saved["model"].predict(segment_saved["scaler"].transform(X_segment))
        cluster_by_id = dict(zip(df[id_column], cluster_labels))
    else:
        cluster_by_id = {}

    has_anomaly = (model_dir / "anomaly_model.pkl").exists()
    if has_anomaly:
        anomaly_results = load_flagged_anomalies(model_dir=model_dir, data_path=data_path)
        anomaly_flag_by_id = dict(zip(anomaly_results[id_column], anomaly_results["anomaly_flag"]))
    else:
        anomaly_flag_by_id = {}

    has_clv = clv_data_path is not None and clv_column is not None
    if has_clv:
        clv_by_id = pd.read_csv(clv_data_path).set_index(id_column)[clv_column]
        clv_percentile_by_id = clv_by_id.rank(pct=True) * 100.0
    else:
        clv_percentile_by_id = pd.Series(dtype=float)

    monthly_charges_by_id = df.set_index(id_column)[revenue_column]
    monthly_charges_median = float(df[revenue_column].median())

    context = {
        "cluster_by_id": cluster_by_id,
        "anomaly_flag_by_id": anomaly_flag_by_id,
        "clv_percentile_by_id": clv_percentile_by_id,
        "monthly_charges_by_id": monthly_charges_by_id,
        "monthly_charges_median": monthly_charges_median,
        "has_segment": has_segment,
        "has_anomaly": has_anomaly,
        "has_clv": has_clv,
    }
    _CACHE[cache_key] = context
    return context


def recommend_action_for_customer(
    customer_id: str,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    clv_data_path: str | Path | None = "data/raw/telco_enriched.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    data_path = Path(data_path)
    clv_data_path = Path(clv_data_path) if clv_data_path is not None else None

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)

    context = _population_context(model_dir, data_path, clv_data_path, tenant_config)
    if customer_id not in context["monthly_charges_by_id"].index:
        raise CustomerNotFoundError(customer_id)

    explanation = explain_customer(customer_id, model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    top_feature_names = [contribution["feature"] for contribution in explanation["top_features"]]

    clv_percentile = float(context["clv_percentile_by_id"].get(customer_id, 0.0))
    is_clv_top20 = clv_percentile >= CLV_TOP_PERCENTILE_CUTOFF
    cluster = context["cluster_by_id"].get(customer_id)
    is_anomaly = context["anomaly_flag_by_id"].get(customer_id) == -1

    recommendation = determine_recommendation(
        top_feature_names=top_feature_names,
        monthly_charges=float(context["monthly_charges_by_id"][customer_id]),
        monthly_charges_median=context["monthly_charges_median"],
        churn_probability=explanation["churn_probability"],
        is_anomaly=is_anomaly,
        is_highest_risk_segment=cluster == HIGHEST_RISK_CLUSTER,
    )

    action = apply_priority_prefix(recommendation["action"], is_clv_top20)

    # Honest coverage disclosure - a tenant missing segment/anomaly/CLV
    # artifacts still gets a real recommendation from churn probability
    # alone, but the response says so rather than silently looking
    # identical to Telco's full-signal result.
    based_on = ["churn_probability"]
    if context["has_clv"]:
        based_on.append("clv")
    if context["has_segment"]:
        based_on.append("segment")
    if context["has_anomaly"]:
        based_on.append("anomaly")

    return {
        "customer_id": customer_id,
        "recommended_action": action,
        "triggered_by": recommendation["triggered_by"],
        "is_priority": is_clv_top20,
        "based_on": based_on,
    }
