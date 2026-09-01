"""Customer health score — a DESIGNED HEURISTIC, not a trained or separately
validated model. It combines up to five already-computed outputs from models
built elsewhere in this project into one 0-100 number. The weights below are
a judgment call, not calibrated against any ground truth of "health."

Components (each mapped to its own 0-100 sub-score, higher = healthier):

- churn_component (weight 0.4): 100 * (1 - churn_probability), from the
  saved classifier (model.pkl). The dominant signal, since it's the most
  directly validated model in this project (ROC-AUC 0.8422 - see
  models/v1/metadata.json, the documented reference baseline as of this
  project's hyperparameter-tuning investigation). Always available -
  business_impact_core (this tool's own gate) requires a trained classifier.
- survival_component (weight 0.2): percentile rank of the customer's Cox
  model (survival_model.pkl) partial hazard among all customers, inverted
  so LOWER hazard (longer expected survival) scores HIGHER.
- segment_component (weight 0.2): 100 - (this customer's K-Means cluster's
  aggregate churn rate). Coarser than churn_component - it reflects the
  whole segment the customer was clustered into, not this individual.
- anomaly_component (weight 0.1): a small, deliberately modest penalty (90
  vs 100) if IsolationForest flagged the customer. NUANCE: being flagged is
  NOT necessarily bad - this project's own anomaly analysis found flagged
  customers actually have a LOWER churn rate on average (10.2% vs 27.4%
  baseline) than normal customers. The penalty exists only because
  "statistically unusual" carries some operational uncertainty worth a
  small ding, not because it predicts higher risk - hence the low weight.
- clv_component (weight 0.1): percentile rank of real CLV among all
  customers. Higher CLV -> higher score, but this reflects business
  priority (worth more retention effort), not customer health - a
  high-CLV, high-churn-risk customer is NOT actually healthy, it's just
  valuable. Kept at low weight for that reason.

Generalized to accept tenant_config (same pattern as prioritize.py/
business_impact.py/recommend.py - read back from model_dir/split_indices.json
via src/data/split.py's load_tenant_config() when omitted). survival_model.pkl/
segment_model.pkl/anomaly_model.pkl/clv data may not exist for a tenant
that's only had the core classifier trained (same graceful-degradation
pattern as business_impact.py's compute_business_impact_bulk()/recommend.py's
_population_context()) - a missing component is DROPPED from the weighted
average entirely (WEIGHTS renormalized across whichever components are
actually available for this tenant), not silently defaulted to a neutral
50 (which would fake a real signal that doesn't exist and pull every
customer's score toward the middle without disclosure - the same "drop the
term, don't fake a value" discipline business_impact.py's opportunity_score()
already uses for a missing CLV weight). Which components were actually used
is reported back via "components_used" so a caller never mistakes a
reduced-signal score for the full one.
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
from src.models.segment import load_cluster_profiles

WEIGHTS: dict[str, float] = {
    "churn_component": 0.4,
    "survival_component": 0.2,
    "segment_component": 0.2,
    "anomaly_component": 0.1,
    "clv_component": 0.1,
}

ANOMALY_FLAGGED_SCORE = 90.0
ANOMALY_NORMAL_SCORE = 100.0

_CACHE: dict[tuple[str, str, str, str], pd.DataFrame] = {}


class CustomerNotFoundError(ValueError):
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        super().__init__(f"Customer '{customer_id}' not found")


def compute_health_scores_bulk(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    clv_model_dir: str | Path | None = "models/v1_enriched",
    clv_data_path: str | Path | None = "data/raw/telco_enriched.csv",
    tenant_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute every available component + the final score for ALL
    customers at once.

    Cached in-memory per (model_dir, data_path, clv_model_dir, clv_data_path)
    for the life of the process - this reuses existing saved model artifacts
    read-only, so repeated calls (e.g. one per row in a priority table) don't
    each re-run inference across the whole population from scratch.
    """
    cache_key = (str(model_dir), str(data_path), str(clv_model_dir), str(clv_data_path))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    model_dir = Path(model_dir)
    data_path = Path(data_path)
    clv_data_path = Path(clv_data_path) if clv_data_path is not None else None

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    target_positive_value = tenant_config["target_positive_value"]
    clv_column = tenant_config.get("clv_column")

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    encoders = joblib.load(model_dir / "encoders.pkl")
    feature_columns = [col for col in df.columns if col not in (id_column, target_column)]

    components: dict[str, pd.Series] = {}

    # 1. churn_component - always available (business_impact_core, this
    # tool's own gate, requires a trained classifier).
    classifier = joblib.load(model_dir / "model.pkl")
    X = transform_categorical_features(df[feature_columns], encoders)
    # Looks up target_positive_value's actual position in the fitted
    # model's own classes_ rather than assuming index 1 - same fix, same
    # reasoning, as src/models/prioritize.py's get_priority_ranking().
    positive_index = list(classifier.classes_).index(target_positive_value)
    churn_probability = classifier.predict_proba(X)[:, positive_index]
    components["churn_component"] = pd.Series(100.0 * (1 - churn_probability), index=df.index)

    # 2. survival_component - only if this tenant has a fitted survival model.
    if (model_dir / "survival_model.pkl").exists():
        survival_saved = joblib.load(model_dir / "survival_model.pkl")
        cph = survival_saved["model"]
        survival_encoders = survival_saved["encoders"]
        survival_feature_columns = survival_saved["feature_columns"]
        X_survival = transform_categorical_features(df[survival_feature_columns], survival_encoders)
        hazards = cph.predict_partial_hazard(X_survival).to_numpy()
        # Percentile of population with a HIGHER hazard than this customer -
        # a customer with the lowest hazard in the population gets ~100
        # (healthiest).
        components["survival_component"] = pd.Series(
            100.0 * pd.Series(hazards).rank(pct=True, ascending=False).to_numpy(), index=df.index
        )

    # 3. segment_component - only if this tenant has a fitted segment model.
    if (model_dir / "segment_model.pkl").exists():
        segment_saved = joblib.load(model_dir / "segment_model.pkl")
        segment_model = segment_saved["model"]
        segment_scaler = segment_saved["scaler"]
        segment_feature_columns = segment_saved["feature_columns"]
        X_segment = transform_categorical_features(df[segment_feature_columns], encoders)
        cluster_labels = segment_model.predict(segment_scaler.transform(X_segment))
        cluster_profiles = load_cluster_profiles(model_dir=model_dir, data_path=data_path)
        churn_rate_by_cluster = cluster_profiles.set_index("cluster")["churn_rate_pct"]
        components["segment_component"] = pd.Series(
            100.0 - pd.Series(cluster_labels).map(churn_rate_by_cluster).to_numpy(), index=df.index
        )

    # 4. anomaly_component - only if this tenant has a fitted anomaly model.
    if (model_dir / "anomaly_model.pkl").exists():
        anomaly_saved = joblib.load(model_dir / "anomaly_model.pkl")
        anomaly_model = anomaly_saved["model"]
        anomaly_feature_columns = anomaly_saved["feature_columns"]
        X_anomaly = transform_categorical_features(df[anomaly_feature_columns], encoders)
        anomaly_flags = anomaly_model.predict(X_anomaly)
        components["anomaly_component"] = pd.Series(
            pd.Series(anomaly_flags).map({-1: ANOMALY_FLAGGED_SCORE, 1: ANOMALY_NORMAL_SCORE}).to_numpy(),
            index=df.index,
        )

    # 5. clv_component - only if this tenant mapped a clv_column AND has a
    # clv_data_path (both must be real - a tenant can have clv_column
    # mapped without clv.py's own sanity gate ever having passed, in which
    # case clv_data_path_for() upstream already returns None).
    if clv_data_path is not None and clv_column is not None:
        clv_df = pd.read_csv(clv_data_path).set_index(id_column)[clv_column]
        components["clv_component"] = pd.Series(
            (clv_df.rank(pct=True) * 100.0).reindex(df[id_column]).to_numpy(), index=df.index
        )

    # Renormalize WEIGHTS across only the components actually available for
    # this tenant - see module docstring for why a missing component is
    # dropped entirely rather than defaulted to a neutral 50.
    components_used = list(components.keys())
    weight_total = sum(WEIGHTS[name] for name in components_used)
    health_score = sum(WEIGHTS[name] * components[name] for name in components_used) / weight_total

    result = pd.DataFrame({id_column: df[id_column].values, **{name: series.values for name, series in components.items()}})
    result["health_score"] = health_score.values
    result["components_used"] = [components_used] * len(result)
    result = result.set_index(id_column)

    _CACHE[cache_key] = result
    return result


def compute_health_score(
    customer_id: str,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    clv_model_dir: str | Path | None = "models/v1_enriched",
    clv_data_path: str | Path | None = "data/raw/telco_enriched.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bulk = compute_health_scores_bulk(model_dir, data_path, clv_model_dir, clv_data_path, tenant_config=tenant_config)
    if customer_id not in bulk.index:
        raise CustomerNotFoundError(customer_id)

    row = bulk.loc[customer_id]
    components_used = row["components_used"]
    # Renormalized across components_used (see compute_health_scores_bulk()'s
    # module-docstring-referenced renormalization) - reports the weights
    # actually applied to reach health_score, not the raw pre-renormalization
    # WEIGHTS table, so weights[name] * components[name] summed reproduces
    # health_score exactly rather than silently under-summing to weight_total.
    weight_total = sum(WEIGHTS[name] for name in components_used)
    return {
        "customer_id": customer_id,
        "health_score": round(float(row["health_score"]), 2),
        "components": {name: round(float(row[name]), 2) for name in components_used},
        "components_used": components_used,
        "weights": {name: round(WEIGHTS[name] / weight_total, 4) for name in components_used},
    }
