"""K-Means customer segmentation — unsupervised, independent of the classifier pipeline.

Churn is never used as a clustering feature. It is only looked up afterward,
per resulting cluster, as a validation check on whether the unsupervised
segments happen to separate churn risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.clean import clean_data
from src.data.load import DEFAULT_TENANT_CONFIG, load_raw
from src.features.encode import transform_categorical_features

# Cluster names for the finalized k=4 model (random_state=42, telco.csv, full
# dataset). Grounded in the actual fitted profile — not the tenure-based
# assumption "cluster index == age of customer": clusters 0 and 1 are both
# ~30 months tenure, so what actually separates them is contract commitment
# (Month-to-month vs Two year) and spend, not age. If the model is ever
# retrained on different data, re-verify with profile_clusters() before
# trusting these labels — cluster indices are not guaranteed to keep the
# same meaning.
CLUSTER_LABELS: dict[int, str] = {
    0: "Mid-tenure, low-spend, flexible contract — moderate risk",
    1: "Mid-tenure, low-spend, locked-in — stable/low risk",
    2: "New, high-spend, flexible contract — highest risk",
    3: "Long-tenure, high-spend, locked-in — elevated risk",
}


def run_segmentation(
    k: int | None = None,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]

    model_dir = Path(model_dir)
    data_path = Path(data_path)

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    encoders = joblib.load(model_dir / "encoders.pkl")

    feature_columns = [col for col in df.columns if col not in (id_column, target_column)]
    X = transform_categorical_features(df[feature_columns], encoders)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_search_results = None
    if k is None:
        k_search_results = []
        for candidate_k in range(2, 11):
            model = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, labels)
            k_search_results.append(
                {"k": candidate_k, "inertia": float(model.inertia_), "silhouette": float(silhouette)}
            )
        chosen_k = max(k_search_results, key=lambda r: r["silhouette"])["k"]
    else:
        chosen_k = k

    final_model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_labels = final_model.fit_predict(X_scaled)

    joblib.dump(
        {"model": final_model, "scaler": scaler, "feature_columns": feature_columns, "tenant_config": tenant_config},
        model_dir / "segment_model.pkl",
    )

    cluster_assignments = pd.DataFrame(
        {id_column: df[id_column].values, "cluster": cluster_labels}
    )

    return {
        "chosen_k": chosen_k,
        "k_search_results": k_search_results,
        "cluster_assignments": cluster_assignments,
        "feature_columns": feature_columns,
    }


def profile_clusters(
    df: pd.DataFrame, cluster_assignments: pd.DataFrame, tenant_config: dict[str, Any] | None = None
) -> pd.DataFrame:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    target_positive_value = tenant_config["target_positive_value"]
    revenue_column = tenant_config["revenue_column"]
    duration_column = tenant_config.get("duration_column")
    segment_feature_column = tenant_config.get("segment_feature_column")

    # CLUSTER_LABELS is finalized, hand-verified descriptive text for
    # Telco's own k=4 fitted profile specifically (see its module-level
    # comment) - not a generic template. It only means what it says when
    # this really is that Telco run (the untouched, no-tenant_config-passed
    # default); applying it to another tenant's cluster indices would
    # describe a profile that was never actually verified against that
    # tenant's real data (e.g. it calls cluster 2 "high-spend" from Telco's
    # fitted profile - for a different tenant, cluster 2 could easily be the
    # cheapest one). is_telco_default is an identity check on purpose - a
    # Value equality, not identity: load_cluster_profiles() rebuilds this
    # dict from a joblib-unpickled segment_model.pkl (a different dict
    # object with the same values, since Telco's own run_segmentation()
    # call never passes tenant_config either), so an identity check would
    # wrongly say "not Telco" for Telco's own real, already-fitted model.
    is_telco_default = tenant_config == DEFAULT_TENANT_CONFIG

    merged = df.merge(cluster_assignments, on=id_column)

    profiles = []
    total = len(merged)
    for cluster_id, group in merged.groupby("cluster"):
        # mean_tenure/dominant_contract are only meaningful when this
        # tenant actually has a duration/segment-feature column mapped
        # (see src/data/load.py's DEFAULT_TENANT_CONFIG docstring) - None
        # here means "not available for this tenant", not zero/unknown.
        profiles.append(
            {
                "cluster": cluster_id,
                "size": len(group),
                "pct_of_total": 100.0 * len(group) / total,
                "mean_tenure": (
                    group[duration_column].mean() if duration_column and duration_column in group.columns else None
                ),
                "mean_MonthlyCharges": group[revenue_column].mean(),
                "dominant_contract": (
                    group[segment_feature_column].mode().iat[0]
                    if segment_feature_column and segment_feature_column in group.columns
                    else None
                ),
                "churn_rate_pct": 100.0 * group[target_column].eq(target_positive_value).mean(),
                "label": CLUSTER_LABELS.get(cluster_id, "unlabeled") if is_telco_default else "unlabeled",
            }
        )
    return pd.DataFrame(profiles).sort_values("cluster").reset_index(drop=True)


def load_cluster_profiles(
    model_dir: str | Path = "models/v1", data_path: str | Path = "data/raw/telco.csv"
) -> pd.DataFrame:
    """Score every customer with the already-fitted segment model (no refit)
    and return the labeled cluster profile table.

    tenant_config isn't a parameter here - the fitted model's own
    tenant_config (saved alongside it by run_segmentation()) is reused, so
    this always matches whatever tenant that segment_model.pkl was actually
    trained for."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    saved = joblib.load(model_dir / "segment_model.pkl")
    model = saved["model"]
    scaler = saved["scaler"]
    feature_columns = saved["feature_columns"]
    tenant_config = saved.get("tenant_config", DEFAULT_TENANT_CONFIG)
    id_column = tenant_config["id_column"]

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    encoders = joblib.load(model_dir / "encoders.pkl")
    X = transform_categorical_features(df[feature_columns], encoders)
    X_scaled = scaler.transform(X)

    cluster_labels = model.predict(X_scaled)
    cluster_assignments = pd.DataFrame({id_column: df[id_column].values, "cluster": cluster_labels})

    return profile_clusters(df, cluster_assignments, tenant_config)
