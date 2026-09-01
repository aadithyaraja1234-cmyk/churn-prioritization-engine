"""Isolation Forest anomaly detection — unsupervised, independent of the classifier pipeline.

Churn is never used as a feature. It is only looked up afterward, per flagged
customer, as a validation check on whether flagged anomalies skew toward churn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data.clean import clean_data
from src.data.load import DEFAULT_TENANT_CONFIG, load_raw
from src.features.encode import transform_categorical_features


def detect_anomalies(
    contamination: float = 0.05,
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

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    flags = model.predict(X)
    scores = model.decision_function(X)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "contamination": contamination,
            "tenant_config": tenant_config,
        },
        model_dir / "anomaly_model.pkl",
    )

    results = pd.DataFrame(
        {id_column: df[id_column].values, "anomaly_flag": flags, "anomaly_score": scores}
    )

    return {
        "contamination": contamination,
        "feature_columns": feature_columns,
        "results": results,
    }


def load_flagged_anomalies(
    model_dir: str | Path = "models/v1", data_path: str | Path = "data/raw/telco.csv"
) -> pd.DataFrame:
    """Score every customer with the already-fitted anomaly model (no refit)
    and return raw features merged with each customer's flag/score.

    tenant_config isn't a parameter here - the fitted model's own
    tenant_config (saved alongside it by detect_anomalies()) is reused, so
    this always matches whatever tenant that anomaly_model.pkl was
    actually trained for."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    saved = joblib.load(model_dir / "anomaly_model.pkl")
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    tenant_config = saved.get("tenant_config", DEFAULT_TENANT_CONFIG)

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    encoders = joblib.load(model_dir / "encoders.pkl")
    X = transform_categorical_features(df[feature_columns], encoders)

    results = df.copy()
    results["anomaly_flag"] = model.predict(X)
    results["anomaly_score"] = model.decision_function(X)
    return results
