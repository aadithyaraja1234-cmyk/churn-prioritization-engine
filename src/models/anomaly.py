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
from src.data.load import load_raw
from src.features.encode import transform_categorical_features


def detect_anomalies(
    contamination: float = 0.05,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    df = clean_data(load_raw(data_path))
    encoders = joblib.load(model_dir / "encoders.pkl")

    feature_columns = [col for col in df.columns if col not in ("customerID", "Churn")]
    X = transform_categorical_features(df[feature_columns], encoders)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    flags = model.predict(X)
    scores = model.decision_function(X)

    joblib.dump(
        {"model": model, "feature_columns": feature_columns, "contamination": contamination},
        model_dir / "anomaly_model.pkl",
    )

    results = pd.DataFrame(
        {"customerID": df["customerID"].values, "anomaly_flag": flags, "anomaly_score": scores}
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
    and return raw features merged with each customer's flag/score."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    saved = joblib.load(model_dir / "anomaly_model.pkl")
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    df = clean_data(load_raw(data_path))
    encoders = joblib.load(model_dir / "encoders.pkl")
    X = transform_categorical_features(df[feature_columns], encoders)

    results = df.copy()
    results["anomaly_flag"] = model.predict(X)
    results["anomaly_score"] = model.decision_function(X)
    return results
