"""Business impact engine: prioritize customers by revenue at risk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices
from src.features.encode import transform_categorical_features


def get_priority_ranking(
    strategy: str = "revenue_weighted",
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
) -> pd.DataFrame:
    """Rank test customers by revenue at risk or churn probability.

    Args:
        strategy: One of "revenue_weighted" (default) or "probability_only".
            - "revenue_weighted": Sort by expected_revenue_at_risk (churn_prob * MonthlyCharges)
            - "probability_only": Sort by churn_probability only
        model_dir: Path to directory containing saved Phase 1 artifacts
        data_path: Path to raw telco.csv data

    Returns:
        DataFrame with columns:
        - customerID: Unique customer identifier
        - churn_probability: Model's predicted churn probability [0, 1]
        - MonthlyCharges: Customer's monthly subscription fee
        - expected_revenue_at_risk: churn_probability * MonthlyCharges
        - rank: 1-indexed rank according to the chosen strategy
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    if strategy not in ("revenue_weighted", "probability_only"):
        raise ValueError(f"strategy must be 'revenue_weighted' or 'probability_only', got {strategy}")

    # Load Phase 1 artifacts
    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    with open(model_dir / "split_indices.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)
        test_idx = split_info["test_idx"]
        feature_names = split_info["feature_names"]

    # Load raw data
    df = clean_data(load_raw(data_path))

    # Extract test set
    X_test = df.loc[test_idx].drop(columns=["customerID", "Churn"]).copy()
    customer_ids = df.loc[test_idx, "customerID"].copy()
    monthly_charges = df.loc[test_idx, "MonthlyCharges"].copy()

    # Encode test features
    X_test_encoded = transform_categorical_features(X_test, encoders).reindex(
        columns=feature_names, fill_value=0
    )

    # Get churn probabilities
    churn_probs = model.predict_proba(X_test_encoded)[:, 1]

    # Build results DataFrame
    results = pd.DataFrame(
        {
            "customerID": customer_ids.values,
            "churn_probability": churn_probs,
            "MonthlyCharges": monthly_charges.values,
        }
    )

    # Compute revenue at risk
    results["expected_revenue_at_risk"] = results["churn_probability"] * results["MonthlyCharges"]

    # Sort by strategy
    if strategy == "revenue_weighted":
        results = results.sort_values("expected_revenue_at_risk", ascending=False)
    else:  # probability_only
        results = results.sort_values("churn_probability", ascending=False)

    # Add rank (1-indexed)
    results = results.reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)

    return results
