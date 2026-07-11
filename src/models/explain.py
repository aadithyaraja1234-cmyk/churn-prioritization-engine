from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices, split_data
from src.features.encode import transform_categorical_features


def _load_phase1_artifacts(model_dir: str | Path) -> tuple[Any, dict[str, dict[object, int]], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[int], list[int]]:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    split_info_path = model_dir / "split_indices.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"Split index file not found: {split_info_path}")

    with split_info_path.open("r", encoding="utf-8") as handle:
        split_info = json.load(handle)

    train_idx = split_info["train_idx"]
    test_idx = split_info["test_idx"]

    return model, encoders, pd.DataFrame(split_info["train_features"]), pd.DataFrame(split_info["test_features"]), pd.Series(split_info["train_labels"]), pd.Series(split_info["test_labels"]), train_idx, test_idx


def get_global_importance(model_dir: str | Path = "models/v1") -> pd.DataFrame:
    """Return a ranked feature-importance table for the trained model."""
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.pkl")

    feature_names = _load_feature_names(model_dir)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def explain_customer(customer_id: Any, model_dir: str | Path = "models/v1", data_path: str | Path = "data/raw/telco.csv") -> dict[str, Any]:
    """Compute feature contributions for a single customer using leave-one-out marginal effects.

    For each feature, replaces it with the training set mean and measures the change in
    predicted churn probability. This reveals whether each feature's actual value increases
    or decreases risk for THIS customer relative to average.
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    df = clean_data(load_raw(data_path))
    X_train, X_test, _, _, train_idx, test_idx = _load_split_from_disk(df, model_dir)

    feature_names = _load_feature_names(model_dir)
    X_train_encoded = transform_categorical_features(X_train, encoders).reindex(columns=feature_names, fill_value=0)
    train_feature_means = X_train_encoded.mean()

    customer_row = df.loc[df["customerID"] == customer_id]
    if customer_row.empty:
        raise KeyError(f"Customer {customer_id} not found")
    if not set([int(idx) for idx in test_idx]).__contains__(int(customer_row.index[0])):
        raise ValueError(f"Customer {customer_id} is not in the saved test split")

    customer_features = customer_row.drop(columns=["customerID", "Churn"]).copy()
    customer_features = customer_features.reindex(columns=feature_names, fill_value=0)
    customer_features = transform_categorical_features(customer_features, encoders).reindex(columns=feature_names, fill_value=0)

    original_probability = float(model.predict_proba(customer_features)[0, 1])

    # Compute leave-one-out marginal effects for each feature
    customer_vector = customer_features.iloc[0].copy()
    marginal_effects = {}

    for feature in feature_names:
        # Create modified vector with this feature set to training mean
        modified_vector = customer_vector.copy()
        modified_vector[feature] = train_feature_means[feature]

        # Get probability with modified feature
        modified_probability = float(model.predict_proba(modified_vector.values.reshape(1, -1))[0, 1])

        # Marginal effect: original probability - modified probability
        # Positive = this feature's value increased risk; negative = decreased risk
        marginal_effects[feature] = original_probability - modified_probability

    # Sort by absolute value and take top 3
    sorted_effects = sorted(marginal_effects.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_effects[:3]

    contributions = []
    for feature_name, contribution in top_features:
        direction = "up" if contribution > 0 else "down"
        contributions.append({
            "feature": feature_name,
            "contribution": float(contribution),
            "direction": direction,
        })

    return {
        "customer_id": customer_id,
        "churn_probability": original_probability,
        "top_features": contributions,
    }


def _load_feature_names(model_dir: str | Path) -> list[str]:
    model_dir = Path(model_dir)
    split_info_path = model_dir / "split_indices.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"Split index file not found: {split_info_path}")

    with split_info_path.open("r", encoding="utf-8") as handle:
        split_info = json.load(handle)

    return split_info["feature_names"]


def _load_split_from_disk(df: pd.DataFrame, model_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[int], list[int]]:
    model_dir = Path(model_dir)
    train_idx, test_idx = load_split_indices(model_dir)

    X_train = df.loc[train_idx].drop(columns=["customerID", "Churn"]).copy()
    X_test = df.loc[test_idx].drop(columns=["customerID", "Churn"]).copy()
    y_train = df.loc[train_idx, "Churn"].copy()
    y_test = df.loc[test_idx, "Churn"].copy()

    return X_train, X_test, y_train, y_test, train_idx, test_idx
