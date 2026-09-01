from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices, load_tenant_config, split_data
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


def explain_customer(
    customer_id: Any,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute feature contributions for a single customer using leave-one-out marginal effects.

    For each feature, replaces it with the training set mean and measures the change in
    predicted churn probability. This reveals whether each feature's actual value increases
    or decreases risk for THIS customer relative to average.

    Generalized to accept tenant_config (same pattern as prioritize.py -
    read back from model_dir/split_indices.json via
    src/data/split.py's load_tenant_config() when omitted).
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    target_positive_value = tenant_config["target_positive_value"]

    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    X_train, X_test, _, _, train_idx, test_idx = _load_split_from_disk(df, model_dir, tenant_config)

    feature_names = _load_feature_names(model_dir)
    X_train_encoded = transform_categorical_features(X_train, encoders).reindex(columns=feature_names, fill_value=0)
    train_feature_means = X_train_encoded.mean()

    customer_row = df.loc[df[id_column] == customer_id]
    if customer_row.empty:
        raise KeyError(f"Customer {customer_id} not found")
    if not set([int(idx) for idx in test_idx]).__contains__(int(customer_row.index[0])):
        raise ValueError(f"Customer {customer_id} is not in the saved test split")

    customer_features = customer_row.drop(columns=[id_column, target_column]).copy()
    customer_features = customer_features.reindex(columns=feature_names, fill_value=0)
    customer_features = transform_categorical_features(customer_features, encoders).reindex(columns=feature_names, fill_value=0)

    # Looks up target_positive_value's actual position in the fitted model's
    # own classes_ rather than assuming index 1 - same fix, same reasoning,
    # as src/models/prioritize.py's get_priority_ranking() (see that
    # module's docstring for the full explanation).
    positive_index = list(model.classes_).index(target_positive_value)
    original_probability = float(model.predict_proba(customer_features)[0, positive_index])

    # Compute leave-one-out marginal effects for each feature
    customer_vector = customer_features.iloc[0].copy()
    marginal_effects = {}

    for feature in feature_names:
        # Create modified vector with this feature set to training mean
        modified_vector = customer_vector.copy()
        modified_vector[feature] = train_feature_means[feature]

        # Get probability with modified feature
        modified_probability = float(model.predict_proba(modified_vector.values.reshape(1, -1))[0, positive_index])

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


def list_test_split_customer_ids(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> list[Any]:
    """Every customer_id in this tenant's saved test split - the exact same
    population explain_customer() (and, via it, recommend_action_for_customer())
    actually accepts, derived from the SAME test_idx rather than duplicating
    its own notion of "eligible" that could silently drift out of sync.

    Exists because /customers (api/main.py) lists every ingested customer -
    train and test alike - but recommend_action_for_customer() only ever
    scores test-split customers (explain_customer() raises "not in the
    saved test split" for the rest, by design: it avoids serving
    recommendations for customers the model was directly trained on). A
    caller that needs to offer only genuinely recommendable customers (e.g.
    Customer 360's picker) filters against this list rather than the full
    /customers set.
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)
    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    _, test_idx = load_split_indices(model_dir)
    return df.loc[test_idx, id_column].tolist()


def _load_feature_names(model_dir: str | Path) -> list[str]:
    model_dir = Path(model_dir)
    split_info_path = model_dir / "split_indices.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"Split index file not found: {split_info_path}")

    with split_info_path.open("r", encoding="utf-8") as handle:
        split_info = json.load(handle)

    return split_info["feature_names"]


def _load_split_from_disk(
    df: pd.DataFrame, model_dir: str | Path, tenant_config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[int], list[int]]:
    model_dir = Path(model_dir)
    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]

    train_idx, test_idx = load_split_indices(model_dir)

    X_train = df.loc[train_idx].drop(columns=[id_column, target_column]).copy()
    X_test = df.loc[test_idx].drop(columns=[id_column, target_column]).copy()
    y_train = df.loc[train_idx, target_column].copy()
    y_test = df.loc[test_idx, target_column].copy()

    return X_train, X_test, y_train, y_test, train_idx, test_idx
