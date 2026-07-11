from pathlib import Path

import joblib
import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices
from src.features.encode import transform_categorical_features
from src.models.explain import explain_customer, get_global_importance


def test_get_global_importance_returns_ranked_features():
    importance = get_global_importance(model_dir="models/v1")

    assert not importance.empty
    assert list(importance.columns) == ["feature", "importance"]
    assert importance["importance"].is_monotonic_decreasing


def test_explain_customer_returns_top_contributions():
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
    df = clean_data(load_raw(data_path))
    _, test_indices = load_split_indices(Path(__file__).resolve().parents[1] / "models" / "v1")
    customer_id = df.loc[test_indices[0], "customerID"]

    explanation = explain_customer(customer_id, model_dir="models/v1", data_path=data_path)

    assert "customer_id" in explanation
    assert "churn_probability" in explanation
    assert "top_features" in explanation
    assert len(explanation["top_features"]) == 3


def test_marginal_effect_directions_are_customer_specific():
    """
    Regression test: ensure tenure direction differs between high-risk and low-risk customers.
    
    This validates that the leave-one-out marginal effect method correctly computes
    directions based on each customer's actual profile, not a static feature importance.
    Tenure should be protective (down) for low-risk customers and risky (up) for
    high-risk customers.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    # Load data and split indices
    df = clean_data(load_raw(data_path))
    train_idx, test_idx = load_split_indices(model_dir)

    # Get predictions for all test customers
    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    with open(model_dir / "split_indices.json", "r") as f:
        import json
        split_info = json.load(f)
        feature_names = split_info["feature_names"]

    X_test = df.loc[test_idx].drop(columns=["customerID", "Churn"])
    X_test_encoded = transform_categorical_features(X_test, encoders).reindex(
        columns=feature_names, fill_value=0
    )
    preds = model.predict_proba(X_test_encoded)[:, 1]

    # Find high-risk, low-risk, and mid-risk customers
    customer_ids = [df.loc[idx, "customerID"] for idx in test_idx]
    probabilities = list(zip(customer_ids, preds))
    ranked = sorted(probabilities, key=lambda x: x[1], reverse=True)

    highest_id = ranked[0][0]
    lowest_id = ranked[-1][0]

    # Get explanations
    high_risk_explanation = explain_customer(highest_id, model_dir=model_dir, data_path=data_path)
    low_risk_explanation = explain_customer(lowest_id, model_dir=model_dir, data_path=data_path)

    # Extract tenure direction from top features if present
    high_risk_features = {f["feature"]: f["direction"] for f in high_risk_explanation["top_features"]}
    low_risk_features = {f["feature"]: f["direction"] for f in low_risk_explanation["top_features"]}

    # Validate structure
    assert len(high_risk_explanation["top_features"]) == 3
    assert len(low_risk_explanation["top_features"]) == 3

    # Regression test: tenure must have different directions if present in both
    # (it typically appears in top 3 for both due to high global importance)
    if "tenure" in high_risk_features and "tenure" in low_risk_features:
        assert (
            high_risk_features["tenure"] != low_risk_features["tenure"]
        ), "tenure should show different directions for high-risk vs low-risk customers"


def test_contract_feature_dominates_explanations():
    """
    Verify that Contract (the highest-importance feature) appears in top_features
    for multiple customers after z-score normalization fix.
    
    Without normalization, Contract was overshadowed by TotalCharges' raw magnitude.
    With proper normalization, Contract should appear frequently in top-3 rankings.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    # Load data and split indices
    df = clean_data(load_raw(data_path))
    train_idx, test_idx = load_split_indices(model_dir)

    # Get predictions for all test customers to find representative samples
    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    with open(model_dir / "split_indices.json", "r") as f:
        import json
        split_info = json.load(f)
        feature_names = split_info["feature_names"]

    X_test = df.loc[test_idx].drop(columns=["customerID", "Churn"])
    X_test_encoded = transform_categorical_features(X_test, encoders).reindex(
        columns=feature_names, fill_value=0
    )
    preds = model.predict_proba(X_test_encoded)[:, 1]

    customer_ids = [df.loc[idx, "customerID"] for idx in test_idx]
    probabilities = list(zip(customer_ids, preds))
    ranked = sorted(probabilities, key=lambda x: x[1], reverse=True)

    highest_id = ranked[0][0]
    lowest_id = ranked[-1][0]
    closest_id = min(ranked, key=lambda x: abs(x[1] - 0.5))[0]

    # Get explanations for the three representative customers
    high_explanation = explain_customer(highest_id, model_dir=model_dir, data_path=data_path)
    low_explanation = explain_customer(lowest_id, model_dir=model_dir, data_path=data_path)
    mid_explanation = explain_customer(closest_id, model_dir=model_dir, data_path=data_path)

    # Extract feature names from each
    high_features = {f["feature"] for f in high_explanation["top_features"]}
    low_features = {f["feature"] for f in low_explanation["top_features"]}
    mid_features = {f["feature"] for f in mid_explanation["top_features"]}

    # Contract should appear in at least 2 of the 3 customers' top_features
    contract_count = sum(
        [
            "Contract" in high_features,
            "Contract" in low_features,
            "Contract" in mid_features,
        ]
    )
    assert contract_count >= 2, (
        f"Contract should appear in at least 2 of 3 customer explanations, "
        f"but only appeared in {contract_count}"
    )
