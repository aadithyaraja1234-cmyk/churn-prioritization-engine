"""Phase 5: Feedback Loop Simulation.

This module demonstrates the feedback loop mechanism by:
1. Simulating a batch of customer retention outcomes
2. Retraining the model on train + synthetic feedback data
3. Comparing predictions before/after

IMPORTANT: All outputs, metadata, and results from this module are SIMULATED
and do NOT represent real model improvements. The retention outcomes used
here are synthetic, not from actual customer interactions. This demonstrates
the update mechanism functions correctly, nothing more.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices
from src.features.encode import fit_categorical_encoders, transform_categorical_features
from src.models.prioritize import get_priority_ranking


def simulate_feedback_batch(
    n_customers: int = 100,
    retention_rate: float = 0.3,
    seed: int = 42,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
) -> pd.DataFrame:
    """Simulate a batch of customer feedback outcomes.

    Samples customers from the top 20% of the test set (by revenue-weighted rank),
    then synthetically marks retention_rate of them as retained (flips Churn to 'No').
    The original test set ground truth is NEVER modified.

    Args:
        n_customers: Number of customers to sample from top 20%
        retention_rate: Fraction of sampled customers to mark as retained [0, 1]
        seed: Random seed for reproducibility
        model_dir: Path to directory containing Phase 1 artifacts
        data_path: Path to raw telco.csv data

    Returns:
        DataFrame with the same columns as the original data, but:
        - Rows are from top 20% of test set (revenue-weighted rank)
        - retention_rate of them have Churn flipped to 'No' (synthetic)
        - All rows have data_source = "synthetic_feedback_not_real"
        - This is a SEPARATE DataFrame; the original data is unchanged
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    # Load the original data and test set indices
    df = clean_data(load_raw(data_path))
    _, test_idx = load_split_indices(model_dir)

    # Get revenue-weighted ranking of test set
    ranking = get_priority_ranking(strategy="revenue_weighted", model_dir=model_dir, data_path=data_path)

    # Compute top 20% cutoff
    top_pct = 0.2
    n_top = max(1, int(len(ranking) * top_pct))
    top_20_pct_customer_ids = ranking.head(n_top)["customerID"].tolist()

    # Sample n_customers from top 20%
    rng = random.Random(seed)
    sampled_ids = rng.sample(top_20_pct_customer_ids, min(n_customers, len(top_20_pct_customer_ids)))

    # Extract rows for sampled customers
    feedback_batch = df[df["customerID"].isin(sampled_ids)].copy()

    # Synthetically mark retention_rate of them as retained
    n_retained = int(len(feedback_batch) * retention_rate)
    retained_indices = rng.sample(range(len(feedback_batch)), n_retained)
    feedback_batch.loc[feedback_batch.index[retained_indices], "Churn"] = "No"

    # Add data_source column to label as synthetic
    feedback_batch["data_source"] = "synthetic_feedback_not_real"

    return feedback_batch


def retrain_with_feedback(
    feedback_batch: pd.DataFrame,
    model_dir: str | Path = "models/v1",
    output_dir: str | Path = "models/v2",
    data_path: str | Path = "data/raw/telco.csv",
    config_path: str | Path = "config/config.yaml",
    retention_rate: float | None = None,
) -> dict[str, Any]:
    """Retrain the model using original training set + synthetic feedback batch.

    Loads the original train set (unchanged), appends the feedback batch,
    retrains the same model architecture/hyperparameters, and saves to output_dir.
    The original v1 model and data are NEVER modified.

    Args:
        feedback_batch: DataFrame returned by simulate_feedback_batch()
        model_dir: Path to directory containing Phase 1 artifacts (v1)
        output_dir: Path to directory where v2 will be saved
        data_path: Path to raw telco.csv data
        config_path: Path to config.yaml
        retention_rate: The retention rate used to generate the feedback batch (for metadata).
            If None, will be inferred from the feedback batch.

    Returns:
        Metadata dict for v2 model, including synthetic data disclosure
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    data_path = Path(data_path)
    config_path = Path(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)

    # Load original data and split indices
    df = clean_data(load_raw(data_path))
    train_idx, test_idx = load_split_indices(model_dir)

    # Extract original train set
    X_train_original = df.loc[train_idx].drop(columns=["customerID", "Churn"]).copy()
    y_train_original = df.loc[train_idx, "Churn"].copy()

    # Prepare feedback batch: remove data_source column, align with original columns
    feedback_batch_clean = feedback_batch.drop(columns=["data_source"], errors="ignore").copy()

    X_feedback = feedback_batch_clean.drop(columns=["customerID", "Churn"]).copy()
    y_feedback = feedback_batch_clean["Churn"].copy()

    # Infer retention rate if not provided
    if retention_rate is None:
        retention_rate = (y_feedback == "No").mean()

    # Append feedback to train set
    X_train_combined = pd.concat([X_train_original, X_feedback], ignore_index=True)
    y_train_combined = pd.concat([y_train_original, y_feedback], ignore_index=True)

    # Fit encoders on combined training data (includes feedback)
    categorical_columns = [col for col in X_train_combined.columns if X_train_combined[col].dtype == "object"]
    encoders = fit_categorical_encoders(X_train_combined, categorical_columns)

    # Encode training data
    X_train_encoded = transform_categorical_features(X_train_combined, encoders)

    # Get feature names from original v1 model
    with open(model_dir / "split_indices.json", "r", encoding="utf-8") as f:
        v1_split_info = json.load(f)
        feature_names = v1_split_info["feature_names"]

    X_train_encoded = X_train_encoded.reindex(columns=feature_names, fill_value=0)

    # Evaluate on original test set for comparison
    X_test = df.loc[test_idx].drop(columns=["customerID", "Churn"]).copy()
    y_test = df.loc[test_idx, "Churn"].copy()
    X_test_encoded = transform_categorical_features(X_test, encoders).reindex(columns=feature_names, fill_value=0)

    # Retrain model with same hyperparameters as v1
    model_config = config.get("model", {})
    model = GradientBoostingClassifier(
        n_estimators=int(model_config.get("n_estimators", 200)),
        max_depth=int(model_config.get("max_depth", 3)),
        learning_rate=float(model_config.get("learning_rate", 0.05)),
        random_state=int(model_config.get("random_state", 42)),
    )
    model.fit(X_train_encoded, y_train_combined.map({"Yes": 1, "No": 0}))

    # Evaluate on original test set
    probabilities = model.predict_proba(X_test_encoded)[:, 1]
    roc_auc = roc_auc_score(y_test.map({"Yes": 1, "No": 0}), probabilities)
    pr_auc = average_precision_score(y_test.map({"Yes": 1, "No": 0}), probabilities)

    # Save model and encoders
    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(encoders, output_dir / "encoders.pkl")

    # Save split indices (same as v1)
    split_info = {"train_idx": train_idx, "test_idx": test_idx, "feature_names": feature_names}
    (output_dir / "split_indices.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    # Create metadata with explicit synthetic data disclosure
    metadata = {
        "run_id": "v2_feedback_loop_simulation",
        "data_source": "includes synthetic_feedback_batch, NOT purely real data",
        "synthetic_data_disclosure": (
            "This model was trained on the original training set PLUS a synthetically "
            "generated feedback batch. The feedback batch contains simulated retention "
            "outcomes, not real customer interactions. This model is for demonstration "
            "of the feedback loop mechanism only and does NOT represent an actual "
            "improvement in model performance."
        ),
        "feedback_batch_info": {
            "n_customers_sampled": len(feedback_batch),
            "retention_rate_applied": float(retention_rate),
            "source": "synthetic_top_20_pct_of_test_set",
        },
        "timestamp": datetime.now().isoformat(),
        "config_used": config,
        "n_train_original": len(X_train_original),
        "n_feedback_synthetic": len(X_feedback),
        "n_train_combined": len(X_train_combined),
        "n_test": len(X_test),
        "churn_rate_train_original": float(y_train_original.eq("Yes").mean()),
        "churn_rate_feedback_synthetic": float(y_feedback.eq("Yes").mean()),
        "churn_rate_train_combined": float(y_train_combined.map({"Yes": 1, "No": 0}).mean()),
        "churn_rate_test": float(y_test.eq("Yes").mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata
