"""Phase 4: Backtest / Proof of Lift — measure revenue captured by each strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices
from src.models.prioritize import get_priority_ranking


def run_backtest(
    top_pct: float = 0.2,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run backtest comparing three prioritization strategies on revenue capture.

    Uses ground-truth Churn labels to measure how much churned revenue each strategy
    would have caught if applied.

    Args:
        top_pct: Percentage of test customers to treat (default 0.2 = top 20%)
        model_dir: Path to directory containing saved Phase 1 artifacts
        data_path: Path to raw telco.csv data with ground-truth Churn labels
        random_seed: Random seed for reproducible random sampling

    Returns:
        dict with keys:
        - total_churned_revenue: Sum of MonthlyCharges for all test customers
          who actually churned (ground truth)
        - revenue_weighted: dict with caught_revenue, caught_pct, strategy name
        - probability_only: dict with caught_revenue, caught_pct, strategy name
        - random: dict with caught_revenue, caught_pct, strategy name
        - top_n: Number of customers in each strategy's top list
        - top_pct: Percentage used
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    # Load raw data with ground-truth Churn labels
    df_raw = load_raw(data_path)
    df_clean = clean_data(df_raw)

    # Load split indices to get test set
    train_idx, test_idx = load_split_indices(model_dir)

    # Extract test set with Churn labels
    df_test = df_clean.loc[test_idx].copy()

    # Compute total churned revenue (ground truth)
    churned_mask = df_test["Churn"] == "Yes"
    total_churned_revenue = df_test.loc[churned_mask, "MonthlyCharges"].sum()

    # Get top N = round(top_pct * len(test_set))
    top_n = round(top_pct * len(df_test))

    # Get all three rankings
    revenue_ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path
    )
    prob_ranking = get_priority_ranking(
        strategy="probability_only", model_dir=model_dir, data_path=data_path
    )

    # Strategy 1: Revenue-weighted
    revenue_top_ids = set(revenue_ranking.head(top_n)["customerID"].values)
    revenue_caught_mask = (df_test["customerID"].isin(revenue_top_ids)) & churned_mask
    revenue_caught = df_test.loc[revenue_caught_mask, "MonthlyCharges"].sum()

    # Strategy 2: Probability-only
    prob_top_ids = set(prob_ranking.head(top_n)["customerID"].values)
    prob_caught_mask = (df_test["customerID"].isin(prob_top_ids)) & churned_mask
    prob_caught = df_test.loc[prob_caught_mask, "MonthlyCharges"].sum()

    # Strategy 3: Random
    rng = __import__("numpy").random.RandomState(random_seed)
    all_customer_ids = df_test["customerID"].values
    random_top_ids = set(rng.choice(all_customer_ids, size=top_n, replace=False))
    random_caught_mask = (df_test["customerID"].isin(random_top_ids)) & churned_mask
    random_caught = df_test.loc[random_caught_mask, "MonthlyCharges"].sum()

    # Compute percentages
    revenue_pct = 100.0 * revenue_caught / total_churned_revenue if total_churned_revenue > 0 else 0.0
    prob_pct = 100.0 * prob_caught / total_churned_revenue if total_churned_revenue > 0 else 0.0
    random_pct = 100.0 * random_caught / total_churned_revenue if total_churned_revenue > 0 else 0.0

    return {
        "total_churned_revenue": float(total_churned_revenue),
        "top_n": int(top_n),
        "top_pct": float(top_pct),
        "revenue_weighted": {
            "strategy": "revenue_weighted",
            "caught_revenue": float(revenue_caught),
            "caught_pct": float(revenue_pct),
        },
        "probability_only": {
            "strategy": "probability_only",
            "caught_revenue": float(prob_caught),
            "caught_pct": float(prob_pct),
        },
        "random": {
            "strategy": "random",
            "caught_revenue": float(random_caught),
            "caught_pct": float(random_pct),
        },
    }
