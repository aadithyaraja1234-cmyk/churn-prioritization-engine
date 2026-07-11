"""Tests for Phase 3 business impact engine (prioritization module)."""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.models.prioritize import get_priority_ranking


def test_priority_ranking_returns_all_test_customers_revenue_weighted():
    """Test that revenue_weighted strategy returns all test customers with expected columns."""
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path
    )

    # Should return all test customers
    assert len(ranking) == 1409, f"Expected 1409 test customers, got {len(ranking)}"

    # Check expected columns
    expected_columns = {
        "customerID",
        "churn_probability",
        "MonthlyCharges",
        "expected_revenue_at_risk",
        "rank",
    }
    assert set(ranking.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {set(ranking.columns)}"
    )

    # Validate data types and ranges
    assert ranking["churn_probability"].min() >= 0.0
    assert ranking["churn_probability"].max() <= 1.0
    assert ranking["MonthlyCharges"].min() >= 0.0
    assert ranking["expected_revenue_at_risk"].min() >= 0.0
    assert (ranking["rank"] == range(1, 1410)).all(), "Ranks should be 1 to 1409"


def test_priority_ranking_returns_all_test_customers_probability_only():
    """Test that probability_only strategy returns all test customers with expected columns."""
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    ranking = get_priority_ranking(
        strategy="probability_only", model_dir=model_dir, data_path=data_path
    )

    # Should return all test customers
    assert len(ranking) == 1409, f"Expected 1409 test customers, got {len(ranking)}"

    # Check expected columns
    expected_columns = {
        "customerID",
        "churn_probability",
        "MonthlyCharges",
        "expected_revenue_at_risk",
        "rank",
    }
    assert set(ranking.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {set(ranking.columns)}"
    )


def test_revenue_weighted_and_probability_only_rankings_differ_meaningfully():
    """
    Test that revenue-weighted and probability-only strategies produce
    meaningfully different rankings (Spearman correlation < 0.98).

    This locks in the finding that the two strategies are not nearly identical.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    revenue_ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path
    )
    prob_ranking = get_priority_ranking(
        strategy="probability_only", model_dir=model_dir, data_path=data_path
    )

    # Create mapping of customerID to rank for each strategy
    revenue_rank_dict = dict(zip(revenue_ranking["customerID"], revenue_ranking["rank"]))
    prob_rank_dict = dict(zip(prob_ranking["customerID"], prob_ranking["rank"]))

    # Extract ranks in same order
    all_customer_ids = revenue_ranking["customerID"].values
    revenue_ranks = [revenue_rank_dict[cid] for cid in all_customer_ids]
    prob_ranks = [prob_rank_dict[cid] for cid in all_customer_ids]

    # Compute Spearman rank correlation
    correlation, pvalue = spearmanr(revenue_ranks, prob_ranks)

    # Assert meaningful differentiation (correlation < 0.98)
    assert (
        correlation < 0.98
    ), f"Correlation {correlation:.4f} is too high; strategies are too similar"

    # The observed correlation should be around 0.949
    assert (
        0.94 < correlation < 0.96
    ), f"Expected correlation ~0.949, got {correlation:.4f}"


def test_revenue_weighted_has_new_top20_entrants_vs_probability_only():
    """
    Test that revenue-weighted strategy brings new customers into the actionable top 20
    that were not in probability_only's top 20.

    This is the core business claim: revenue-weighted prioritization changes WHO you act on,
    not just the order. This regression test locks in that finding.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    revenue_ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path
    )
    prob_ranking = get_priority_ranking(
        strategy="probability_only", model_dir=model_dir, data_path=data_path
    )

    # Get top 20 for each strategy
    revenue_top20 = set(revenue_ranking.head(20)["customerID"].values)
    prob_top20 = set(prob_ranking.head(20)["customerID"].values)

    # Compute new entrants (in revenue-weighted but not in probability-only)
    new_entrants = revenue_top20 - prob_top20

    # Assert at least 1 new entrant (the finding was 12)
    assert (
        len(new_entrants) >= 1
    ), "Revenue-weighted strategy should bring at least 1 new customer into top 20"

    # The observed value should be around 12
    assert len(new_entrants) >= 10, (
        f"Expected ~12 new entrants, got {len(new_entrants)}; "
        f"actionable set change may be degraded"
    )
