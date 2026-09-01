"""Phase 4: Backtest / Proof of Lift — measure revenue captured by each strategy.

Generalized to accept tenant_config (same pattern as prioritize.py/
business_impact.py/explain.py/recommend.py/scenario.py - read back from
model_dir/split_indices.json via src/data/split.py's load_tenant_config()
when omitted).

SANITY GATE: MIN_CHURNED_TEST_CUSTOMERS (see below) - a genuinely new gate,
not present before this generalization pass, since backtest previously only
ever ran against Telco/Banking's own thousand-plus-row test sets and never
had a reason to ask "is there enough ground truth here to compare three
strategies meaningfully at all". A tenant whose test set doesn't clear it
stays honestly ungated for backtest (same {"available": false, "reason":
...} shape as every other module), not silently shown a noisy number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices, load_tenant_config
from src.models.prioritize import get_priority_ranking

# Below this many actually-churned customers in the test set, "caught_pct"
# for each strategy is dominated by noise, not signal: catching or missing
# even a single customer swings the percentage by more than 2 points
# (1/50 = 2%), and by far more at the bottom of this range - not a
# meaningful basis for comparing three strategies against each other.
# 50 is a real, if conservative, floor within the task's own suggested
# 50-100 range: Telco's real test set has ~374 churned customers
# (0.2654 churn rate x 1409), Aurora's has ~232 (0.29 x 800) - both land
# comfortably above this floor with real room to spare, which is what a
# floor grounded in "how much does one customer's inclusion move the
# number" should look like, not a round number picked without a reason.
MIN_CHURNED_TEST_CUSTOMERS = 50


def backtest_sanity_check(n_churned_test: int) -> str | None:
    """Returns a human-readable failure reason, or None if this tenant's
    test set has enough real churned customers for a meaningful backtest
    comparison. Mirrors the floor-and-reason shape of every other module's
    sanity check (see src/models/tenant_training.py's _sanity_check() and
    per-module gates)."""
    if n_churned_test < MIN_CHURNED_TEST_CUSTOMERS:
        return (
            f"only {n_churned_test} actually-churned customers in the test set - below the "
            f"{MIN_CHURNED_TEST_CUSTOMERS} floor this needs for a meaningful strategy comparison. "
            "Catching or missing even one customer at this size swings the caught-revenue "
            "percentage by more than this project considers a stable, comparable number."
        )
    return None


def _load_backtest_context(
    model_dir: str | Path,
    data_path: str | Path,
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything a backtest needs that does NOT depend on top_pct: the test
    set, ground-truth churned revenue, and both strategies' full rankings
    (computed once - model.predict_proba() is a single vectorized call over
    the whole test set, not a per-customer loop, so re-ranking is cheap, but
    there is no reason to re-load the model/encoders/CSV from disk once per
    requested percentage when run_backtest_curve() wants many of them)."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    target_positive_value = tenant_config["target_positive_value"]
    revenue_column = tenant_config["revenue_column"]

    df_raw = load_raw(data_path, tenant_config)
    df_clean = clean_data(df_raw, tenant_config)

    train_idx, test_idx = load_split_indices(model_dir)
    df_test = df_clean.loc[test_idx].copy()

    churned_mask = df_test[target_column] == target_positive_value
    total_churned_revenue = df_test.loc[churned_mask, revenue_column].sum()

    revenue_ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path, tenant_config=tenant_config
    )
    prob_ranking = get_priority_ranking(
        strategy="probability_only", model_dir=model_dir, data_path=data_path, tenant_config=tenant_config
    )

    return {
        "df_test": df_test,
        "id_column": id_column,
        "revenue_column": revenue_column,
        "churned_mask": churned_mask,
        "total_churned_revenue": total_churned_revenue,
        "n_churned_test": int(churned_mask.sum()),
        "revenue_ranking": revenue_ranking,
        "prob_ranking": prob_ranking,
    }


def _score_at_top_pct(top_pct: float, context: dict[str, Any], random_seed: int = 42) -> dict[str, Any]:
    """Score all three strategies at one top_pct, given a pre-loaded context.
    Each call re-seeds RandomState(random_seed) fresh before drawing, exactly
    matching what a standalone run_backtest(top_pct=...) call would draw for
    the same top_pct - so results are identical whether this point came from
    run_backtest() or run_backtest_curve()."""
    df_test = context["df_test"]
    id_column = context["id_column"]
    revenue_column = context["revenue_column"]
    churned_mask = context["churned_mask"]
    total_churned_revenue = context["total_churned_revenue"]
    revenue_ranking = context["revenue_ranking"]
    prob_ranking = context["prob_ranking"]

    top_n = round(top_pct * len(df_test))

    # revenue_ranking/prob_ranking are get_priority_ranking()'s CANONICAL
    # output ("customerID" always, regardless of tenant - see that module's
    # docstring), so these two stay literal; only df_test's own columns
    # (this tenant's real id_column/revenue_column) are tenant-dynamic.
    revenue_top_ids = set(revenue_ranking.head(top_n)["customerID"].values)
    revenue_caught_mask = (df_test[id_column].isin(revenue_top_ids)) & churned_mask
    revenue_caught = df_test.loc[revenue_caught_mask, revenue_column].sum()

    prob_top_ids = set(prob_ranking.head(top_n)["customerID"].values)
    prob_caught_mask = (df_test[id_column].isin(prob_top_ids)) & churned_mask
    prob_caught = df_test.loc[prob_caught_mask, revenue_column].sum()

    rng = __import__("numpy").random.RandomState(random_seed)
    all_customer_ids = df_test[id_column].values
    random_top_ids = set(rng.choice(all_customer_ids, size=top_n, replace=False))
    random_caught_mask = (df_test[id_column].isin(random_top_ids)) & churned_mask
    random_caught = df_test.loc[random_caught_mask, revenue_column].sum()

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


def run_backtest(
    top_pct: float = 0.2,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    random_seed: int = 42,
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run backtest comparing three prioritization strategies on revenue capture.

    Uses ground-truth churn labels to measure how much churned revenue each strategy
    would have caught if applied.

    Args:
        top_pct: Percentage of test customers to treat (default 0.2 = top 20%)
        model_dir: Path to directory containing saved Phase 1 artifacts
        data_path: Path to this tenant's raw customer data, with ground-truth labels
        random_seed: Random seed for reproducible random sampling
        tenant_config: id/target/revenue column names etc. - read back from
            model_dir/split_indices.json when omitted.

    Returns:
        dict with keys:
        - total_churned_revenue: Sum of the revenue column for all test customers
          who actually churned (ground truth)
        - revenue_weighted: dict with caught_revenue, caught_pct, strategy name
        - probability_only: dict with caught_revenue, caught_pct, strategy name
        - random: dict with caught_revenue, caught_pct, strategy name
        - top_n: Number of customers in each strategy's top list
        - top_pct: Percentage used
    """
    context = _load_backtest_context(model_dir, data_path, tenant_config)
    return _score_at_top_pct(top_pct, context, random_seed=random_seed)


def run_backtest_curve(
    top_pcts: list[float],
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    random_seed: int = 42,
    tenant_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Same computation as run_backtest(), scored at many top_pct values in
    one pass - loads the model/encoders/CSV and ranks the test set exactly
    once, then re-uses that for every requested percentage, rather than
    paying the full run_backtest() cost once per point. Powers the
    Dashboard's interactive ROI-by-%-treated slider with real, exact
    (not interpolated) numbers at every point the user can drag to."""
    context = _load_backtest_context(model_dir, data_path, tenant_config)
    return [_score_at_top_pct(top_pct, context, random_seed=random_seed) for top_pct in top_pcts]


def count_churned_test_customers(
    model_dir: str | Path,
    data_path: str | Path,
    tenant_config: dict[str, Any] | None = None,
) -> int:
    """Real count of actually-churned customers in this tenant's test set -
    what backtest_sanity_check() gates on. Reuses _load_backtest_context()
    (the same data/split loading run_backtest() itself uses) rather than
    duplicating that logic - deliberately a separate function instead of a
    new key on run_backtest()'s own return dict, since that dict's exact
    key set is an existing, asserted test contract (tests/test_backtest.py's
    test_backtest_returns_expected_structure)."""
    context = _load_backtest_context(model_dir, data_path, tenant_config)
    return context["n_churned_test"]
