"""Business impact engine: prioritize customers by revenue at risk.

Generalized to accept tenant_config (same pattern as survival.py/
segment.py/anomaly.py/clv.py): every hardcoded Telco literal
("customerID"/"Churn"/"MonthlyCharges") has been replaced by a
tenant_config-driven lookup, defaulting to DEFAULT_TENANT_CONFIG (Telco's
exact literal column names) when no tenant_config is given - so nothing
changes for a caller that doesn't pass one. When tenant_config is None,
it's read back from model_dir/split_indices.json via
src/data/split.py's load_tenant_config() - train_model() now saves it
there at training time - rather than requiring every call site to
re-resolve or hardcode it, the same "read it back from the artifact"
pattern the other four generalized modules already use for their own
pickles.

The OUTPUT DataFrame's column names ("customerID", "MonthlyCharges",
"churn_probability", "expected_revenue_at_risk", "rank") are deliberately
NOT tenant-dynamic, unlike segment.py/anomaly.py's id_column-keyed
outputs - this is a stable, canonical contract every downstream consumer
(business_impact.py, backtest.py, scenario.py, tests/test_prioritize.py,
the priority-ranking API endpoint, the frontend) already depends on by
these literal names, for every tenant alike. Only the INPUT reads are
tenant-aware; the output schema never changes.

SANITY GATE: priority_ranking_sanity_check() below - a genuinely new gate,
added when priority_ranking was wired into the per-tenant training pipeline
alongside backtest/scenario_simulator/customer_timeline. Reuses this
project's own existing precedent for "meaningfully different" rather than
inventing a new threshold: tests/test_prioritize.py's
test_revenue_weighted_and_probability_only_rankings_differ_meaningfully()
already asserts Telco's two strategies correlate at Spearman < 0.98 (real
value: ~0.9463) and calls anything above that "too similar" to be worth
offering as two distinct strategies. The gate applies that exact same bar
to any tenant: if revenue-weighted and probability-only rankings correlate
above 0.98, the revenue column isn't meaningfully reordering anything
relative to churn_probability alone (e.g. because it has too little
variance, or too little relationship to who's actually flagged as
high-risk) - "revenue-weighted prioritization" would be a relabeled
duplicate of probability-only, not a real second option.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.stats import spearmanr

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_tenant_config
from src.features.encode import transform_categorical_features

# See module docstring's SANITY GATE note - matches
# tests/test_prioritize.py's own existing "too similar" bar for Telco.
MAX_SANE_REVENUE_PROBABILITY_CORRELATION = 0.98


def get_priority_ranking(
    strategy: str = "revenue_weighted",
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Rank test customers by revenue at risk or churn probability.

    Args:
        strategy: One of "revenue_weighted" (default) or "probability_only".
            - "revenue_weighted": Sort by expected_revenue_at_risk (churn_prob * revenue)
            - "probability_only": Sort by churn_probability only
        model_dir: Path to directory containing saved Phase 1 artifacts
        data_path: Path to this tenant's raw customer data
        tenant_config: id/target/revenue column names etc. - see
            src/data/load.py's DEFAULT_TENANT_CONFIG. Read back from
            model_dir/split_indices.json when omitted (None).

    Returns:
        DataFrame with columns (same names for every tenant - see module
        docstring):
        - customerID: Unique customer identifier
        - churn_probability: Model's predicted churn probability [0, 1]
        - MonthlyCharges: Customer's revenue column value
        - expected_revenue_at_risk: churn_probability * MonthlyCharges
        - rank: 1-indexed rank according to the chosen strategy
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    if strategy not in ("revenue_weighted", "probability_only"):
        raise ValueError(f"strategy must be 'revenue_weighted' or 'probability_only', got {strategy}")

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    target_positive_value = tenant_config["target_positive_value"]
    revenue_column = tenant_config["revenue_column"]

    # Load Phase 1 artifacts
    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    with open(model_dir / "split_indices.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)
        test_idx = split_info["test_idx"]
        feature_names = split_info["feature_names"]

    # Load raw data
    df = clean_data(load_raw(data_path, tenant_config), tenant_config)

    # Extract test set
    X_test = df.loc[test_idx].drop(columns=[id_column, target_column]).copy()
    customer_ids = df.loc[test_idx, id_column].copy()
    revenue = df.loc[test_idx, revenue_column].copy()

    # Encode test features
    X_test_encoded = transform_categorical_features(X_test, encoders).reindex(
        columns=feature_names, fill_value=0
    )

    # Get churn probabilities - looks up target_positive_value's actual
    # position in the fitted model's own classes_ rather than assuming
    # index 1, the same discipline src/models/train.py's
    # _positive_class_scorer() already uses for exactly this reason: sklearn
    # sorts classes_ alphabetically/numerically, and nothing guarantees the
    # positive label lands at index 1 for every possible tenant's label
    # values (it does for Telco's "Yes"/"No" and Banking's 0/1 today, but
    # that's a coincidence of those specific label spellings, not a rule).
    positive_index = list(model.classes_).index(target_positive_value)
    churn_probs = model.predict_proba(X_test_encoded)[:, positive_index]

    # Build results DataFrame - canonical column names, see module docstring
    results = pd.DataFrame(
        {
            "customerID": customer_ids.values,
            "churn_probability": churn_probs,
            "MonthlyCharges": revenue.values,
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


def priority_ranking_sanity_check(
    model_dir: str | Path,
    data_path: str | Path,
    tenant_config: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Returns (failure_reason_or_None, metrics) - see module docstring's
    SANITY GATE note. Computes both real rankings for this tenant's own
    test set (the same call get_priority_ranking() itself makes, no
    separate code path) and checks they're not too similar to be worth
    offering as two distinct strategies."""
    revenue_ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path, tenant_config=tenant_config
    )
    prob_ranking = get_priority_ranking(
        strategy="probability_only", model_dir=model_dir, data_path=data_path, tenant_config=tenant_config
    )
    revenue_rank_by_id = dict(zip(revenue_ranking["customerID"], revenue_ranking["rank"]))
    prob_rank_by_id = dict(zip(prob_ranking["customerID"], prob_ranking["rank"]))
    all_ids = revenue_ranking["customerID"].values
    correlation, _ = spearmanr(
        [revenue_rank_by_id[cid] for cid in all_ids],
        [prob_rank_by_id[cid] for cid in all_ids],
    )
    metrics = {"revenue_vs_probability_spearman_correlation": float(correlation)}
    if correlation >= MAX_SANE_REVENUE_PROBABILITY_CORRELATION:
        return (
            f"revenue-weighted and probability-only rankings correlate at {correlation:.4f} - at or "
            f"above the {MAX_SANE_REVENUE_PROBABILITY_CORRELATION} bar this project uses for "
            "'too similar to be a meaningfully distinct strategy' (see tests/test_prioritize.py's own "
            "Telco baseline of this exact check).",
            metrics,
        )
    return None, metrics
