"""Tests for Phase 4 backtest module."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.backtest import MIN_CHURNED_TEST_CUSTOMERS, backtest_sanity_check, run_backtest, run_backtest_curve

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def test_backtest_returns_expected_structure():
    """Test that run_backtest returns a dict with expected keys and structure."""
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    result = run_backtest(top_pct=0.2, model_dir=model_dir, data_path=data_path)

    # Check top-level keys
    expected_keys = {
        "total_churned_revenue",
        "top_n",
        "top_pct",
        "revenue_weighted",
        "probability_only",
        "random",
    }
    assert set(result.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(result.keys())}"

    # Check each strategy dict has the right keys
    for strategy_name in ["revenue_weighted", "probability_only", "random"]:
        strategy_result = result[strategy_name]
        expected_strategy_keys = {"strategy", "caught_revenue", "caught_pct"}
        assert set(strategy_result.keys()) == expected_strategy_keys, (
            f"Strategy {strategy_name} missing keys: "
            f"expected {expected_strategy_keys}, got {set(strategy_result.keys())}"
        )


def test_revenue_weighted_beats_random():
    """
    Test that revenue-weighted strategy catches significantly more churned revenue
    than random selection (at least 15 percentage points higher).

    This is a key proof-of-lift: the model-driven strategy should outperform
    random baseline by a meaningful margin.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    result = run_backtest(top_pct=0.2, model_dir=model_dir, data_path=data_path)

    revenue_pct = result["revenue_weighted"]["caught_pct"]
    random_pct = result["random"]["caught_pct"]

    assert revenue_pct > random_pct + 15, (
        f"Revenue-weighted ({revenue_pct:.2f}%) should beat random ({random_pct:.2f}%) "
        f"by at least 15 percentage points"
    )


def test_probability_only_beats_random():
    """
    Test that probability-only strategy also beats random baseline.

    This is a sanity check: the underlying model should have signal,
    even without revenue weighting.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    result = run_backtest(top_pct=0.2, model_dir=model_dir, data_path=data_path)

    prob_pct = result["probability_only"]["caught_pct"]
    random_pct = result["random"]["caught_pct"]

    assert prob_pct > random_pct + 15, (
        f"Probability-only ({prob_pct:.2f}%) should beat random ({random_pct:.2f}%) "
        f"by at least 15 percentage points"
    )


def test_revenue_weighted_beats_probability_only():
    """
    CORE PROJECT CLAIM: Revenue-weighted strategy catches more churned revenue
    than probability-only strategy (the fundamental business case for the model).
    
    This regression test locks in that revenue-weighting adds measurable value
    on top of the underlying churn model.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    result = run_backtest(top_pct=0.2, model_dir=model_dir, data_path=data_path)

    revenue_pct = result["revenue_weighted"]["caught_pct"]
    prob_pct = result["probability_only"]["caught_pct"]

    assert revenue_pct > prob_pct, (
        f"Revenue-weighted ({revenue_pct:.2f}%) must beat probability-only ({prob_pct:.2f}%); "
        f"this is the core business claim that revenue-weighting creates value"
    )


def test_backtest_identifies_best_performer():
    """
    Identify and print which strategy performed best in this backtest run.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    result = run_backtest(top_pct=0.2, model_dir=model_dir, data_path=data_path)

    revenue_pct = result["revenue_weighted"]["caught_pct"]
    prob_pct = result["probability_only"]["caught_pct"]
    random_pct = result["random"]["caught_pct"]

    # Determine winner
    strategies = {
        "revenue_weighted": revenue_pct,
        "probability_only": prob_pct,
        "random": random_pct,
    }
    winner = max(strategies, key=strategies.get)
    winning_pct = strategies[winner]

    # Print results
    print()
    print("=" * 100)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 100)
    print(f"Total test customers: 1409")
    print(f"Top {result['top_pct']:.1%} = {result['top_n']} customers")
    print()
    print(f"Total churned revenue (ground truth): ${result['total_churned_revenue']:.2f}")
    print()
    print("Strategy Performance:")
    print(f"  Revenue-Weighted:  ${result['revenue_weighted']['caught_revenue']:.2f}  ({revenue_pct:.2f}%)")
    print(f"  Probability-Only:  ${result['probability_only']['caught_revenue']:.2f}  ({prob_pct:.2f}%)")
    print(f"  Random:            ${result['random']['caught_revenue']:.2f}  ({random_pct:.2f}%)")
    print()
    print(f"✓ BEST PERFORMER: {winner.upper()} ({winning_pct:.2f}%)")
    print("=" * 100)
    print()


def test_backtest_robustness_across_thresholds():
    """
    Test backtest robustness: verify revenue-weighted consistently beats
    probability-only across multiple top_pct thresholds (0.1, 0.2, 0.3).
    
    If revenue-weighted only wins at one threshold but loses at others,
    this test will identify that instability explicitly.
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    thresholds = [0.1, 0.2, 0.3]
    results_by_threshold = {}

    for top_pct in thresholds:
        result = run_backtest(top_pct=top_pct, model_dir=model_dir, data_path=data_path)
        results_by_threshold[top_pct] = {
            "revenue_weighted_pct": result["revenue_weighted"]["caught_pct"],
            "probability_only_pct": result["probability_only"]["caught_pct"],
            "random_pct": result["random"]["caught_pct"],
        }

    # Print results table
    print()
    print("=" * 100)
    print("ROBUSTNESS TEST: BACKTEST PERFORMANCE ACROSS THRESHOLDS")
    print("=" * 100)
    print()
    print(f"{'Top %':>8} {'Strategy':>20} {'Caught %':>12} {'vs Random':>12} {'vs Prob-Only':>15}")
    print("-" * 100)

    for top_pct in thresholds:
        metrics = results_by_threshold[top_pct]
        rev_pct = metrics["revenue_weighted_pct"]
        prob_pct = metrics["probability_only_pct"]
        rand_pct = metrics["random_pct"]

        rev_vs_rand = rev_pct - rand_pct
        rev_vs_prob = rev_pct - prob_pct

        print(f"{top_pct:>7.0%} {'Revenue-Weighted':>20} {rev_pct:>11.2f}% {rev_vs_rand:>+11.2f}pp {rev_vs_prob:>+14.2f}pp")
        print(f"{'':>8} {'Probability-Only':>20} {prob_pct:>11.2f}% {prob_pct - rand_pct:>+11.2f}pp {'':>15}")
        print(f"{'':>8} {'Random':>20} {rand_pct:>11.2f}% {'':>12} {'':>15}")
        print("-" * 100)

    print()

    # Check consistency: revenue_weighted should beat probability_only at ALL thresholds
    inconsistent_thresholds = []
    for top_pct in thresholds:
        metrics = results_by_threshold[top_pct]
        if metrics["revenue_weighted_pct"] <= metrics["probability_only_pct"]:
            inconsistent_thresholds.append(top_pct)

    if inconsistent_thresholds:
        print(f"⚠️  INCONSISTENCY DETECTED: Revenue-weighted loses to probability-only at threshold(s): {inconsistent_thresholds}")
    else:
        print("✓ CONSISTENT: Revenue-weighted beats probability-only at ALL thresholds (0.1, 0.2, 0.3)")

    print("=" * 100)
    print()

    # Assert consistency
    assert len(inconsistent_thresholds) == 0, (
        f"Revenue-weighted should beat probability-only at all thresholds, "
        f"but lost at: {inconsistent_thresholds}"
    )


def test_backtest_curve_matches_individual_run_backtest_calls():
    """The whole point of run_backtest_curve() is a performance optimization
    (rank the test set once, reuse it for every requested percentage) - this
    locks in that it's ONLY a performance optimization, not a behavior
    change, by checking a handful of points against independent
    run_backtest(top_pct=...) calls."""
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    sample_pcts = [0.05, 0.2, 0.5, 0.9]
    curve = run_backtest_curve(sample_pcts, model_dir=model_dir, data_path=data_path)

    assert len(curve) == len(sample_pcts)
    for top_pct, curve_point in zip(sample_pcts, curve):
        individual = run_backtest(top_pct=top_pct, model_dir=model_dir, data_path=data_path)
        assert curve_point == individual


def test_backtest_curve_returns_one_point_per_requested_pct():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    top_pcts = [pct / 100 for pct in range(1, 101)]
    curve = run_backtest_curve(top_pcts, model_dir=model_dir, data_path=data_path)

    assert len(curve) == 100
    assert curve[0]["top_pct"] == 0.01
    assert curve[-1]["top_pct"] == 1.0
    # At 100% treated, every strategy necessarily catches all churned revenue.
    assert curve[-1]["revenue_weighted"]["caught_pct"] == 100.0
    assert curve[-1]["probability_only"]["caught_pct"] == 100.0
    assert curve[-1]["random"]["caught_pct"] == 100.0


def test_revenue_weighted_does_not_always_beat_probability_only_across_full_range():
    """Documents a real, honest finding surfaced by building the full 1-100%
    curve (not just the 10/20/30% thresholds test_backtest_robustness_
    across_thresholds checks): the core project claim holds at the
    specifically-validated thresholds and across most of the range, but not
    at literally every percentage. This test exists so that fact stays
    visible and intentional, not an unnoticed regression - if this ever
    starts failing (the two strategies never diverge at all), that's a
    signal something about the model or ranking changed, not that the test
    is wrong."""
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "raw" / "telco.csv"
    model_dir = repo_root / "models" / "v1"

    top_pcts = [pct / 100 for pct in range(1, 101)]
    curve = run_backtest_curve(top_pcts, model_dir=model_dir, data_path=data_path)

    trailing_pcts = [
        point["top_pct"]
        for point in curve
        if point["revenue_weighted"]["caught_pct"] < point["probability_only"]["caught_pct"]
    ]
    assert len(trailing_pcts) > 0, (
        "Expected revenue-weighted to trail probability-only at a handful of untested "
        "percentages (this was true when the full curve was first built) - if this "
        "assertion fails because revenue-weighted now wins everywhere, that's a real "
        "improvement, not a bug; update this test's expectation rather than treating it "
        "as broken."
    )



# --- Regression: generalized to accept tenant_config (same recurring
# hardcoded-Telco-column bug this session already found in business_impact.py/
# recommend.py) - AND a genuinely new sanity gate (MIN_CHURNED_TEST_CUSTOMERS)
# this generalization pass introduced. ---

AURORA_MODEL_DIR = "models/aurora-streaming"
AURORA_DATA_PATH = "data/tenant_uploads/aurora-streaming.csv"


def test_backtest_sanity_check_floor():
    assert backtest_sanity_check(MIN_CHURNED_TEST_CUSTOMERS) is None
    assert backtest_sanity_check(MIN_CHURNED_TEST_CUSTOMERS - 1) is not None
    assert backtest_sanity_check(0) is not None


def test_aurora_backtest_loads_with_genuinely_different_column_names():
    """End-to-end against Aurora's real trained artifacts - id/target/
    revenue columns (customer_id/churned/monthly_fee) share NOT ONE literal
    name with Telco's (customerID/Churn/MonthlyCharges). Used to crash
    outright (KeyError on "Churn"/"MonthlyCharges"/"customerID"); must now
    return a real result, and Aurora's real test set (~232 churned
    customers) comfortably clears the new sanity gate."""
    result = run_backtest(top_pct=0.2, model_dir=AURORA_MODEL_DIR, data_path=AURORA_DATA_PATH)

    assert result["total_churned_revenue"] > 0
    for strategy in ("revenue_weighted", "probability_only", "random"):
        assert 0.0 <= result[strategy]["caught_pct"] <= 100.0

    from src.data.split import load_tenant_config
    from src.data.clean import clean_data
    from src.data.load import load_raw
    from src.data.split import load_split_indices

    tenant_config = load_tenant_config(AURORA_MODEL_DIR)
    df = clean_data(load_raw(AURORA_DATA_PATH, tenant_config), tenant_config)
    _, test_idx = load_split_indices(AURORA_MODEL_DIR)
    n_churned = int((df.loc[test_idx, tenant_config["target_column"]] == tenant_config["target_positive_value"]).sum())
    assert backtest_sanity_check(n_churned) is None, "Aurora's real test set should clear the backtest sanity gate"


def test_backtest_for_a_freshly_trained_tenant_with_non_telco_column_names(tmp_path):
    """Self-contained variant (trains its own tiny synthetic tenant in
    tmp_path) so this specific guard runs anywhere - CI included."""
    import numpy as np

    from src.models.train import train_model

    rng = np.random.default_rng(13)
    n = 400
    tenure_like = rng.integers(0, 60, size=n)
    fee = np.round(rng.normal(20, 5, size=n).clip(5, 40), 2)
    logit = -0.3 - 0.02 * tenure_like + rng.normal(0, 0.6, size=n)
    left = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame(
        {
            "acct_id": [f"A{i:04d}" for i in range(n)],
            "months_active": tenure_like,
            "fee_amount": fee,
            "left": left,
        }
    )
    data_path = tmp_path / "synthetic.csv"
    df.to_csv(data_path, index=False)

    tenant_config = {
        "id_column": "acct_id",
        "target_column": "left",
        "target_positive_value": "Yes",
        "revenue_column": "fee_amount",
        "tuning_enabled": False,
    }
    model_dir = tmp_path / "model"
    train_model(data_path, CONFIG_PATH, output_dir=model_dir, tenant_config=tenant_config)

    result = run_backtest(top_pct=0.2, model_dir=model_dir, data_path=data_path)
    assert result["total_churned_revenue"] > 0
    for strategy in ("revenue_weighted", "probability_only", "random"):
        assert 0.0 <= result[strategy]["caught_pct"] <= 100.0

    curve = run_backtest_curve([0.1, 0.2, 0.3], model_dir=model_dir, data_path=data_path)
    assert len(curve) == 3
