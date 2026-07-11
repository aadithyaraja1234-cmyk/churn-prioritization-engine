"""Tests for Phase 4 backtest module."""

from pathlib import Path

from src.models.backtest import run_backtest


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

