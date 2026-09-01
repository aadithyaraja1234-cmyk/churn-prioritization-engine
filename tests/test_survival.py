import random
from pathlib import Path

from src.models.survival import train_survival_model
from src.models.train import train_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def test_c_index_meaningfully_above_random():
    result = train_survival_model(model_dir=ROOT / "models" / "v1", data_path=ROOT / "data" / "raw" / "telco.csv")
    assert result["c_index"] > 0.55


def test_c_index_not_suspiciously_high():
    # With TotalCharges excluded (it leaks duration = tenure * MonthlyCharges
    # back in), the true c-index is ~0.868, matching published benchmarks
    # (0.83-0.89) for this dataset. 0.90 leaves headroom above that while
    # still catching leakage reintroduced by a future change (it previously
    # inflated the score to 0.931).
    result = train_survival_model(model_dir=ROOT / "models" / "v1", data_path=ROOT / "data" / "raw" / "telco.csv")
    assert result["c_index"] < 0.90


def test_contract_is_a_significant_hazard_factor():
    result = train_survival_model(model_dir=ROOT / "models" / "v1", data_path=ROOT / "data" / "raw" / "telco.csv")

    coefficients = result["coefficients"]
    assert "Contract" in coefficients

    ranked_by_magnitude = sorted(coefficients.items(), key=lambda item: abs(item[1] - 1.0), reverse=True)
    top_feature, _ = ranked_by_magnitude[0]
    assert top_feature == "Contract"


def _make_totalcharges_shaped_leak_csv(n_rows: int, seed: int) -> str:
    """A synthetic dataset with a planted TotalCharges-shaped leak: leak_col
    = duration * revenue + tiny noise - the exact mathematical relationship
    that inflated Telco's own real c-index from 0.868 to 0.931 before it
    was caught and excluded (see src/models/survival.py's module
    docstring). No duration_leakage_column is mapped for this tenant (see
    the test below) - the ONLY thing that should catch this is
    train_survival_model()'s automatic correlation-based fallback."""
    rng = random.Random(seed)
    lines = ["customer_id,target,revenue,duration,leak_col,other_feature"]
    for i in range(n_rows):
        target = "Yes" if i % 2 == 0 else "No"
        revenue = round(rng.uniform(20, 120), 2)
        duration = round(rng.uniform(1, 72), 2)
        leak_col = round(duration * revenue + rng.uniform(-0.5, 0.5), 4)
        other_feature = round(rng.uniform(0, 1), 4)
        lines.append(f"C{i},{target},{revenue},{duration},{leak_col},{other_feature}")
    return "\n".join(lines) + "\n"


def test_planted_totalcharges_shaped_leak_is_automatically_caught_and_excluded(tmp_path):
    """Regression test for the exact failure class this project has now
    hit twice: a covariate that's approximately duration * revenue
    inflates c-index into "too good to be true" territory if left in (see
    module docstring). No duration_leakage_column is mapped in
    tenant_config below - proving the AUTOMATIC fallback (not a human
    picking the right column) is what catches this."""
    csv_text = _make_totalcharges_shaped_leak_csv(n_rows=600, seed=7)
    data_path = tmp_path / "leak.csv"
    data_path.write_text(csv_text, encoding="utf-8")

    tenant_config = {
        "id_column": "customer_id",
        "target_column": "target",
        "target_positive_value": "Yes",
        "revenue_column": "revenue",
        "duration_column": "duration",
        # deliberately no "duration_leakage_column" key at all
    }

    model_dir = tmp_path / "model"
    train_model(data_path=data_path, config_path=CONFIG_PATH, output_dir=model_dir, tenant_config=tenant_config)

    result = train_survival_model(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)

    # Caught, named, and excluded - not silently trusted.
    assert "duration_leakage_warning" in result
    assert "leak_col" in result["duration_leakage_warning"]
    assert "leak_col" not in result["coefficients"]
    assert "other_feature" in result["coefficients"]  # a real, unrelated feature is untouched

    # With the leak excluded, c-index lands back in a plausible, non-
    # "too good to be true" range - not the near-perfect score leak_col
    # alone would produce (duration is reconstructible almost exactly from
    # leak_col/revenue).
    assert 0.45 < result["c_index"] < 0.90
