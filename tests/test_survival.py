from pathlib import Path

from src.models.survival import train_survival_model

ROOT = Path(__file__).resolve().parents[1]


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
