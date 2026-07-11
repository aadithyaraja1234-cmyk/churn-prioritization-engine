from pathlib import Path

from src.models.clv import train_clv_model

ROOT = Path(__file__).resolve().parents[1]


def test_r2_meaningfully_above_zero():
    result = train_clv_model(
        model_dir=ROOT / "models" / "v1_enriched",
        data_path=ROOT / "data" / "raw" / "telco_enriched.csv",
    )
    assert result["r2"] > 0.1


def test_r2_below_leakage_sanity_ceiling():
    # Same pattern as the survival model's <0.90 check: CLTV's correlation
    # with MonthlyCharges/tenure/TotalCharges is well below 0.9, so nothing
    # was excluded on leakage grounds, but this still guards against a
    # future change accidentally reintroducing a near-deterministic feature.
    result = train_clv_model(
        model_dir=ROOT / "models" / "v1_enriched",
        data_path=ROOT / "data" / "raw" / "telco_enriched.csv",
    )
    assert result["r2"] < 0.95
