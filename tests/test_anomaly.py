from pathlib import Path

from src.models.anomaly import detect_anomalies

ROOT = Path(__file__).resolve().parents[1]

# 0.05 is the finalized choice: top anomalies are rank-stable across
# contamination settings, and 0.05 already captures the cleanest, most
# distinctive outliers (a coherent no-phone-service/DSL-only/long-tenure
# profile) without pulling in the noisier border cases seen at 0.10.
CHOSEN_CONTAMINATION = 0.05


def test_churn_and_customer_id_excluded_from_features():
    result = detect_anomalies(
        contamination=CHOSEN_CONTAMINATION,
        model_dir=ROOT / "models" / "v1",
        data_path=ROOT / "data" / "raw" / "telco.csv",
    )

    assert "Churn" not in result["feature_columns"]
    assert "customerID" not in result["feature_columns"]


def test_flagged_count_roughly_matches_contamination_rate():
    result = detect_anomalies(
        contamination=CHOSEN_CONTAMINATION,
        model_dir=ROOT / "models" / "v1",
        data_path=ROOT / "data" / "raw" / "telco.csv",
    )

    results = result["results"]
    n_flagged = (results["anomaly_flag"] == -1).sum()
    expected = CHOSEN_CONTAMINATION * len(results)

    assert abs(n_flagged - expected) / expected < 0.1


def test_every_customer_gets_exactly_one_flag():
    result = detect_anomalies(
        contamination=CHOSEN_CONTAMINATION,
        model_dir=ROOT / "models" / "v1",
        data_path=ROOT / "data" / "raw" / "telco.csv",
    )

    results = result["results"]
    assert results["anomaly_flag"].isna().sum() == 0
    assert set(results["anomaly_flag"].unique()).issubset({-1, 1})
    assert results["customerID"].is_unique
    assert len(results) == 7043
