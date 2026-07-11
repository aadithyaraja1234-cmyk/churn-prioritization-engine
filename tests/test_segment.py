from pathlib import Path

from src.models.segment import run_segmentation

ROOT = Path(__file__).resolve().parents[1]

# k=4 is the finalized, deliberate choice for this segmentation, not a
# silhouette-search default. k=2 and k=4 are silhouette-tied (0.1569 vs
# 0.1553), but k=4 surfaces a real second axis (spend level) that k=2
# collapses: it splits the high-churn Month-to-month bucket into a 45.9%
# and a 25.7% churn group, and the low-churn Two-year bucket into a 7.4%
# and a 15.5% churn group — a materially more informative split than k=2's
# single 34.6%/13.7% divide. Tests below hardcode k=4 accordingly.
CHOSEN_K = 4


def test_churn_and_customer_id_excluded_from_features():
    result = run_segmentation(k=CHOSEN_K, model_dir=ROOT / "models" / "v1", data_path=ROOT / "data" / "raw" / "telco.csv")

    assert "Churn" not in result["feature_columns"]
    assert "customerID" not in result["feature_columns"]


def test_number_of_clusters_matches_chosen_k():
    result = run_segmentation(k=CHOSEN_K, model_dir=ROOT / "models" / "v1", data_path=ROOT / "data" / "raw" / "telco.csv")

    assert result["chosen_k"] == CHOSEN_K
    assert result["cluster_assignments"]["cluster"].nunique() == CHOSEN_K


def test_every_customer_assigned_exactly_once():
    result = run_segmentation(k=CHOSEN_K, model_dir=ROOT / "models" / "v1", data_path=ROOT / "data" / "raw" / "telco.csv")

    assignments = result["cluster_assignments"]
    assert assignments["cluster"].isna().sum() == 0
    assert assignments["customerID"].is_unique
    assert len(assignments) == 7043
