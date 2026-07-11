from pathlib import Path

from src.data.clean import clean_data
from src.data.load import load_raw


def test_load_raw_validates_and_reads_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
    df = load_raw(data_path)

    assert {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}.issubset(df.columns)
    assert df["Churn"].isin(["Yes", "No"]).all()
    assert (df["tenure"] >= 0).all()
    assert df["customerID"].is_unique


def test_clean_data_fills_total_charges_for_zero_tenure_rows():
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"
    raw_df = load_raw(data_path)
    cleaned = clean_data(raw_df)

    zero_tenure_mask = (cleaned["tenure"] == 0) & cleaned["TotalCharges"].isna()
    assert not zero_tenure_mask.any()

    zero_tenure_rows = cleaned.loc[cleaned["tenure"] == 0, ["MonthlyCharges", "TotalCharges"]]
    assert (zero_tenure_rows["TotalCharges"] == zero_tenure_rows["MonthlyCharges"]).all()
