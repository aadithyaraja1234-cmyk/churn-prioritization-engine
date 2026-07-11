"""One-off shim: map the extended Telco dataset's column names/values onto the
original schema, without touching load.py/clean.py. Produces two CSVs:

- a slim file with exactly the original 21 columns (for the existing
  classifier pipeline, which validates against that exact column set)
- a full file that keeps every extended column (geo, CLTV, churn reason)
  for new capabilities (e.g. CLTV regression) to consume.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RENAME_MAP = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Contract": "Contract",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
}

ORIGINAL_SCHEMA_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def prepare_extended(
    source_xlsx: str | Path,
    slim_output_csv: str | Path,
    full_output_csv: str | Path,
) -> None:
    df = pd.read_excel(source_xlsx)

    renamed = df.rename(columns=RENAME_MAP)
    renamed["SeniorCitizen"] = renamed["SeniorCitizen"].map({"Yes": 1, "No": 0}).astype("int64")

    slim = renamed[ORIGINAL_SCHEMA_COLUMNS]
    slim.to_csv(Path(slim_output_csv), index=False)

    full = renamed.drop(columns=["Churn Value"])
    full.to_csv(Path(full_output_csv), index=False)


NEW_ENRICHMENT_COLUMNS = [
    "City",
    "State",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]


def build_enriched_dataset(
    telco_csv: str | Path,
    source_xlsx: str | Path,
    output_csv: str | Path,
) -> None:
    """Left-join telco.csv (authoritative for every existing column) with
    only the genuinely new columns from the extended source, joined on
    customerID. Any column that already exists in telco.csv is discarded
    from the extended source, even if its values disagree — telco.csv wins,
    unconditionally, for anything it already has.
    """
    base = pd.read_csv(Path(telco_csv))

    extended = pd.read_excel(source_xlsx).rename(columns={"CustomerID": "customerID"})
    new_columns_only = extended[["customerID", *NEW_ENRICHMENT_COLUMNS]]

    enriched = base.merge(new_columns_only, on="customerID", how="left")
    enriched.to_csv(Path(output_csv), index=False)


if __name__ == "__main__":
    prepare_extended(
        source_xlsx="data/raw/telco_extended.xlsx",
        slim_output_csv="data/raw/telco_extended.csv",
        full_output_csv="data/raw/telco_extended_full.csv",
    )
    build_enriched_dataset(
        telco_csv="data/raw/telco.csv",
        source_xlsx="data/raw/telco_extended.xlsx",
        output_csv="data/raw/telco_enriched.csv",
    )
