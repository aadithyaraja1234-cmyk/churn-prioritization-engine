from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.load import DEFAULT_TENANT_CONFIG


def clean_data(df: pd.DataFrame, tenant_config: dict[str, Any] | None = None) -> pd.DataFrame:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    cleaned = df.copy()

    revenue_column = tenant_config["revenue_column"]
    cleaned[revenue_column] = pd.to_numeric(cleaned[revenue_column], errors="coerce")

    # Telco-specific quirk: some zero-tenure customers have a blank TotalCharges
    # even though MonthlyCharges was billed. Only applies when those exact
    # columns exist (i.e. the Telco tenant) so it never fires for other tenants.
    if "TotalCharges" in cleaned.columns and "tenure" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        zero_tenure_mask = (cleaned["tenure"] == 0) & cleaned["TotalCharges"].isna()
        cleaned.loc[zero_tenure_mask, "TotalCharges"] = cleaned.loc[zero_tenure_mask, revenue_column]
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0.0)
    else:
        cleaned[revenue_column] = cleaned[revenue_column].fillna(0.0)

    return cleaned
