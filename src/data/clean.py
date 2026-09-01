from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.load import DEFAULT_TENANT_CONFIG


def clean_data(df: pd.DataFrame, tenant_config: dict[str, Any] | None = None) -> pd.DataFrame:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    cleaned = df.copy()

    # Coerce id_column to string dtype, once, at the single shared entry
    # point virtually every module reads raw data through. Found live:
    # Banking's real CustomerId column is a genuine int64 (15634602, not
    # "15634602") - every per-customer lookup elsewhere in this project
    # (explain.py/survival.py/timeline.py/whatif.py/scenario.py: `df[id_
    # column] == customer_id`) compares against a customer_id that always
    # arrives as a plain string from the API (FastAPI path params, JSON
    # request bodies), which pandas never coerces for an int64 column - the
    # comparison silently returns an all-False mask and every real customer
    # 404s. This went unnoticed for as long as it did because Banking's
    # business_impact_core (and everything downstream of it) was disabled
    # until this session - the numeric-id code path was simply never
    # exercised for a real request before. Telco/every other real tenant
    # already has a string-shaped id (customerID like "7590-VHVEG"), so
    # this is a no-op for them - only a tenant with a genuinely numeric id
    # column changes behavior here, and only for the better.
    id_column = tenant_config["id_column"]
    cleaned[id_column] = cleaned[id_column].astype(str)

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
