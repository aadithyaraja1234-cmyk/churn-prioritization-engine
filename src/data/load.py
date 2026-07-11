from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TENANT_CONFIG: dict[str, Any] = {
    "target_column": "Churn",
    "target_positive_value": "Yes",
    "revenue_column": "MonthlyCharges",
    "id_column": "customerID",
}


def load_raw(path: str | Path, tenant_config: dict[str, Any] | None = None) -> pd.DataFrame:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {data_path}")

    df = pd.read_csv(data_path)

    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    revenue_column = tenant_config["revenue_column"]

    required_columns = {id_column, target_column, revenue_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    assert df[id_column].is_unique, f"Duplicate IDs in column '{id_column}'"

    return df
