from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TENANT_CONFIG: dict[str, Any] = {
    "target_column": "Churn",
    "target_positive_value": "Yes",
    "revenue_column": "MonthlyCharges",
    "id_column": "customerID",
    # --- Optional-module keys below: only survival.py/clv.py (and
    # survival.py's per-category reporting helpers) read these, and only
    # when the corresponding module is actually run for a tenant. They are
    # absent from load_raw()'s required_columns on purpose - a tenant with
    # no duration/CLV-equivalent column can still train the core classifier
    # and run segment.py/anomaly.py, which never touch these keys. ---
    # duration_column: survival.py's Cox PH duration ("tenure"-equivalent).
    "duration_column": "tenure",
    # duration_leakage_column: a derived "duration * revenue" proxy column
    # (Telco's TotalCharges ~= tenure * MonthlyCharges) that leaks duration
    # back into survival.py's covariates if left in - see survival.py's
    # module docstring for the 0.868 -> 0.931 c-index inflation this caused
    # before it was excluded. None/absent means "no such column for this
    # tenant" (nothing is excluded on this specific ground).
    "duration_leakage_column": "TotalCharges",
    # clv_column: the real, per-customer lifetime-value column clv.py
    # regresses on (Telco's enriched CLTV column). Only present in Telco's
    # *enriched* dataset, not the base one used everywhere else - clv.py
    # itself errors clearly if this column isn't in the loaded data.
    "clv_column": "CLTV",
    # segment_feature_column: the categorical "contract type"-equivalent
    # feature survival.py's median_survival_by_contract()/
    # churn_likelihood_within_window() vary while holding every other
    # feature at its training-set median. Telco's is literally "Contract";
    # other tenants' analog (e.g. a subscription-tier column) is whatever
    # this key names.
    "segment_feature_column": "Contract",
    # clv_excluded_columns: extra columns clv.py must not use as regression
    # features on top of id/target/clv (already always excluded) - Telco's
    # enriched dataset's geo identifiers and churn-derived columns (see
    # clv.py's module docstring for why each one is excluded). Empty by
    # default for any tenant_config that doesn't set it explicitly.
    "clv_excluded_columns": ["City", "State", "Zip Code", "Latitude", "Longitude", "Churn Score", "Churn Reason"],
    # tuning_enabled: False specifically for Telco (this default IS what
    # run_pipeline.py's real reference-model retrain uses - tenant_config
    # defaults to this whole dict when omitted). A real, controlled
    # comparison (src/models/train.py's RandomizedSearchCV -
    # tests/test_hyperparameter_tuning.py) found tuning made Telco's
    # held-out ROC-AUC slightly WORSE (0.8471 -> 0.8400) despite a higher
    # cross-validation estimate, and every number this project has ever
    # published for Telco was computed under these original, fixed
    # hyperparameters - so they stay fixed. Self-registered tenants never
    # inherit this default (they always build an explicit tenant_config -
    # see src/models/tenant_training.py - which omits this key, defaulting
    # True in src/models/train.py).
    "tuning_enabled": False,
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
