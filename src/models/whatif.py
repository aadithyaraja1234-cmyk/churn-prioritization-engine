"""What-if simulator — scores feature overrides against the already-trained
classifier. No retraining: this only re-runs model.pkl/encoders.pkl on a
modified copy of a real customer's feature vector.

Generalized to accept tenant_config (same pattern as prioritize.py/
business_impact.py/explain.py/recommend.py/scenario.py/backtest.py/
timeline.py - read back from model_dir/split_indices.json via
src/data/split.py's load_tenant_config() when omitted). Every hardcoded
Telco literal ("customerID"/"Churn", and the positive-class index 1
predict_proba() lookup) has been replaced by a tenant_config-driven lookup -
this was the last of this project's original modules still hardcoded to
Telco's literal column names, the same recurring hardcoded-column bug this
project has hit and fixed at least three times before elsewhere: Meridian's
action-queue/alerts, Aurora's business-impact, recommend.py again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_tenant_config
from src.features.encode import transform_categorical_features


class CustomerNotFoundError(ValueError):
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        super().__init__(f"Customer '{customer_id}' not found")


class InvalidFeatureError(ValueError):
    def __init__(self, feature: str, valid_features: list[str]):
        self.feature = feature
        self.valid_features = valid_features
        super().__init__(f"'{feature}' is not a valid feature column")


class InvalidCategoryError(ValueError):
    def __init__(self, feature: str, value: Any, valid_values: list[str]):
        self.feature = feature
        self.value = value
        self.valid_values = valid_values
        super().__init__(f"'{value}' is not a valid value for '{feature}'")


def simulate_whatif(
    customer_id: str,
    overrides: dict[str, Any],
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    target_positive_value = tenant_config["target_positive_value"]

    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    feature_columns = [col for col in df.columns if col not in (id_column, target_column)]

    customer_row = df.loc[df[id_column] == customer_id]
    if customer_row.empty:
        raise CustomerNotFoundError(customer_id)

    # Validate every override before applying any of them.
    for feature, value in overrides.items():
        if feature not in feature_columns:
            raise InvalidFeatureError(feature, feature_columns)
        if feature in encoders and value not in encoders[feature]:
            raise InvalidCategoryError(feature, value, sorted(encoders[feature].keys()))

    baseline = customer_row.iloc[0][feature_columns].copy()
    modified = baseline.copy()
    for feature, value in overrides.items():
        modified[feature] = value

    baseline_encoded = transform_categorical_features(baseline.to_frame().T, encoders).reindex(
        columns=feature_columns, fill_value=0
    )
    modified_encoded = transform_categorical_features(modified.to_frame().T, encoders).reindex(
        columns=feature_columns, fill_value=0
    )

    # Looks up target_positive_value's actual position in the fitted
    # model's own classes_ rather than assuming index 1 - same fix, same
    # reasoning, as src/models/prioritize.py's get_priority_ranking() (see
    # that module's docstring for the full explanation): sklearn sorts
    # classes_ alphabetically/numerically, and nothing guarantees the
    # positive label lands at index 1 for every possible tenant's label
    # values (it does for Telco's "Yes"/"No" and Banking's 0/1 today, but
    # that's a coincidence of those specific label spellings, not a rule).
    positive_index = list(model.classes_).index(target_positive_value)
    original_probability = float(model.predict_proba(baseline_encoded)[0, positive_index])
    new_probability = float(model.predict_proba(modified_encoded)[0, positive_index])

    return {
        "customer_id": customer_id,
        "original_probability": original_probability,
        "new_probability": new_probability,
        "delta": new_probability - original_probability,
        "overrides_applied": overrides,
    }
