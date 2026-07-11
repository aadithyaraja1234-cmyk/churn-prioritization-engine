"""Cox Proportional Hazards survival model — independent of the classifier pipeline.

TotalCharges is excluded as a covariate alongside tenure (the duration column):
TotalCharges is approximately tenure * MonthlyCharges, so it leaks duration
information back into the model. Including it inflated the test c-index from
0.868 to 0.931 - well above published benchmarks (0.83-0.89) for this dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lifelines import CoxPHFitter

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices
from src.features.encode import fit_categorical_encoders, transform_categorical_features


def train_survival_model(model_dir: str | Path = "models/v1", data_path: str | Path = "data/raw/telco.csv") -> dict[str, Any]:
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    df = clean_data(load_raw(data_path))
    train_idx, test_idx = load_split_indices(model_dir)

    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    # tenure is the survival duration, so it must not also appear as a covariate.
    # TotalCharges is excluded too: it is approximately tenure * MonthlyCharges,
    # so it leaks duration information back in (see module docstring).
    feature_columns = [
        col for col in df.columns if col not in ("customerID", "Churn", "tenure", "TotalCharges")
    ]
    categorical_columns = [col for col in feature_columns if df_train[col].dtype == "object"]
    encoders = fit_categorical_encoders(df_train[feature_columns], categorical_columns)

    X_train = transform_categorical_features(df_train[feature_columns], encoders)
    X_test = transform_categorical_features(df_test[feature_columns], encoders)

    cox_train = X_train.copy()
    cox_train["tenure"] = df_train["tenure"].values
    cox_train["event"] = df_train["Churn"].eq("Yes").astype(int).values

    cox_test = X_test.copy()
    cox_test["tenure"] = df_test["tenure"].values
    cox_test["event"] = df_test["Churn"].eq("Yes").astype(int).values

    cph = CoxPHFitter()
    cph.fit(cox_train, duration_col="tenure", event_col="event")

    c_index = cph.score(cox_test, scoring_method="concordance_index")

    joblib.dump(
        {"model": cph, "encoders": encoders, "feature_columns": feature_columns},
        model_dir / "survival_model.pkl",
    )

    hazard_ratios = cph.hazard_ratios_.reindex(
        cph.hazard_ratios_.sub(1.0).abs().sort_values(ascending=False).index
    )
    coefficients = {feature: float(hr) for feature, hr in hazard_ratios.items()}

    return {
        "c_index": float(c_index),
        "coefficients": coefficients,
    }


def median_survival_by_contract(
    model_dir: str | Path = "models/v1", data_path: str | Path = "data/raw/telco.csv"
) -> dict[str, float]:
    """Median predicted survival time (tenure, months) per contract type.

    Holds every other feature at its training-set median and varies only
    Contract, using the already-fitted model - no refitting.
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    saved = joblib.load(model_dir / "survival_model.pkl")
    cph = saved["model"]
    encoders = saved["encoders"]
    feature_columns = saved["feature_columns"]

    df = clean_data(load_raw(data_path))
    train_idx, _ = load_split_indices(model_dir)
    X_train = transform_categorical_features(df.loc[train_idx, feature_columns], encoders)
    baseline_row = X_train.median()

    segments = {}
    for contract_name, code in encoders["Contract"].items():
        row = baseline_row.copy()
        row["Contract"] = code
        segments[contract_name] = row

    segment_df = pd.DataFrame(segments.values(), index=list(segments.keys()))
    medians = cph.predict_median(segment_df)
    # predict_median() returns inf when the survival curve never drops below
    # 0.5 within the observed data range (e.g. Two year contracts churn so
    # rarely that the median lies beyond what was observed) - inf isn't valid
    # JSON, and it also isn't a real time estimate, so it's reported as None.
    return {name: (None if medians[name] == float("inf") else float(medians[name])) for name in segments}
