"""Customer Lifetime Value regression on the real CLTV column — independent of the classifier pipeline.

Uses the real, per-customer CLTV column from telco_enriched.csv (not the
TotalCharges proxy used elsewhere). CLTV's correlation with MonthlyCharges
(0.099), tenure (0.396), and TotalCharges (0.342) is well below the 0.9
leakage threshold used for survival.py's TotalCharges exclusion, so no
feature is excluded on those grounds. Features are restricted to the same
19-column set the classifier uses; the other enrichment columns (City,
State, Zip Code, Latitude, Longitude, Churn Score, Churn Reason) are
excluded because they are either high-cardinality geo identifiers or
derived from/biased toward the Churn label, not meant to double as CLV
predictors here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_split_indices
from src.features.encode import fit_categorical_encoders, transform_categorical_features

NON_FEATURE_COLUMNS = {
    "customerID",
    "Churn",
    "CLTV",
    "City",
    "State",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Churn Score",
    "Churn Reason",
}


def train_clv_model(
    model_dir: str | Path = "models/v1_enriched",
    data_path: str | Path = "data/raw/telco_enriched.csv",
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    df = clean_data(load_raw(data_path))
    train_idx, test_idx = load_split_indices(model_dir)

    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS]
    categorical_columns = [col for col in feature_columns if df_train[col].dtype == "object"]
    encoders = fit_categorical_encoders(df_train[feature_columns], categorical_columns)

    X_train = transform_categorical_features(df_train[feature_columns], encoders)
    X_test = transform_categorical_features(df_test[feature_columns], encoders)

    y_train = df_train["CLTV"]
    y_test = df_test["CLTV"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(
        {"model": model, "encoders": encoders, "feature_columns": feature_columns},
        model_dir / "clv_model.pkl",
    )

    feature_importances = sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "feature_importances": feature_importances,
    }


def load_feature_importances(model_dir: str | Path = "models/v1_enriched") -> list[tuple[str, float]]:
    """Feature importances from the already-fitted CLV model (no refit)."""
    model_dir = Path(model_dir)
    saved = joblib.load(model_dir / "clv_model.pkl")
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    return sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
