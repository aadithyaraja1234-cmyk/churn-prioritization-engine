from __future__ import annotations

import pandas as pd


def fit_categorical_encoders(train_df: pd.DataFrame, categorical_columns: list[str]) -> dict[str, dict[object, int]]:
    encoders: dict[str, dict[object, int]] = {}
    for column in categorical_columns:
        if column in train_df.columns:
            unique_values = sorted(train_df[column].dropna().unique().tolist())
            encoders[column] = {value: index for index, value in enumerate(unique_values)}
    return encoders


def transform_categorical_features(df: pd.DataFrame, encoders: dict[str, dict[object, int]]) -> pd.DataFrame:
    transformed = df.copy()
    for column, encoder in encoders.items():
        if column in transformed.columns:
            transformed[column] = transformed[column].map(encoder).fillna(-1)
    return transformed
