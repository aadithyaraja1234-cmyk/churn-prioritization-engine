from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import split_data
from src.features.encode import transform_categorical_features


def evaluate_model(data_path: str | Path, config_path: str | Path, model_dir: str | Path | None = None) -> dict[str, Any]:
    data_path = Path(data_path)
    config_path = Path(config_path)
    model_dir = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parents[2] / "models" / "v1"

    config = load_config(config_path)
    df = clean_data(load_raw(data_path))
    X_train, X_test, y_train, y_test, _, _ = split_data(df, config)

    encoders = joblib.load(model_dir / "encoders.pkl")
    model = joblib.load(model_dir / "model.pkl")

    X_test_encoded = transform_categorical_features(X_test, encoders)
    probabilities = model.predict_proba(X_test_encoded)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    report = classification_report(
        y_test.map({"Yes": 1, "No": 0}),
        predictions,
        output_dict=True,
        zero_division=0,
    )
    return {"classification_report": report, "predictions": predictions.tolist()}
