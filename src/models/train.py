from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import DEFAULT_TENANT_CONFIG, load_raw
from src.data.split import split_data
from src.features.encode import fit_categorical_encoders, transform_categorical_features


def train_model(
    data_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    data_path = Path(data_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "models" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    X_train, X_test, y_train, y_test, _, _ = split_data(df, config, tenant_config)

    categorical_columns = [col for col in X_train.columns if X_train[col].dtype == "object"]
    encoders = fit_categorical_encoders(X_train, categorical_columns)
    X_train_encoded = transform_categorical_features(X_train, encoders)
    X_test_encoded = transform_categorical_features(X_test, encoders)

    model_config = config.get("model", {})
    model = GradientBoostingClassifier(
        n_estimators=int(model_config.get("n_estimators", 200)),
        max_depth=int(model_config.get("max_depth", 3)),
        learning_rate=float(model_config.get("learning_rate", 0.05)),
        random_state=int(model_config.get("random_state", 42)),
    )
    model.fit(X_train_encoded, y_train)

    positive_value = tenant_config["target_positive_value"]
    positive_class_index = list(model.classes_).index(positive_value)
    probabilities = model.predict_proba(X_test_encoded)[:, positive_class_index]

    y_train_binary = (y_train == positive_value).astype(int)
    y_test_binary = (y_test == positive_value).astype(int)
    roc_auc = roc_auc_score(y_test_binary, probabilities)
    pr_auc = average_precision_score(y_test_binary, probabilities)

    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(encoders, output_dir / "encoders.pkl")

    metadata = {
        "run_id": "v1",
        "timestamp": datetime.now().isoformat(),
        "config_used": config,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate_train": float(y_train_binary.mean()),
        "churn_rate_test": float(y_test_binary.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
