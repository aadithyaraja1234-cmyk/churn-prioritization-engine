"""Migrate CSV source data into the SQLite persistence layer.

Loads data/raw/telco.csv (tenant_id="telco") and data/raw/bank_churn.csv
(tenant_id="banking") into the customers table, storing each row's full
feature set as JSON.

models/v1/metadata.json and models/banking_v1/metadata.json only hold
aggregate metrics (ROC-AUC, PR-AUC, n_train, etc.) - there are no
per-customer predictions to "load" from them directly. Their presence is
instead used as the signal that a trained model.pkl/encoders.pkl is
available for that tenant; when so, that saved model is used to score
every migrated customer and populate real per-customer prediction rows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from database.db import SessionLocal, init_db
from database.models import Customer, Prediction
from src.config import load_config
from src.data.clean import clean_data
from src.data.load import load_raw
from src.features.encode import transform_categorical_features

ROOT = Path(__file__).resolve().parent.parent


def migrate_customers(session: Session, csv_path: Path, tenant_id: str, id_column: str) -> int:
    df = pd.read_csv(csv_path)
    now = datetime.now(timezone.utc)
    count = 0
    for _, row in df.iterrows():
        session.add(
            Customer(
                tenant_id=tenant_id,
                customer_id=str(row[id_column]),
                raw_features=json.loads(row.to_json()),
                created_at=now,
            )
        )
        count += 1
    session.commit()
    return count


def migrate_predictions(
    session: Session,
    model_dir: Path,
    tenant_id: str,
    data_path: Path,
    tenant_config: dict[str, Any],
) -> int:
    # Predictions stored here cover the FULL customer population (train+test)
    # for product/demo purposes - this is not the evaluation set. All reported
    # accuracy metrics (ROC-AUC, backtest lift, etc.) come exclusively from the
    # held-out test split via models/v1/metadata.json and the backtest module,
    # computed independently of this table.
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return 0

    model = joblib.load(model_dir / "model.pkl")
    encoders = joblib.load(model_dir / "encoders.pkl")

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    feature_columns = [col for col in df.columns if col not in (id_column, target_column)]

    X = transform_categorical_features(df[feature_columns], encoders)
    positive_value = tenant_config["target_positive_value"]
    positive_class_index = list(model.classes_).index(positive_value)
    probabilities = model.predict_proba(X)[:, positive_class_index]

    now = datetime.now(timezone.utc)
    model_version = model_dir.name
    count = 0
    for customer_id, probability in zip(df[id_column], probabilities):
        session.add(
            Prediction(
                tenant_id=tenant_id,
                customer_id=str(customer_id),
                model_version=model_version,
                churn_probability=float(probability),
                predicted_at=now,
            )
        )
        count += 1
    session.commit()
    return count


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        config = load_config(ROOT / "config" / "config.yaml")
        telco_config = config["tenants"]["telco"]
        banking_config = config["tenants"]["banking"]

        telco_customers = migrate_customers(
            session, ROOT / "data" / "raw" / "telco.csv", "telco", telco_config["id_column"]
        )
        banking_customers = migrate_customers(
            session, ROOT / "data" / "raw" / "bank_churn.csv", "banking", banking_config["id_column"]
        )

        telco_predictions = migrate_predictions(
            session, ROOT / "models" / "v1", "telco", ROOT / "data" / "raw" / "telco.csv", telco_config
        )
        banking_predictions = migrate_predictions(
            session, ROOT / "models" / "banking_v1", "banking", ROOT / "data" / "raw" / "bank_churn.csv", banking_config
        )

        print(f"telco customers migrated:      {telco_customers}")
        print(f"banking customers migrated:    {banking_customers}")
        print(f"telco predictions migrated:    {telco_predictions}")
        print(f"banking predictions migrated:  {banking_predictions}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
