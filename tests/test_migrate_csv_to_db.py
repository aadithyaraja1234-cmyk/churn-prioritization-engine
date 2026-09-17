"""Tests for database/migrate_csv_to_db.py's idempotency - calling
migrate_customers()/migrate_predictions() twice for the same tenant must
insert rows only once, not duplicate them. Found live while preparing for
a Postgres/Supabase deployment: the previous only-guard was
docker-entrypoint.sh checking whether a local SQLite file existed yet,
which stops protecting anything the moment DATABASE_URL is set (a fresh
container has no local file regardless of whether the real DB already has
data) - every container restart would have silently duplicated Telco/
Banking's rows in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.migrate_csv_to_db import migrate_customers, migrate_predictions
from database.models import Base, Customer, Prediction

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()


def test_migrate_customers_is_idempotent(session):
    csv_path = ROOT / "data" / "raw" / "telco.csv"

    first_count = migrate_customers(session, csv_path, "telco", "customerID")
    assert first_count > 0

    second_count = migrate_customers(session, csv_path, "telco", "customerID")
    assert second_count == 0

    total_rows = session.query(Customer).filter(Customer.tenant_id == "telco").count()
    assert total_rows == first_count


def test_migrate_customers_does_not_block_a_different_tenant(session):
    telco_count = migrate_customers(session, ROOT / "data" / "raw" / "telco.csv", "telco", "customerID")
    assert telco_count > 0

    banking_count = migrate_customers(session, ROOT / "data" / "raw" / "bank_churn.csv", "banking", "CustomerId")
    assert banking_count > 0


def test_migrate_predictions_is_idempotent(session):
    from src.config import load_config

    config = load_config(ROOT / "config" / "config.yaml")
    telco_config = config["tenants"]["telco"]
    model_dir = ROOT / "models" / "v1"
    data_path = ROOT / "data" / "raw" / "telco.csv"

    first_count = migrate_predictions(session, model_dir, "telco", data_path, telco_config)
    assert first_count > 0

    second_count = migrate_predictions(session, model_dir, "telco", data_path, telco_config)
    assert second_count == 0

    total_rows = session.query(Prediction).filter(Prediction.tenant_id == "telco").count()
    assert total_rows == first_count
