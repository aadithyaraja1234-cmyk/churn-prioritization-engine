import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.db import get_tenant_scoped_query
from database.models import Base, Customer, Prediction


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _seed_customers(session):
    session.add_all(
        [
            Customer(tenant_id="telco", customer_id="T-001", raw_features={"a": 1}),
            Customer(tenant_id="telco", customer_id="T-002", raw_features={"a": 2}),
            Customer(tenant_id="banking", customer_id="B-001", raw_features={"b": 1}),
            Customer(tenant_id="banking", customer_id="B-002", raw_features={"b": 2}),
            Customer(tenant_id="banking", customer_id="B-003", raw_features={"b": 3}),
        ]
    )
    session.commit()


def test_telco_scoped_query_excludes_all_banking_rows(db_session):
    _seed_customers(db_session)

    telco_rows = get_tenant_scoped_query(db_session, Customer, tenant_id="telco").all()

    assert len(telco_rows) == 2
    assert all(row.tenant_id == "telco" for row in telco_rows)
    assert sum(1 for row in telco_rows if row.tenant_id == "banking") == 0


def test_banking_scoped_query_excludes_all_telco_rows(db_session):
    _seed_customers(db_session)

    banking_rows = get_tenant_scoped_query(db_session, Customer, tenant_id="banking").all()

    assert len(banking_rows) == 3
    assert all(row.tenant_id == "banking" for row in banking_rows)
    assert sum(1 for row in banking_rows if row.tenant_id == "telco") == 0


def test_isolation_holds_for_predictions_table_too(db_session):
    db_session.add_all(
        [
            Prediction(tenant_id="telco", customer_id="T-001", model_version="v1", churn_probability=0.4),
            Prediction(tenant_id="banking", customer_id="B-001", model_version="banking_v1", churn_probability=0.7),
        ]
    )
    db_session.commit()

    telco_predictions = get_tenant_scoped_query(db_session, Prediction, tenant_id="telco").all()
    banking_predictions = get_tenant_scoped_query(db_session, Prediction, tenant_id="banking").all()

    assert len(telco_predictions) == 1
    assert telco_predictions[0].tenant_id == "telco"
    assert len(banking_predictions) == 1
    assert banking_predictions[0].tenant_id == "banking"
