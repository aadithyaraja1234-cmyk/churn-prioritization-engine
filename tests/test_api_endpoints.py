import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base, Customer

BANKING_CLV_UNAVAILABLE = {
    "available": False,
    "reason": (
        "no CLV-equivalent column exists in this tenant's raw data (bank_churn.csv has no "
        "lifetime-value-shaped field) - genuinely unavailable, not just unmapped. See clv_estimated "
        "for a formula-based proxy instead."
    ),
}


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        # Exposed so a test can seed rows (e.g. Customer) directly into this
        # same in-memory DB, without a public endpoint for doing so - see
        # test_customers_recommend_eligible_only_filters_to_test_split_customers.
        test_client.db_sessionmaker = TestingSessionLocal
        yield test_client
    app.dependency_overrides.clear()


def _get_token(client, email, password, tenant_id, role="analyst"):
    client.post("/auth/register", json={"email": email, "password": password, "tenant_id": tenant_id, "role": role})
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture()
def telco_token(client):
    return _get_token(client, "telco-user@example.com", "s3cret-pw", "telco")


@pytest.fixture()
def banking_token(client):
    return _get_token(client, "banking-user@example.com", "s3cret-pw", "banking")


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


TELCO_ENDPOINTS = [
    ("/api/model/metrics", {}),
    ("/api/model/importance", {}),
    ("/api/priority", {"strategy": "revenue_weighted", "limit": 10}),
    ("/api/backtest", {"top_pct": 0.2}),
    ("/api/backtest/curve", {"min_pct": 10, "max_pct": 30, "step_pct": 10}),
    ("/api/survival/segments", {}),
    ("/api/segments", {}),
    ("/api/anomalies", {"limit": 5}),
    ("/api/clv/importance", {}),
    ("/api/clv/estimate", {}),
    ("/customers", {}),
    # Dashboard reordering (moving Priority Table / Backtest / Survival /
    # Segments / Anomalies / CLV / Cohort Comparison off the main dashboard
    # onto /analytics or /admin) is a frontend-only component move - these
    # underlying endpoints must keep returning 200 regardless of which page
    # calls them.
    ("/api/business-impact", {}),
    ("/api/action-queue", {"limit": 5}),
    ("/api/alerts", {}),
    ("/api/cohort-comparison", {}),
]


@pytest.mark.parametrize("path,params", TELCO_ENDPOINTS)
def test_telco_endpoints_return_200_when_authenticated(client, telco_token, path, params):
    response = client.get(path, params=params, headers=_auth_headers(telco_token))
    assert response.status_code == 200, response.text


def test_health_endpoint_returns_200_unauthenticated():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_banking_model_metrics_and_importance_are_real_and_return_200(client, banking_token):
    metrics_response = client.get("/api/model/metrics", headers=_auth_headers(banking_token))
    assert metrics_response.status_code == 200
    assert "roc_auc" in metrics_response.json()

    importance_response = client.get("/api/model/importance", headers=_auth_headers(banking_token))
    assert importance_response.status_code == 200
    assert len(importance_response.json()) > 0


def test_banking_priority_and_backtest_return_real_data(client, banking_token):
    """Banking was fully trained (see config/config.yaml's banking entry) -
    priority_ranking/backtest now genuinely pass their own real sanity
    gates (Spearman 0.646, 407 real churned test customers) using Balance
    as the revenue proxy, real numbers not a guess. This replaces the
    tenant's earlier "has not been backtested" caveat, which turned out to
    be an untested assumption, not a real limitation - a real backtest run
    clears the bar cleanly."""
    priority_response = client.get("/api/priority", headers=_auth_headers(banking_token))
    assert priority_response.status_code == 200
    priority_rows = priority_response.json()
    assert len(priority_rows) > 0
    assert "available" not in priority_rows[0]

    backtest_response = client.get("/api/backtest", headers=_auth_headers(banking_token))
    assert backtest_response.status_code == 200
    assert backtest_response.json().get("available") is not False
    assert "revenue_weighted" in backtest_response.json()

    curve_response = client.get("/api/backtest/curve", headers=_auth_headers(banking_token))
    assert curve_response.status_code == 200
    assert len(curve_response.json()) > 0
    assert curve_response.json()[0].get("available") is not False


def test_backtest_curve_matches_backtest_at_the_same_percentage(client, telco_token):
    """The curve endpoint and the single-point endpoint must agree exactly -
    same underlying computation, just batched."""
    single_response = client.get(
        "/api/backtest", params={"top_pct": 0.2}, headers=_auth_headers(telco_token)
    )
    curve_response = client.get(
        "/api/backtest/curve",
        params={"min_pct": 20, "max_pct": 20, "step_pct": 1},
        headers=_auth_headers(telco_token),
    )
    assert single_response.status_code == 200
    assert curve_response.status_code == 200
    assert curve_response.json() == [single_response.json()]


def test_backtest_curve_rejects_invalid_range(client, telco_token):
    response = client.get(
        "/api/backtest/curve",
        params={"min_pct": 50, "max_pct": 10, "step_pct": 1},
        headers=_auth_headers(telco_token),
    )
    assert response.status_code == 400


def test_banking_survival_segments_anomalies_return_real_data(client, banking_token):
    """Banking was fully trained - survival (c-index 0.716, using the real
    Tenure column)/segments (silhouette 0.089, k=2)/anomalies (5.0%
    flagged) all genuinely passed their own gates. Real data, not a
    placeholder or a simulated 'trained' flag."""
    survival_response = client.get("/api/survival/segments", headers=_auth_headers(banking_token))
    assert survival_response.status_code == 200
    assert survival_response.json().get("available") is not False

    # /api/segments and /api/anomalies return a bare LIST when enabled
    # (unavailable_response()'s {"available": False, ...} is a dict) - a
    # plain isinstance check distinguishes the two shapes without assuming
    # either one has a .get() method.
    segments_response = client.get("/api/segments", headers=_auth_headers(banking_token))
    assert segments_response.status_code == 200
    assert isinstance(segments_response.json(), list) and len(segments_response.json()) > 0

    anomalies_response = client.get("/api/anomalies", headers=_auth_headers(banking_token))
    assert anomalies_response.status_code == 200
    assert isinstance(anomalies_response.json(), list)


def test_banking_clv_stays_honestly_unavailable(client, banking_token):
    """The one genuine, structural gap: bank_churn.csv has no CLV-
    equivalent column at all - correctly reported unavailable with a
    specific reason, not fabricated or silently defaulted to 'not yet
    trained'."""
    response = client.get("/api/clv/importance", headers=_auth_headers(banking_token))
    assert response.status_code == 200, response.text
    assert response.json() == BANKING_CLV_UNAVAILABLE


def test_banking_clv_estimate_returns_real_derived_data(client, banking_token):
    """The formula-based proxy that stands in for the real gap above:
    Banking has no CLV column, but does have a real duration_column and a
    passing survival model, so /api/clv/estimate returns a genuine,
    clearly-labeled derived estimate instead of unavailable."""
    response = client.get("/api/clv/estimate", headers=_auth_headers(banking_token))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("available") is not False
    assert body["estimated"] is True
    assert "not a trained regression" in body["methodology_note"]
    assert body["count"] > 0
    assert body["mean"] >= 0
    assert len(body["top_customers"]) > 0


def test_telco_clv_estimate_stays_unavailable_since_real_clv_already_exists(client, telco_token):
    """Mutually exclusive by design (see src/models/tenant_training.py's
    _run_optional_modules()): a tenant with a real, mapped CLV column
    (Telco: CLTV) never gets the formula-based proxy too - /api/clv/importance
    is the real thing for Telco, /api/clv/estimate stays unavailable."""
    response = client.get("/api/clv/estimate", headers=_auth_headers(telco_token))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is False
    assert "a real, trained CLV model already exists" in body["reason"]


# Real Telco customerIDs, one from each side of models/v1's saved split -
# see tests/test_explain.py's own test for the same technique (df.loc[idx,
# "customerID"]). Hardcoded rather than re-derived per test run so this
# test doesn't silently drift if the split ever changes underneath it -
# a change here should be a deliberate, visible diff.
TELCO_TEST_SPLIT_CUSTOMER_ID = "8091-TTVAX"
TELCO_TRAIN_SPLIT_CUSTOMER_ID = "7590-VHVEG"


def test_customers_recommend_eligible_only_filters_to_test_split_customers(client, telco_token):
    """Regression test for the Customer 360 dropdown bug: /customers used to
    list every ingested customer (train and test alike), but
    recommend_action_for_customer() only ever scores test-split customers -
    so ~80% of the unfiltered dropdown's own options 404'd with "not in the
    saved test split". recommend_eligible_only=true must filter the train-
    split customer out and keep the test-split one, and - the real proof,
    not just a filtered count - every customer it DOES return must actually
    succeed a real /api/recommend/{customer_id} call."""
    db = client.db_sessionmaker()
    try:
        db.add_all(
            [
                Customer(tenant_id="telco", customer_id=TELCO_TEST_SPLIT_CUSTOMER_ID, raw_features={}),
                Customer(tenant_id="telco", customer_id=TELCO_TRAIN_SPLIT_CUSTOMER_ID, raw_features={}),
            ]
        )
        db.commit()
    finally:
        db.close()

    unfiltered = client.get("/customers", headers=_auth_headers(telco_token))
    assert unfiltered.status_code == 200
    unfiltered_ids = {row["customer_id"] for row in unfiltered.json()}
    assert {TELCO_TEST_SPLIT_CUSTOMER_ID, TELCO_TRAIN_SPLIT_CUSTOMER_ID} <= unfiltered_ids

    filtered = client.get(
        "/customers", params={"recommend_eligible_only": True}, headers=_auth_headers(telco_token)
    )
    assert filtered.status_code == 200
    filtered_ids = {row["customer_id"] for row in filtered.json()}
    assert TELCO_TEST_SPLIT_CUSTOMER_ID in filtered_ids
    assert TELCO_TRAIN_SPLIT_CUSTOMER_ID not in filtered_ids

    # The real proof: every customer_id the filtered list returns must
    # actually get a real recommendation, not the "not in the saved test
    # split" 404 this filter exists to prevent.
    for customer_id in filtered_ids:
        response = client.get(f"/api/recommend/{customer_id}", headers=_auth_headers(telco_token))
        assert response.status_code == 200, (customer_id, response.text)
        assert "recommended_action" in response.json()

    # And the customer this filter excluded must still 404 the old way if
    # called directly - confirms the filter is removing a REAL trap, not a
    # false positive.
    train_response = client.get(
        f"/api/recommend/{TELCO_TRAIN_SPLIT_CUSTOMER_ID}", headers=_auth_headers(telco_token)
    )
    assert train_response.status_code == 404
    assert "not in the saved test split" in train_response.json()["detail"]
