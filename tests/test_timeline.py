import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base, Prediction


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
        test_client.SessionLocal = TestingSessionLocal  # exposed for tests that need to seed rows directly
        yield test_client
    app.dependency_overrides.clear()


def _get_token(client, tenant_id="telco"):
    email = f"timeline-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


def _first_customer_id(client, token):
    response = client.get("/api/priority", params={"limit": 1}, headers={"Authorization": f"Bearer {token}"})
    return response.json()[0]["customerID"]


def test_signup_event_present_and_labeled_estimated(client):
    token = _get_token(client)
    customer_id = _first_customer_id(client, token)

    response = client.get(f"/api/customer/{customer_id}/timeline", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    events = response.json()["events"]

    signup_events = [e for e in events if e["event_type"] == "signup"]
    assert len(signup_events) == 1
    assert signup_events[0]["estimated_from_tenure"] is True
    assert signup_events[0]["data_source"] == "inferred_from_snapshot"

    contract_events = [e for e in events if e["event_type"] == "contract_snapshot"]
    assert len(contract_events) == 1
    assert contract_events[0]["data_source"] == "inferred_from_snapshot"


def test_real_prediction_and_recommendation_events_appear_ordered(client):
    token = _get_token(client)
    customer_id = _first_customer_id(client, token)
    headers = {"Authorization": f"Bearer {token}"}

    client.post(
        "/api/whatif",
        json={"customer_id": customer_id, "overrides": {"Contract": "Two year"}},
        headers=headers,
    )
    client.post(
        "/api/whatif",
        json={"customer_id": customer_id, "overrides": {"Contract": "Month-to-month"}},
        headers=headers,
    )
    client.get(f"/api/recommend/{customer_id}", params={"log_event": True}, headers=headers)

    response = client.get(f"/api/customer/{customer_id}/timeline", headers=headers)
    assert response.status_code == 200
    events = response.json()["events"]

    real_events = [e for e in events if e["data_source"] == "real_system_event"]
    prediction_events = [e for e in real_events if e["event_type"] == "prediction"]
    recommendation_events = [e for e in real_events if e["event_type"] == "recommendation"]

    assert len(prediction_events) == 2
    assert len(recommendation_events) == 1
    for event in prediction_events:
        assert "churn_probability" in event["details"]
        assert event["details"]["model_version"] == "v1-whatif"

    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps)


def test_timeline_cross_tenant_lookup_404s_for_banking_tenant(client):
    """Banking was fully trained (see config/config.yaml's banking entry) -
    customer_timeline genuinely passed its own real gate, so this is now a
    genuine attempted lookup against banking's OWN population, not a
    gated-off "unavailable" response - a real telco customer id correctly
    404s (never found in banking's differently-shaped CustomerId data),
    proving isolation the same way an unknown id anywhere else does."""
    telco_token = _get_token(client, tenant_id="telco")
    customer_id = _first_customer_id(client, telco_token)

    banking_token = _get_token(client, tenant_id="banking")
    response = client.get(
        f"/api/customer/{customer_id}/timeline", headers={"Authorization": f"Bearer {banking_token}"}
    )
    assert response.status_code == 404


def test_timeline_returns_real_data_for_a_real_banking_customer(client):
    """Same module, the positive case: a REAL banking customer id gets a
    real (if likely thin/inferred-only) timeline back, not "not trained"."""
    import pandas as pd

    df = pd.read_csv("data/raw/bank_churn.csv")
    real_banking_customer_id = str(df.iloc[0]["CustomerId"])

    banking_token = _get_token(client, tenant_id="banking")
    response = client.get(
        f"/api/customer/{real_banking_customer_id}/timeline", headers={"Authorization": f"Bearer {banking_token}"}
    )
    assert response.status_code == 200
    assert response.json().get("available") is not False
    assert "events" in response.json()


def test_timeline_requires_authentication(client):
    response = client.get("/api/customer/9999-TEST/timeline")
    assert response.status_code == 401


def test_timeline_404_for_unknown_customer(client):
    token = _get_token(client)
    response = client.get(
        "/api/customer/NOT-A-REAL-CUSTOMER/timeline", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404


def test_recommend_call_without_log_event_flag_does_not_create_timeline_event(client):
    # /api/recommend/{id} is also called ~20x per Dashboard page load (one
    # per PriorityTable row, purely for display) - without an explicit
    # log_event=true, that must NOT create real audit-trail rows, or every
    # customer shown in the Priority Table would falsely look "active."
    token = _get_token(client)
    customer_id = _first_customer_id(client, token)
    headers = {"Authorization": f"Bearer {token}"}

    plain_response = client.get(f"/api/recommend/{customer_id}", headers=headers)
    assert plain_response.status_code == 200

    response = client.get(f"/api/customer/{customer_id}/timeline", headers=headers)
    events = response.json()["events"]
    real_events = [e for e in events if e["data_source"] == "real_system_event"]
    assert real_events == []


def test_bulk_migration_baseline_prediction_excluded_from_timeline(client):
    # Every customer gets a one-time baseline Prediction row (model_version
    # == the bare model dir name, e.g. "v1") from database/migrate_csv_to_db.py
    # at setup time - that's a data-loading artifact, not a real per-customer
    # audit-trail event, so it must not make an untouched customer's timeline
    # look "active" when nothing has actually been run for them.
    token = _get_token(client)
    customer_id = _first_customer_id(client, token)

    session = client.SessionLocal()
    session.add(
        Prediction(
            tenant_id="telco",
            customer_id=customer_id,
            model_version="v1",
            churn_probability=0.42,
        )
    )
    session.commit()
    session.close()

    response = client.get(f"/api/customer/{customer_id}/timeline", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    events = response.json()["events"]
    real_events = [e for e in events if e["data_source"] == "real_system_event"]
    assert real_events == []


# --- Regression: extracted into src/models/timeline.py and generalized to
# accept tenant_config (same recurring hardcoded-Telco-column bug this
# session already found in business_impact.py/recommend.py/backtest.py/
# scenario.py) - PLUS a real, separate missing-db= bug found in
# api/main.py's customer_timeline() endpoint while doing this (see that
# function's own comment): every one of feature_enabled()/model_dir_for()/
# data_path_for() there was being called without db=db, which silently
# made this endpoint report "unavailable" for every self-registered tenant
# regardless of their real feature_flags_json. ---

from pathlib import Path as _Path

from src.models.timeline import CustomerNotFoundError, load_customer_snapshot, timeline_sanity_check

AURORA_MODEL_DIR = "models/aurora-streaming"
AURORA_DATA_PATH = "data/tenant_uploads/aurora-streaming.csv"


def test_aurora_timeline_snapshot_loads_with_genuinely_different_column_names():
    """id column (customer_id) shares NOT ONE literal name with Telco's
    (customerID). Used to crash outright; must now return real (if
    possibly empty) events. Aurora has no duration_column mapped (see
    tests/test_business_impact.py's module comment) and no resolved
    segment_feature_column (no survival model) - both inferred-snapshot
    events are legitimately skipped, not crashed on."""
    import json

    split_info = json.loads((_Path(AURORA_MODEL_DIR) / "split_indices.json").read_text(encoding="utf-8"))
    import pandas as pd

    df = pd.read_csv(AURORA_DATA_PATH)
    test_customer_id = df.loc[split_info["test_idx"][0], "customer_id"]

    events = load_customer_snapshot(test_customer_id, model_dir=AURORA_MODEL_DIR, data_path=AURORA_DATA_PATH)
    assert events == []  # no duration_column, no resolved segment column - honest empty snapshot, not a crash

    with pytest.raises(CustomerNotFoundError):
        load_customer_snapshot("not-a-real-id", model_dir=AURORA_MODEL_DIR, data_path=AURORA_DATA_PATH)


def test_aurora_timeline_sanity_check_passes():
    from src.data.split import load_tenant_config

    tenant_config = load_tenant_config(AURORA_MODEL_DIR)
    assert timeline_sanity_check(AURORA_MODEL_DIR, AURORA_DATA_PATH, tenant_config) is None


def test_timeline_snapshot_for_a_freshly_trained_tenant_with_non_telco_column_names(tmp_path):
    """Self-contained variant (trains its own tiny synthetic tenant, WITH
    both a duration column and a resolved segment_feature_column, in
    tmp_path) so this specific guard - including both inferred-event types
    actually firing, which Aurora's real data doesn't exercise - runs
    anywhere, CI included."""
    import numpy as np
    import pandas as pd

    from src.models.segment import run_segmentation
    from src.models.survival import train_survival_model
    from src.models.train import train_model

    ROOT = _Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config" / "config.yaml"

    rng = np.random.default_rng(19)
    n = 400
    tenure_like = rng.integers(0, 60, size=n)
    fee = np.round(rng.normal(20, 5, size=n).clip(5, 40), 2)
    plan = rng.choice(["Alpha", "Beta"], size=n, p=[0.5, 0.5])
    logit = -0.3 - 0.02 * tenure_like + 0.9 * (plan == "Alpha") + rng.normal(0, 0.6, size=n)
    left = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame(
        {
            "acct_id": [f"A{i:04d}" for i in range(n)],
            "months_active": tenure_like,
            "fee_amount": fee,
            "plan_name": plan,
            "left": left,
        }
    )
    data_path = tmp_path / "synthetic.csv"
    df.to_csv(data_path, index=False)

    tenant_config = {
        "id_column": "acct_id",
        "target_column": "left",
        "target_positive_value": "Yes",
        "revenue_column": "fee_amount",
        "duration_column": "months_active",
        "tuning_enabled": False,
    }
    model_dir = tmp_path / "model"
    train_model(data_path, CONFIG_PATH, output_dir=model_dir, tenant_config=tenant_config)
    run_segmentation(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    train_survival_model(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)

    assert timeline_sanity_check(model_dir, data_path, tenant_config) is None

    test_customer_id = df.loc[0, "acct_id"]
    # Force it into the visible split by just using row 0's real id - simpler
    # than reaching into split_indices.json for this synthetic check.
    import json as _json

    split_info = _json.loads((model_dir / "split_indices.json").read_text(encoding="utf-8"))
    test_customer_id = df.loc[split_info["test_idx"][0], "acct_id"]

    events = load_customer_snapshot(test_customer_id, model_dir=model_dir, data_path=data_path)
    event_types = {e["event_type"] for e in events}
    assert event_types == {"signup", "contract_snapshot"}
