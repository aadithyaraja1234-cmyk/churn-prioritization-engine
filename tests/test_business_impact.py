from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base
from src.models.business_impact import (
    FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT,
    INTERVENTION_SUCCESS_RATE,
    MIN_REMAINING_TENURE_MONTHS,
    business_impact_metadata,
    compute_business_impact_bulk,
    confidence,
    ease_of_saving,
    expected_remaining_tenure_months,
    opportunity_score,
    recoverable_revenue,
    to_business_language,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Meridian Wireless: a real self-registered tenant with only the core
# classifier trained (no CLV data at all - clv_data_path=None, the same
# value src.tenant_registry's Company-backed profile resolution actually
# passes for a self-registered tenant). Real artifacts, not mocks.
MERIDIAN_MODEL_DIR = "models/meridian-wireless"
MERIDIAN_DATA_PATH = "data/tenant_uploads/meridian-wireless.csv"


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
        yield test_client
    app.dependency_overrides.clear()


def _get_token(client, tenant_id="telco"):
    email = f"business-impact-{tenant_id}@example.com"
    client.post(
        "/auth/register",
        json={"email": email, "password": "pw-123456", "tenant_id": tenant_id, "role": "analyst"},
    )
    response = client.post("/auth/login", json={"email": email, "password": "pw-123456"})
    return response.json()["access_token"]


# --- Pure function range checks (synthetic inputs, no model dependency) ---


@pytest.mark.parametrize("churn_probability", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_confidence_always_in_valid_range(churn_probability):
    result = confidence(churn_probability)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("churn_probability", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_ease_of_saving_always_in_valid_range(churn_probability):
    result = ease_of_saving(churn_probability)
    assert 0.0 <= result <= 1.0


def test_confidence_and_ease_of_saving_are_complementary():
    for p in (0.0, 0.2, 0.5, 0.8, 1.0):
        assert confidence(p) + ease_of_saving(p) == pytest.approx(1.0)


def test_opportunity_score_non_negative_for_valid_inputs():
    result = opportunity_score(revenue_at_risk=100.0, ease_of_saving_value=0.8, clv_percentile_weight=0.9)
    assert result >= 0.0
    assert result == pytest.approx(100.0 * 0.8 * 0.9)


def test_opportunity_score_drops_clv_term_entirely_when_weight_is_none():
    """clv_percentile_weight=None (no CLV data for this tenant at all)
    must NOT be treated as a 0.0 weight - that would zero out the whole
    product for every customer alike, making the "ranking" arbitrary
    rather than a real, reduced-signal ordering by revenue_at_risk *
    ease_of_saving alone."""
    result = opportunity_score(revenue_at_risk=100.0, ease_of_saving_value=0.8, clv_percentile_weight=None)
    assert result == pytest.approx(100.0 * 0.8)
    assert result != 0.0


def test_recoverable_revenue_uses_labeled_intervention_rate_and_remaining_tenure():
    assert recoverable_revenue(100.0, 12.0) == pytest.approx(100.0 * INTERVENTION_SUCCESS_RATE * 12.0)
    assert INTERVENTION_SUCCESS_RATE == 0.30


def test_expected_remaining_tenure_months_subtracts_tenure_from_contract_median():
    survival_map = {"Month-to-month": 45.0, "One year": 72.0, "Two year": None}
    assert expected_remaining_tenure_months("Month-to-month", 18, survival_map) == pytest.approx(27.0)


def test_expected_remaining_tenure_months_floors_at_minimum_when_tenure_exceeds_median():
    survival_map = {"Month-to-month": 45.0, "One year": 72.0, "Two year": None}
    assert expected_remaining_tenure_months("Month-to-month", 60, survival_map) == MIN_REMAINING_TENURE_MONTHS


def test_expected_remaining_tenure_months_uses_fallback_when_contract_median_is_none():
    survival_map = {"Month-to-month": 45.0, "One year": 72.0, "Two year": None}
    assert expected_remaining_tenure_months("Two year", 10, survival_map) == pytest.approx(
        FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT - 10
    )


def test_business_language_mapping_has_no_raw_column_passthrough_for_known_features():
    assert to_business_language("Contract") == "their contract type"
    assert to_business_language("tenure") != "tenure"


# --- API-level checks ---


def test_business_impact_endpoint_includes_assumption_metadata(client):
    token = _get_token(client)
    response = client.get("/api/business-impact", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    assert "metadata" in payload
    assert payload["metadata"]["intervention_success_rate"] == 0.30
    assert "note" in payload["metadata"]["intervention_success_rate_note"].lower() or len(
        payload["metadata"]["intervention_success_rate_note"]
    ) > 0
    assert "not measured" in payload["metadata"]["intervention_success_rate_note"].lower()


def test_business_impact_endpoint_values_in_valid_ranges(client):
    token = _get_token(client)
    response = client.get("/api/business-impact", headers={"Authorization": f"Bearer {token}"})
    payload = response.json()

    for customer in payload["customers"]:
        assert 0.0 <= customer["confidence"] <= 1.0
        assert customer["revenue_at_risk"] >= 0.0
        assert customer["recoverable_revenue"] >= 0.0
        assert customer["opportunity_score"] >= 0.0


def test_action_queue_sorted_by_opportunity_score_descending(client):
    token = _get_token(client)
    response = client.get("/api/action-queue", params={"limit": 10}, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    payload = response.json()
    scores = [row["opportunity_score"] for row in payload["action_queue"]]
    assert scores == sorted(scores, reverse=True)
    assert "metadata" in payload
    assert payload["metadata"]["intervention_success_rate"] == 0.30


def test_business_language_explanation_contains_no_raw_technical_column_names(client):
    token = _get_token(client)
    priority_response = client.get("/api/priority", params={"limit": 3}, headers={"Authorization": f"Bearer {token}"})
    customer_id = priority_response.json()[0]["customerID"]

    response = client.get(
        f"/api/business-language-explanation/{customer_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    explanation_text = response.json()["explanation"]

    raw_column_names = [
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "InternetService",
        "PaymentMethod",
        "PaperlessBilling",
        "SeniorCitizen",
        "PhoneService",
        "MultipleLines",
        "StreamingTV",
        "StreamingMovies",
    ]
    for raw_name in raw_column_names:
        assert raw_name not in explanation_text


def test_meridian_opportunity_score_is_real_and_ordered_by_revenue_times_ease_of_saving(client):
    """End-to-end against Meridian's real trained artifacts with no CLV
    data (clv_data_path=None) - opportunity_score must be real and
    non-zero (not the old, arbitrary all-zeros result), and the whole
    result must be sorted by exactly revenue_at_risk * ease_of_saving
    (the formula with the CLV term dropped), not some other order."""
    result = compute_business_impact_bulk(
        model_dir=MERIDIAN_MODEL_DIR,
        data_path=MERIDIAN_DATA_PATH,
        clv_data_path=None,
    )

    assert len(result) > 0
    assert (result["opportunity_score"] > 0.0).all()
    expected_score = result["revenue_at_risk"] * result["ease_of_saving"]
    assert result["opportunity_score"].to_numpy() == pytest.approx(expected_score.to_numpy())

    scores = result["opportunity_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_business_impact_metadata_notes_missing_clv_only_when_absent():
    meridian_metadata = business_impact_metadata(
        model_dir=MERIDIAN_MODEL_DIR, data_path=MERIDIAN_DATA_PATH, clv_data_path=None
    )
    assert meridian_metadata["opportunity_score_note"] == (
        "computed without CLV weighting; CLV model not yet trained for this tenant."
    )

    telco_metadata = business_impact_metadata(
        model_dir="models/v1", data_path="data/raw/telco.csv", clv_data_path="data/raw/telco_enriched.csv"
    )
    assert "opportunity_score_note" not in telco_metadata


def test_banking_tenant_gets_real_business_impact_data_without_clv_weighting(client):
    """Banking was fully trained (see config/config.yaml's banking entry) -
    business_impact_core genuinely passed. Real data, not "not yet
    trained" - but no CLV weighting, since bank_churn.csv has no CLV-
    equivalent column at all (clv_data_path stays null for this tenant)."""
    token = _get_token(client, tenant_id="banking")

    for path in ("/api/business-impact", "/api/action-queue"):
        response = client.get(path, headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        body = response.json()
        assert body.get("available") is not False

    impact_response = client.get("/api/business-impact", headers={"Authorization": f"Bearer {token}"})
    assert len(impact_response.json()["customers"]) > 0
    assert "opportunity_score_note" in impact_response.json()["metadata"]


# --- Regression: a self-registered tenant whose column names genuinely
# differ from Telco's must be able to actually load business-impact data,
# not just have the feature flag set. ---
#
# This is the exact bug this session found live: Aurora Streaming showed
# business_impact_core as "enabled" (the classifier's own sanity gate
# passed - that's the only thing that flag depends on) but
# /api/business-impact 500'd with "Missing required columns: ['Churn',
# 'MonthlyCharges', 'customerID']". Root cause: compute_business_impact_bulk()
# called get_priority_ranking() (and did its own raw-data reads) with no
# tenant_config at all - both were hardcoded to Telco's literal column
# names. This went undetected for every self-registered tenant tested
# before Aurora (Meridian, Fernwood, Harborline, Cobalt, Redline) because
# every one of them reuses Telco's exact synthetic schema
# (scripts/generate_onboarding_samples.py's make_telecom_frame()) - the
# hardcoded assumption never actually got exercised. Same bug class this
# project has hit at least twice before (see docs/ADDING_A_TENANT.md's
# recurring-lessons section) - a module generalized for training time but
# not for its own read-time call chain.
AURORA_MODEL_DIR = "models/aurora-streaming"
AURORA_DATA_PATH = "data/tenant_uploads/aurora-streaming.csv"


def test_aurora_business_impact_loads_with_genuinely_different_column_names():
    """End-to-end against Aurora's real trained artifacts - id/target/
    revenue/clv columns (customer_id/churned/monthly_fee/
    lifetime_value_estimate) share NOT ONE literal name with Telco's
    (customerID/Churn/MonthlyCharges/CLTV). Used to crash outright; must
    now return real, sane numbers for every test customer."""
    result = compute_business_impact_bulk(
        model_dir=AURORA_MODEL_DIR,
        data_path=AURORA_DATA_PATH,
        clv_data_path=AURORA_DATA_PATH,  # self-registered tenants: same file as data_path, not a separate enriched one
    )

    assert len(result) > 0
    assert result[["revenue_at_risk", "recoverable_revenue", "opportunity_score", "confidence"]].notna().all().all()
    assert (result["revenue_at_risk"] >= 0.0).all()
    assert (result["recoverable_revenue"] >= 0.0).all()
    # Real CLV weighting (Aurora's clv module passed its own sanity gate) -
    # not the CLV-absent fallback path.
    assert result["clv_percentile_weight"].between(0.0, 1.0).all()


def test_aurora_business_impact_metadata_discloses_missing_duration_column():
    """Aurora's confirmed column mapping never assigned the 'duration' role
    to any column (see this test file's module comment) - distinct from
    just missing a survival model. Must be honestly disclosed, not silently
    treated as if a real per-customer tenure value existed."""
    metadata = business_impact_metadata(
        model_dir=AURORA_MODEL_DIR, data_path=AURORA_DATA_PATH, clv_data_path=AURORA_DATA_PATH
    )
    assert "expected_remaining_tenure_note" in metadata
    assert "no tenure/duration-equivalent column" in metadata["expected_remaining_tenure_note"]


def test_business_impact_loads_for_a_freshly_trained_tenant_with_non_telco_column_names(tmp_path):
    """Self-contained variant of the Aurora regression above (trains its
    own tiny synthetic tenant in tmp_path) so this specific guard runs
    anywhere - CI included - rather than only on a dev machine that
    happens to already have a real self-registered tenant's (gitignored,
    not committed) local artifacts sitting around."""
    import numpy as np

    from src.models.train import train_model

    rng = np.random.default_rng(7)
    n = 320
    tenure_like = rng.integers(0, 60, size=n)
    fee = np.round(rng.normal(20, 5, size=n).clip(5, 40), 2)
    logit = -0.5 - 0.02 * tenure_like + rng.normal(0, 0.6, size=n)
    left = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    value_score = np.round(tenure_like * fee * rng.lognormal(0, 0.4, size=n) + rng.normal(0, 20, size=n), 2).clip(0, None)

    df = pd.DataFrame(
        {
            "acct_id": [f"A{i:04d}" for i in range(n)],
            "months_active": tenure_like,  # deliberately mapped as an ordinary feature, not "duration"
            "fee_amount": fee,
            "value_score": value_score,
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
        "clv_column": "value_score",
        "tuning_enabled": False,  # keep this test fast - not what's under test here
    }
    model_dir = tmp_path / "model"
    train_model(data_path, CONFIG_PATH, output_dir=model_dir, tenant_config=tenant_config)

    result = compute_business_impact_bulk(model_dir=model_dir, data_path=data_path, clv_data_path=data_path)
    assert len(result) > 0
    assert result[["revenue_at_risk", "recoverable_revenue", "opportunity_score", "confidence"]].notna().all().all()

    metadata = business_impact_metadata(model_dir=model_dir, data_path=data_path, clv_data_path=data_path)
    assert "expected_remaining_tenure_note" in metadata  # no duration_column mapped, same as Aurora
