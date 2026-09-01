"""Parts B (column mapping) and C (data-sufficiency validation) of
self-service data onboarding, Stage 1."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base
from src.onboarding.schema_fields import CORE_SIGNAL_FIELDS
from src.onboarding.validation import MIN_RECOMMENDED_FEATURES, MIN_RECOMMENDED_ROWS
from src.onboarding.schema_fields import suggest_mapping


# --- Part B: name-similarity mapping suggestions (pure unit tests) ---


def test_suggest_mapping_exact_and_alias_matches():
    suggestions = suggest_mapping(["customerID", "Churn", "MonthlyCharges", "tenure", "monthly_charge"])
    assert suggestions["customerID"]["suggested_role"] == "customer_id"
    assert suggestions["Churn"]["suggested_role"] == "target"
    assert suggestions["MonthlyCharges"]["suggested_role"] == "revenue"
    # "tenure" suggests role="duration", not "feature" - reversed from an
    # earlier version of this suggester that deliberately kept "tenure"
    # feature-only, requiring a human to manually re-pick "duration" every
    # time (see schema_fields.py's own "tenure" entry comment for the real
    # tenant, cascade-retails, that cost: Survival Analysis silently never
    # trained because nothing ever suggested the more useful role). A
    # duration-mapped column is still kept as an ordinary classifier
    # feature too (src/models/tenant_training.py's _filtered_training_frame()),
    # so this loses nothing Telco's own "tenure" didn't already have.
    assert suggestions["tenure"]["suggested_role"] == "duration"
    assert suggestions["monthly_charge"]["suggested_role"] == "revenue"  # alias match, different casing/underscore


def test_suggest_mapping_fuzzy_name_similarity():
    suggestions = suggest_mapping(["ContractType", "Tech_Support"])
    assert suggestions["ContractType"]["suggested_role"] == "feature"
    assert suggestions["ContractType"]["matched_known_field"] == "Contract"
    assert suggestions["Tech_Support"]["matched_known_field"] == "TechSupport"


def test_suggest_mapping_unrelated_column_falls_back_to_ignore_not_forced():
    suggestions = suggest_mapping(["favorite_color_xyz"])
    assert suggestions["favorite_color_xyz"]["suggested_role"] == "ignore"
    assert suggestions["favorite_color_xyz"]["matched_known_field"] is None


# --- Regression: real tenants whose duration/clv-equivalent columns used
# to be missed entirely by name-similarity matching (a near-miss below
# SUGGESTION_THRESHOLD, or a wrong/no match) - see schema_fields.py's
# "tenure"/CLTV/Churn entries' own comments for the exact real-tenant
# history behind each. Locks in that a genuinely realistic, differently-
# worded upload gets every optional role suggested correctly, not just
# Telco's own literal column names. ---


def test_suggest_mapping_catches_a_real_previously_missed_tenant_schema():
    """cascade-retails' real confirmed column set, before it was corrected
    - see this project's own retrain history. Every one of these five
    roles used to require a manual override; all five should now be
    suggested correctly with no human correction needed."""
    columns = [
        "subscriber_id", "unsubscribed", "subscriber_value_estimate", "tenure_months", "monthly_price",
    ]
    suggestions = suggest_mapping(columns)
    assert suggestions["subscriber_id"]["suggested_role"] == "customer_id"
    assert suggestions["unsubscribed"]["suggested_role"] == "target"
    assert suggestions["subscriber_value_estimate"]["suggested_role"] == "clv"
    assert suggestions["tenure_months"]["suggested_role"] == "duration"
    assert suggestions["monthly_price"]["suggested_role"] == "revenue"


def test_suggest_mapping_does_not_confuse_a_target_column_for_duration():
    """Regression test for a real false-positive found live: "unsubscribed"
    (a plausible real target column name) used to score 0.79 similarity
    against the duration alias "months_subscribed" (both share the
    "subscri..." stem) and got suggested as role="duration" - a genuinely
    dangerous mistake, since a careless "accept all suggestions" click
    would have mapped the actual churn label as a duration column instead
    of a target. Must suggest "target", never "duration", for this name."""
    suggestions = suggest_mapping(["unsubscribed"])
    assert suggestions["unsubscribed"]["suggested_role"] == "target"


# --- Part C: /api/onboarding/validate (integration) ---


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


def _get_token(client, email="onboarding-tester@example.com", tenant_id="telco"):
    client.post("/auth/register", json={"email": email, "password": "s3cret-pw", "tenant_id": tenant_id, "role": "analyst"})
    response = client.post("/auth/login", json={"email": email, "password": "s3cret-pw"})
    return response.json()["access_token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


THIN_CSV = (
    "id,label,tenure\n"
    "1,No,5\n"
    "2,Yes,10\n"
    "3,No,15\n"
)

THIN_MAPPING = {"id": "customer_id", "label": "target", "tenure": "feature"}


def _upload(client, token, content: str, filename="upload.csv"):
    return client.post(
        "/api/onboarding/upload",
        headers=_auth_headers(token),
        files={"file": (filename, content.encode("utf-8"), "text/csv")},
    )


def _build_sufficient_csv() -> str:
    """Generates a CSV that clears BOTH the row-count and feature-richness
    thresholds - >= MIN_RECOMMENDED_ROWS rows, with all 8 core-signal
    feature columns populated and varying."""
    contracts = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet = ["DSL", "Fiber optic", "No"]
    yn = ["Yes", "No"]

    lines = ["customerID,Churn,MonthlyCharges,tenure,Contract,PaymentMethod,InternetService,OnlineSecurity,TechSupport,PaperlessBilling,TotalCharges"]
    n_rows = MIN_RECOMMENDED_ROWS + 10
    for i in range(n_rows):
        churn = "Yes" if i % 4 == 0 else "No"
        lines.append(
            f"C-{i:07d},{churn},{40 + (i % 60):.2f},{1 + (i % 72)},{contracts[i % 3]},"
            f"{payment_methods[i % 4]},{internet[i % 3]},{yn[i % 2]},{yn[i % 2]},{yn[i % 2]},"
            f"{500 + (i % 5000):.2f}"
        )
    return "\n".join(lines) + "\n"


SUFFICIENT_MAPPING = {
    "customerID": "customer_id",
    "Churn": "target",
    "MonthlyCharges": "revenue",
    "tenure": "feature",
    "Contract": "feature",
    "PaymentMethod": "feature",
    "InternetService": "feature",
    "OnlineSecurity": "feature",
    "TechSupport": "feature",
    "PaperlessBilling": "feature",
    "TotalCharges": "feature",
}


def test_validate_blocks_when_id_and_target_not_mapped(client):
    token = _get_token(client)
    upload = _upload(client, token, THIN_CSV).json()
    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": {"id": "ignore", "label": "ignore", "tenure": "feature"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["can_proceed"] is False
    checks_by_name = {c["name"]: c for c in body["checks"]}
    assert checks_by_name["id_column_mapped"]["passed"] is False
    assert checks_by_name["target_column_mapped"]["passed"] is False


def test_validate_blocks_on_duplicate_ids(client):
    token = _get_token(client, email="dup-ids@example.com")
    content = "id,label,tenure\n1,No,5\n1,Yes,10\n"
    upload = _upload(client, token, content).json()
    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": THIN_MAPPING},
    )
    body = response.json()
    checks_by_name = {c["name"]: c for c in body["checks"]}
    assert checks_by_name["id_column_unique_and_non_null"]["passed"] is False
    assert body["can_proceed"] is False


def test_validate_blocks_on_single_class_target(client):
    token = _get_token(client, email="single-class@example.com")
    content = "id,label,tenure\n1,No,5\n2,No,10\n3,No,15\n"
    upload = _upload(client, token, content).json()
    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": THIN_MAPPING},
    )
    body = response.json()
    checks_by_name = {c["name"]: c for c in body["checks"]}
    assert checks_by_name["target_has_two_classes"]["passed"] is False
    assert body["can_proceed"] is False


def test_validate_thin_data_is_a_warning_not_a_hard_block_with_specific_guidance(client):
    token = _get_token(client, email="thin-data@example.com")
    upload = _upload(client, token, THIN_CSV).json()
    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": THIN_MAPPING},
    )
    body = response.json()

    # Structural checks all pass (id/target mapped, unique, two classes) -
    # so this is allowed to proceed, just with a warning.
    assert body["can_proceed"] is True

    sufficiency = body["data_sufficiency"]
    assert sufficiency["row_count"] == 3
    assert sufficiency["row_count_sufficient"] is False
    assert sufficiency["feature_richness_score"] == 1  # only "tenure" mapped as feature
    assert sufficiency["feature_richness_sufficient"] is False
    assert sufficiency["warning"] is True
    assert sufficiency["requires_acknowledgment"] is True
    assert str(MIN_RECOMMENDED_FEATURES) in sufficiency["guidance"]
    assert "Consider adding" in sufficiency["guidance"]
    # tenure was mapped, so it must not appear in what's still missing.
    assert "tenure" not in sufficiency["missing_core_fields"]
    assert "Contract" in sufficiency["missing_core_fields"]


def _build_cobalt_like_csv() -> str:
    """Same shape as _build_sufficient_csv() (all 8 core-signal columns
    present under their correct, recognized names, row count above
    MIN_RECOMMENDED_ROWS), except PaperlessBilling and OnlineSecurity are
    constant-valued throughout - fully populated, present by name, but
    carrying no predictive signal. Mirrors the exact scenario the Cobalt
    Health Network onboarding demo (data/test_onboarding_samples/README.md)
    surfaced: right column names, wrong usability."""
    contracts = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet = ["DSL", "Fiber optic", "No"]
    yn = ["Yes", "No"]

    lines = ["customerID,Churn,MonthlyCharges,tenure,Contract,PaymentMethod,InternetService,OnlineSecurity,TechSupport,PaperlessBilling,TotalCharges"]
    n_rows = MIN_RECOMMENDED_ROWS + 10
    for i in range(n_rows):
        churn = "Yes" if i % 4 == 0 else "No"
        lines.append(
            f"C-{i:07d},{churn},{40 + (i % 60):.2f},{1 + (i % 72)},{contracts[i % 3]},"
            f"{payment_methods[i % 4]},{internet[i % 3]},No,{yn[i % 2]},"
            f"Yes,{500 + (i % 5000):.2f}"
        )
    return "\n".join(lines) + "\n"


def test_validate_distinguishes_unusable_mapped_columns_from_missing_core_fields(client):
    """Cobalt Health Network's exact real-world scenario: every core-signal
    column is present under its correct, recognized name, but two of them
    (PaperlessBilling, OnlineSecurity) are constant-valued throughout - so a
    presence-by-name check alone would call the upload fine. This must
    surface as unusable_mapped_columns, NOT as missing_core_fields, and the
    guidance text must name the actual problem columns rather than just a
    generic low usable-count number."""
    token = _get_token(client, email="cobalt-like@example.com")
    upload = _upload(client, token, _build_cobalt_like_csv()).json()

    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": SUFFICIENT_MAPPING},
    )
    body = response.json()
    assert body["can_proceed"] is True

    sufficiency = body["data_sufficiency"]
    # Row count is sufficient - this is a data-QUALITY problem, not a
    # data-quantity one, unlike the thin-data case above.
    assert sufficiency["row_count_sufficient"] is True
    assert sufficiency["feature_richness_score"] == len(CORE_SIGNAL_FIELDS) - 2
    assert sufficiency["feature_richness_sufficient"] is False

    # Every core field IS present under a recognized name - nothing is
    # actually missing, so missing_core_fields must stay empty...
    assert sufficiency["missing_core_fields"] == []
    # ...while unusable_mapped_columns names the two specific columns whose
    # VALUES (not names) are the actual problem.
    assert sufficiency["unusable_mapped_columns"] == [
        "OnlineSecurity (constant value)",
        "PaperlessBilling (constant value)",
    ]

    assert "These mapped columns exist but aren't usable as-is" in sufficiency["guidance"]
    assert "PaperlessBilling (constant value)" in sufficiency["guidance"]
    assert "OnlineSecurity (constant value)" in sufficiency["guidance"]
    # Nothing is missing by name, so this guidance line must not appear.
    assert "Consider adding" not in sufficiency["guidance"]


def test_validate_sufficient_data_passes_cleanly_with_no_warning(client):
    token = _get_token(client, email="sufficient-data@example.com")
    upload = _upload(client, token, _build_sufficient_csv()).json()
    assert upload["row_count"] == MIN_RECOMMENDED_ROWS + 10

    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": SUFFICIENT_MAPPING},
    )
    body = response.json()
    assert body["can_proceed"] is True
    sufficiency = body["data_sufficiency"]
    assert sufficiency["row_count_sufficient"] is True
    assert sufficiency["feature_richness_score"] == len(CORE_SIGNAL_FIELDS)
    assert sufficiency["feature_richness_sufficient"] is True
    assert sufficiency["warning"] is False
    assert sufficiency["guidance"] is None
    assert sufficiency["requires_acknowledgment"] is False


def test_validate_rejects_unknown_role(client):
    token = _get_token(client, email="bad-role@example.com")
    upload = _upload(client, token, THIN_CSV).json()
    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": upload["upload_id"], "column_mapping": {"id": "not_a_real_role"}},
    )
    assert response.status_code == 400


def test_validate_unknown_upload_id_returns_404(client):
    token = _get_token(client, email="missing-upload@example.com")
    response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(token),
        json={"upload_id": 999999, "column_mapping": {}},
    )
    assert response.status_code == 404


def test_upload_is_tenant_isolated_from_validate(client):
    """A banking-tenant user must never be able to validate a telco
    tenant's upload_id, even if they happen to guess a valid id."""
    telco_token = _get_token(client, email="telco-owner@example.com", tenant_id="telco")
    banking_token = _get_token(client, email="banking-other@example.com", tenant_id="banking")

    upload = _upload(client, telco_token, THIN_CSV).json()

    cross_tenant_response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(banking_token),
        json={"upload_id": upload["upload_id"], "column_mapping": THIN_MAPPING},
    )
    assert cross_tenant_response.status_code == 404

    same_tenant_response = client.post(
        "/api/onboarding/validate",
        headers=_auth_headers(telco_token),
        json={"upload_id": upload["upload_id"], "column_mapping": THIN_MAPPING},
    )
    assert same_tenant_response.status_code == 200
