"""GET /api/tenant-features and src.tenant_registry.feature_coverage() -
confirms the "what you have vs what's possible" feature list is read
live from each tenant's real feature_flags (config.yaml for Telco/
Banking, Company.feature_flags_json for a self-registered tenant), never
cached or hardcoded per-tenant."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.db import get_db
from database.models import Base, Company
from src.tenant_registry import FEATURE_CATALOG, feature_coverage


@pytest.fixture()
def engine_and_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    return engine, TestingSessionLocal


@pytest.fixture()
def client(engine_and_session):
    _engine, TestingSessionLocal = engine_and_session

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


def _get_token(client, email, tenant_id):
    client.post("/auth/register", json={"email": email, "password": "s3cret-pw", "tenant_id": tenant_id, "role": "analyst"})
    response = client.post("/auth/login", json={"email": email, "password": "s3cret-pw"})
    return response.json()["access_token"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_telco_has_every_catalog_feature_except_clv_estimated_enabled():
    """clv_estimated is the one deliberate gap: Telco already has a real,
    trained CLV model (CLTV) - the formula-based proxy is mutually
    exclusive with that, by design (see src/models/tenant_training.py's
    _run_optional_modules()), not a coverage regression."""
    coverage = feature_coverage("telco")
    all_keys = {c["key"] for c in coverage}
    enabled_keys = {c["key"] for c in coverage if c["enabled"]}
    assert all_keys == {key for key, _ in FEATURE_CATALOG}
    assert enabled_keys == all_keys - {"clv_estimated"}
    clv_estimated_entry = next(c for c in coverage if c["key"] == "clv_estimated")
    assert clv_estimated_entry["reason"] is not None
    assert all(c["reason"] is None for c in coverage if c["enabled"])


def test_banking_has_every_module_except_clv_enabled():
    """Banking was fully trained (see config/config.yaml's banking entry) -
    every module genuinely passed its own real sanity gate except clv,
    the one genuine structural gap: bank_churn.csv has no CLV-equivalent
    column at all. test_untrained_self_registered_tenant_has_nothing_
    enabled below covers the genuinely-nothing-trained-yet case now that
    Banking no longer is one."""
    coverage = feature_coverage("banking")
    enabled_keys = {c["key"] for c in coverage if c["enabled"]}
    all_keys = {c["key"] for c in coverage}
    assert enabled_keys == all_keys - {"clv"}
    clv_entry = next(c for c in coverage if c["key"] == "clv")
    assert clv_entry["reason"] is not None
    assert all(c["reason"] is None for c in coverage if c["enabled"])


def test_untrained_self_registered_tenant_has_nothing_enabled(engine_and_session):
    _engine, TestingSessionLocal = engine_and_session
    db = TestingSessionLocal()
    db.add(Company(tenant_id="acme-widgets", company_name="Acme Widgets"))
    db.commit()

    coverage = feature_coverage("acme-widgets", db=db)
    assert all(not c["enabled"] for c in coverage)
    assert all(c["reason"] == "not yet trained for this tenant" for c in coverage)
    db.close()


def test_coverage_reads_live_from_feature_flags_not_cached(engine_and_session):
    """The exact same tenant_id, queried twice, must reflect a real DB
    change made in between - proving this isn't cached anywhere. Mirrors
    what a training job's _apply_training_outcome() actually writes for a
    self-registered tenant that passed its sanity check (business_impact_
    core only - never the whole catalog)."""
    _engine, TestingSessionLocal = engine_and_session
    db = TestingSessionLocal()
    db.add(Company(tenant_id="acme-widgets", company_name="Acme Widgets"))
    db.commit()

    before = feature_coverage("acme-widgets", db=db)
    assert all(not c["enabled"] for c in before)

    company = db.query(Company).filter(Company.tenant_id == "acme-widgets").first()
    company.model_dir = "models/acme-widgets"
    company.data_path = "data/tenant_uploads/acme-widgets.csv"
    company.feature_flags_json = {"business_impact_core": True}
    db.commit()

    after = feature_coverage("acme-widgets", db=db)
    enabled_keys = {c["key"] for c in after if c["enabled"]}
    assert enabled_keys == {"business_impact_core"}  # partial coverage - not the whole catalog
    disabled = [c for c in after if not c["enabled"]]
    assert len(disabled) == len(FEATURE_CATALOG) - 1
    assert all(c["reason"] == "not yet trained for this tenant" for c in disabled)
    db.close()


def test_meridian_style_partial_tenant_and_telco_render_genuinely_different_lists(engine_and_session):
    """Same function, same catalog, two different tenants - a partial-
    coverage self-registered tenant (standing in for the real Meridian
    Wireless, whose actual training outcome is exactly this shape - see
    api/training.py's _apply_training_outcome) must NOT look like Telco's
    full-coverage list."""
    _engine, TestingSessionLocal = engine_and_session
    db = TestingSessionLocal()
    db.add(
        Company(
            tenant_id="meridian-style-co",
            company_name="Meridian-style Co",
            model_dir="models/meridian-style-co",
            data_path="data/tenant_uploads/meridian-style-co.csv",
            feature_flags_json={"business_impact_core": True},
        )
    )
    db.commit()

    partial = feature_coverage("meridian-style-co", db=db)
    full = feature_coverage("telco")

    partial_enabled = {c["key"] for c in partial if c["enabled"]}
    full_enabled = {c["key"] for c in full if c["enabled"]}
    assert partial_enabled == {"business_impact_core"}
    # Telco's one deliberate gap: clv_estimated, mutually exclusive with
    # its real, trained CLV model - see test_telco_has_every_catalog_
    # feature_except_clv_estimated_enabled above.
    assert full_enabled == {key for key, _ in FEATURE_CATALOG} - {"clv_estimated"}
    assert partial_enabled != full_enabled
    db.close()


def test_tenant_features_endpoint_reflects_real_tenant(client, engine_and_session):
    _engine, TestingSessionLocal = engine_and_session
    telco_token = _get_token(client, "features-telco@example.com", "telco")

    response = client.get("/api/tenant-features", headers=_auth_headers(telco_token))
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "telco"
    assert len(body["features"]) == len(FEATURE_CATALOG)
    # clv_estimated is the one deliberate gap - see test_telco_has_every_
    # catalog_feature_except_clv_estimated_enabled above.
    assert all(f["enabled"] for f in body["features"] if f["key"] != "clv_estimated")
    assert not next(f for f in body["features"] if f["key"] == "clv_estimated")["enabled"]


def test_tenant_features_endpoint_requires_authentication(client):
    response = client.get("/api/tenant-features")
    assert response.status_code == 401
