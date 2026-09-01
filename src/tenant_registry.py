"""Single source of truth for which tenants exist, what data/model
artifacts back each one, and which optional modules have been
independently validated and enabled for them.

Replaces the old pattern of hardcoding `if tenant_id != "telco": return
NOT_TRAINED` at every endpoint/tool call site. Enabling a validated
feature for a tenant is now a config.yaml `feature_flags` edit, not a
code change - see docs/ADDING_A_TENANT.md for what re-validation each
module needs before its flag is safe to flip to true.

Two sources of tenant profiles, checked in order:
  1. config.yaml's static `tenants:` section - the hand-curated, human-
     reviewed demo tenants (telco/banking). Always takes priority.
  2. The database.models.Company row, for a self-registered tenant
     (POST /api/companies/register) once a training job has actually
     succeeded with a sane result (api/training.py) - model_dir/
     data_path/feature_flags_json are all still None/empty otherwise, so
     a not-yet-trained self-registered tenant naturally falls through to
     the same "not yet trained for this tenant" response as before.
     Only consulted when a `db` session is passed in - callers that never
     pass one (e.g. anything computed once at process startup) simply
     never see self-registered tenants, which is correct for them too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.config import load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"

DEFAULT_UNAVAILABLE_REASON = "not yet trained for this tenant"

# Every feature_flags key this project has, with a human-readable label -
# the single catalog backing any UI that wants to show "what you have vs
# what's possible" (see api/main.py's /api/tenant-features). Mirrors
# config.yaml's feature_flags: comment exactly - add a new module's flag
# here too, or it silently won't show up in that UI.
FEATURE_CATALOG: list[tuple[str, str]] = [
    ("business_impact_core", "Business Impact, Action Queue & Alerts"),
    ("priority_ranking", "Priority Ranking"),
    ("backtest", "Backtesting"),
    ("survival", "Survival Analysis"),
    ("segments", "Customer Segments"),
    ("anomalies", "Anomaly Detection"),
    ("clv", "Customer Lifetime Value"),
    ("clv_estimated", "Estimated Customer Lifetime Value (formula-based)"),
    ("scenario_simulator", "Scenario Simulator"),
    ("budget_optimizer", "Budget Optimizer"),
    ("customer_timeline", "Customer Timeline"),
]


def load_tenant_profiles(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    config = load_config(config_path)
    return config.get("tenants", {})


def _company_backed_profile(tenant_id: str, db: Session) -> dict[str, Any] | None:
    """Synthesizes a config.yaml-shaped profile dict from a self-registered
    tenant's Company row, or None if there isn't one (never registered) or
    it hasn't been trained yet (model_dir still unset) - either way, the
    caller falls back to the same "not yet trained" behavior as before this
    existed. Local import to avoid a hard DB dependency for every caller
    that never passes a db session (database.models never imports this
    module, so no circular-import risk)."""
    from database.models import Company

    company = db.query(Company).filter(Company.tenant_id == tenant_id).first()
    if company is None or not company.model_dir:
        return None
    return {
        "model_dir": company.model_dir,
        "data_path": company.data_path,
        # Unlike Telco (whose CLV model trains against a SEPARATE enriched
        # dataset/model_dir - see config.yaml's clv_model_dir/clv_data_path),
        # a self-registered tenant's clv.py run (src/models/tenant_training.py's
        # _run_optional_modules()) writes clv_model.pkl into this SAME
        # model_dir, against this SAME filtered data_path - so that's what
        # clv_model_dir_for()/clv_data_path_for() must point at too, not
        # None (which would 500 on load_clv_importances(model_dir=None)
        # for any tenant whose clv flag is actually true).
        "clv_model_dir": company.model_dir,
        "clv_data_path": company.data_path,
        "unavailable_reason": DEFAULT_UNAVAILABLE_REASON,
        "feature_flags": company.feature_flags_json or {},
    }


def get_tenant_profile(
    tenant_id: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> dict[str, Any] | None:
    profiles = profiles if profiles is not None else load_tenant_profiles()
    profile = profiles.get(tenant_id)
    if profile is not None:
        return profile
    if db is None:
        return None
    return _company_backed_profile(tenant_id, db)


def feature_enabled(
    tenant_id: str,
    feature: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> bool:
    profile = get_tenant_profile(tenant_id, profiles, db=db)
    if profile is None:
        return False
    return bool(profile.get("feature_flags", {}).get(feature, False))


def unavailable_response(
    tenant_id: str,
    feature: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> dict[str, Any]:
    profile = get_tenant_profile(tenant_id, profiles, db=db) or {}
    reason = profile.get("unavailable_reasons", {}).get(feature) or profile.get(
        "unavailable_reason", DEFAULT_UNAVAILABLE_REASON
    )
    return {"available": False, "reason": reason}


def feature_coverage(
    tenant_id: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """Every entry in FEATURE_CATALOG for tenant_id, enabled or not - reads
    feature_enabled()/unavailable_response() live for each flag (same
    calls every gated endpoint already makes), never cached and never
    hardcoded per-tenant. A tenant with a partial rollout (e.g. a self-
    registered Company-backed profile) and one with everything enabled
    (e.g. Telco) naturally produce different lists from the exact same
    function, since the profile lookup underneath is per-tenant."""
    coverage = []
    for key, label in FEATURE_CATALOG:
        enabled = feature_enabled(tenant_id, key, profiles, db=db)
        reason = None if enabled else unavailable_response(tenant_id, key, profiles, db=db)["reason"]
        coverage.append({"key": key, "label": label, "enabled": enabled, "reason": reason})
    return coverage


def model_dir_for(
    tenant_id: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> Path | None:
    profile = get_tenant_profile(tenant_id, profiles, db=db)
    if not profile or "model_dir" not in profile:
        return None
    return ROOT / profile["model_dir"]


def data_path_for(
    tenant_id: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> Path | None:
    profile = get_tenant_profile(tenant_id, profiles, db=db)
    if not profile or "data_path" not in profile:
        return None
    return ROOT / profile["data_path"]


def clv_model_dir_for(
    tenant_id: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> Path | None:
    profile = get_tenant_profile(tenant_id, profiles, db=db) or {}
    clv_dir = profile.get("clv_model_dir")
    return ROOT / clv_dir if clv_dir else None


def clv_data_path_for(
    tenant_id: str,
    profiles: dict[str, dict[str, Any]] | None = None,
    db: Session | None = None,
) -> Path | None:
    profile = get_tenant_profile(tenant_id, profiles, db=db) or {}
    clv_path = profile.get("clv_data_path")
    return ROOT / clv_path if clv_path else None
