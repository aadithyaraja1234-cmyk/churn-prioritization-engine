"""FastAPI app. Every data-returning endpoint is protected by get_current_user()
(interactive JWT auth) or, for a deliberate subset of system-integration-friendly
endpoints, get_current_user_or_api_key() (JWT or an X-API-Key header - see
api/api_keys.py)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# MUST run before any project module is imported below - api/auth.py and
# database/db.py both read JWT_SECRET_KEY/DATABASE_URL from os.environ at
# MODULE IMPORT time (not lazily inside a function), so calling this any
# later than this point is too late: a real .env value would already have
# lost the race and been silently ignored, with api/auth.py falling back
# to its (loudly-warned-about, but easy to miss if you're not watching
# startup logs) dev default. Found live: setting a real JWT_SECRET_KEY in
# .env had NO effect until this was added, because the only load_dotenv()
# call in the whole codebase used to live in src/onboarding/ai_mapping.py,
# imported (via api/onboarding.py, below) AFTER api.auth had already read
# the environment.
load_dotenv()

from api.auth import CurrentUser, get_current_user
from api.auth import router as auth_router
from api.api_keys import get_current_user_or_api_key
from api.api_keys import router as api_keys_router
from api.companies import router as companies_router
from api.onboarding import router as onboarding_router
from api.training import router as training_router
from database.db import get_db, get_tenant_scoped_query, init_db
from database.models import Customer, CustomerTracking, Prediction, RecommendationLog, Scenario
from src.copilot import tools as copilot_tools
from src.models.alerts import scan_for_alerts
from src.models.anomaly import load_flagged_anomalies
from src.models.budget_optimizer import InvalidBudgetError, optimize_budget
from src.models.business_impact import (
    business_impact_metadata,
    business_language_explanation,
    compute_business_impact_bulk,
)
from src.models.clv import load_clv_estimate_summary, load_clv_metrics, load_feature_importances as load_clv_importances
from src.models.explain import get_global_importance, list_test_split_customer_ids
from src.models.health_score import CustomerNotFoundError as HealthScoreCustomerNotFoundError, compute_health_score
from src.models.prioritize import get_priority_ranking
from src.models.backtest import run_backtest as run_backtest_module, run_backtest_curve as run_backtest_curve_module
from src.models.recommend import (
    CustomerNotFoundError as RecommendCustomerNotFoundError,
    recommend_action_for_customer,
)
from src.models.scenario import InvalidScenarioError, get_customer_scenario_impact, run_scenario
from src.models.segment import load_cluster_profiles
from src.models.timeline import CustomerNotFoundError as TimelineCustomerNotFoundError, load_customer_snapshot as timeline_load_customer_snapshot
from src.models.survival import (
    CustomerNotFoundError as SurvivalCustomerNotFoundError,
    churn_likelihood_for_customer,
    median_survival_by_contract,
)
from src.models.whatif import (
    CustomerNotFoundError as WhatIfCustomerNotFoundError,
    InvalidCategoryError,
    InvalidFeatureError,
    simulate_whatif,
)
from src.copilot.agent import (
    CopilotUnavailableError,
    chat as copilot_chat,
    generate_morning_brief as copilot_morning_brief,
)
from src.tenant_registry import (
    clv_data_path_for,
    clv_model_dir_for,
    data_path_for,
    feature_coverage,
    feature_enabled,
    load_tenant_profiles,
    model_dir_for,
    unavailable_response,
)

API_DESCRIPTION = """
A multi-tenant churn-prediction and retention-decisioning API: it scores each
customer's churn probability, translates that into a dollar figure (revenue at
risk / recoverable revenue), ranks who to act on first, and lets you simulate
"what if we changed X" before spending a retention budget for real.

Every number this API returns is either a real, trained model's output or an
explicitly labeled assumption - nothing is fabricated. When a module hasn't
been validated for a given tenant yet, the endpoint returns
`{"available": false, "reason": "..."}` rather than a guess.

## Authentication

This API supports two independent auth methods:

- **JWT bearer token** (`Authorization: Bearer <token>`) - for interactive,
  human use. Obtain one via `POST /auth/login`. Required for every endpoint
  not listed below, including account management (`/api/api-keys/*`) and the
  AI Copilot (`/api/copilot/*`), which stay JWT-only regardless of what API
  keys exist.
- **API key** (`X-API-Key: <key>`) - for server-to-server integrations. Once
  you've logged in with a JWT, create a key via `POST /api/api-keys`, then use
  it directly (no login step needed) against the deliberate subset of
  data-read endpoints tagged for API-key use below (business impact, priority
  ranking, alerts, scenarios, survival, segments, anomalies, CLV, and customer
  detail lookups). Each key is scoped to one tenant and rate-limited
  (60 requests/minute by default).

## Quick start (API key)

```bash
# 1. Log in once to get a JWT (interactive/browser flow)
curl -X POST http://localhost:8000/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email": "demo-telco@churn-engine.local", "password": "DemoPass123"}'

# 2. Create an API key using that JWT (do this once; save the raw key - it is
#    only ever shown in this response)
curl -X POST http://localhost:8000/api/api-keys \\
  -H "Authorization: Bearer <jwt_from_step_1>" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "my-integration"}'

# 3. Call a protected endpoint with only the API key from here on - no JWT needed
curl http://localhost:8000/api/business-impact -H "X-API-Key: <key_from_step_2>"
```
"""

app = FastAPI(
    title="Churn Engine API",
    description=API_DESCRIPTION,
    version="1.0.0",
)
app.include_router(auth_router)
app.include_router(api_keys_router)
app.include_router(companies_router)
app.include_router(onboarding_router)
app.include_router(training_router)

# Idempotent: only creates tables that don't already exist yet (e.g.
# customer_tracking, added after this app.db was first set up) - never
# touches or drops existing tables/data.
init_db()

# The frontend (Vite dev server, or the real deployed frontend origin) runs on
# a different origin than the API, so without this, browsers block every
# request as cross-origin - curl/pytest never hit this since CORS is a
# browser-enforced restriction, not a server one. Vite falls back to 5174+
# when 5173 is already taken, so both are allowed by default. CORS_ORIGINS
# (comma-separated) adds real deployed origins on top of the local-dev
# defaults - set it to your deployed frontend's URL(s) before going live;
# the local-dev origins stay allowed alongside it since Docker/local
# workflows still rely on them.
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
_EXTRA_CORS_ORIGINS = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "").split(",") if origin.strip()]
# Optional regex, for platforms with dynamic per-deployment preview URLs
# that a static CORS_ORIGINS list can't enumerate - e.g. Vercel preview
# deployments (https://<project>-<hash>-<team>.vercel.app, a different
# URL every push). Unset by default (the safer default: only the exact
# origins in CORS_ORIGINS are allowed) - set CORS_ORIGIN_REGEX to something
# like r"https://your-project.*\.vercel\.app" to allow every preview build
# for that one project, never an open-ended wildcard across all of Vercel.
_CORS_ORIGIN_REGEX = os.environ.get("CORS_ORIGIN_REGEX") or None
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_CORS_ORIGINS + _EXTRA_CORS_ORIGINS,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent.parent

# Loaded once at import time from config.yaml's "tenants" section - the
# single source of truth for every tenant's data/model paths and which
# advanced modules are validated/enabled for it. See src/tenant_registry.py
# and docs/ADDING_A_TENANT.md.
TENANT_PROFILES = load_tenant_profiles()

# Classifier assets exist per-tenant (any tenant with a model_dir gets
# Model Performance / Feature Importance / Cohort Comparison automatically -
# this isn't gated by feature_flags, since it's the base artifact every
# tenant profile declares). Advanced modules (survival/segment/anomaly/CLV/
# etc.) are separately gated per-tenant by feature_flags below.
CLASSIFIER_MODEL_DIRS = {
    tenant_id: model_dir_for(tenant_id, TENANT_PROFILES) for tenant_id in TENANT_PROFILES
}
CLASSIFIER_DATA_PATHS = {
    tenant_id: data_path_for(tenant_id, TENANT_PROFILES) for tenant_id in TENANT_PROFILES
}

class WhatIfRequest(BaseModel):
    customer_id: str
    overrides: dict[str, Any]


class ScenarioRequest(BaseModel):
    scenario_type: str
    params: dict[str, Any]
    scenario_name: str


class BudgetOptimizerRequest(BaseModel):
    budget_amount: float
    cost_per_intervention: float | None = None
    stop_at_positive_roi: bool = False


class TrackingUpdateRequest(BaseModel):
    assigned_manager: str | None = None
    status: str = "not_started"
    call_scheduled_date: str | None = None  # ISO date/datetime string, or null to clear


class CopilotChatRequest(BaseModel):
    message: str


@app.get(
    "/health",
    tags=["Health"],
    summary="Liveness check",
    description="Unauthenticated liveness probe. Returns `{\"status\": \"ok\"}` if the API "
    "process is up. Does not check database connectivity or model availability - use this "
    "only to confirm the server is running, e.g. from a load balancer or deploy script.",
)
def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/api/tenant-features",
    tags=["Account"],
    summary="Get every known feature and whether it's enabled for your tenant",
    description="Returns every feature_flags entry this project has (src.tenant_registry's "
    "FEATURE_CATALOG), each marked enabled/disabled for your tenant with a reason when "
    "disabled - the same live feature_enabled()/unavailable_response() calls every gated "
    "endpoint already makes, never a cached or per-tenant-hardcoded list. For showing a "
    "company both what it has access to and what more is possible once additional modules "
    "are trained and validated for it.",
)
def tenant_features(current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> Any:
    return {
        "tenant_id": current_user.tenant_id,
        "features": feature_coverage(current_user.tenant_id, TENANT_PROFILES, db=db),
    }


@app.get(
    "/api/model/metrics",
    tags=["Admin/Cohort"],
    summary="Get the classifier's evaluation metrics for your tenant",
    description="Returns the trained churn classifier's held-out evaluation metrics for your "
    "tenant: ROC-AUC, PR-AUC, train/test set sizes, and churn rates. For an admin or analyst "
    "who wants to know how trustworthy the model is before acting on its predictions. Returns "
    "an `{\"available\": false, ...}` shape if no model has been trained for your tenant yet.",
)
def model_metrics(current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    # CLASSIFIER_MODEL_DIRS is a startup-time snapshot of ONLY config.yaml's
    # static tenants (telco/banking) - it never includes a self-registered
    # tenant, so falling back to model_dir_for(..., db=db) here (same fix
    # shape as every other missing-db= bug this project has hit) is
    # required for this endpoint to ever work for one, not an optimization.
    model_dir = CLASSIFIER_MODEL_DIRS.get(current_user.tenant_id) or model_dir_for(
        current_user.tenant_id, TENANT_PROFILES, db=db
    )
    metadata_path = model_dir / "metadata.json" if model_dir else None
    if metadata_path is None or not metadata_path.exists():
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@app.get(
    "/api/model/importance",
    tags=["Admin/Cohort"],
    summary="Get the classifier's global feature importances",
    description="Returns which input features most influence the churn classifier's "
    "predictions, ranked by importance, for your tenant's model. For an admin or analyst "
    "auditing what the model actually learned, not a per-customer explanation (see "
    "`/api/business-language-explanation/{customer_id}` for that).",
)
def model_importance(current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> Any:
    model_dir = CLASSIFIER_MODEL_DIRS.get(current_user.tenant_id) or model_dir_for(
        current_user.tenant_id, TENANT_PROFILES, db=db
    )
    if model_dir is None:
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    importance_df = get_global_importance(model_dir=model_dir)
    return importance_df.to_dict(orient="records")


def _cohort_stats(model_dir: Path) -> dict[str, Any] | None:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    importance_df = get_global_importance(model_dir=model_dir)
    return {
        "roc_auc": metadata["roc_auc"],
        "pr_auc": metadata["pr_auc"],
        "n_train": metadata["n_train"],
        "n_test": metadata["n_test"],
        "churn_rate_train": metadata["churn_rate_train"],
        "churn_rate_test": metadata["churn_rate_test"],
        "top_features": importance_df.head(5).to_dict(orient="records"),
    }


@app.get(
    "/api/cohort-comparison",
    tags=["Admin/Cohort"],
    summary="Compare the Telco and Banking models side by side",
    description="Returns evaluation metrics and top features for both the Telco and Banking "
    "tenants' models, side by side, regardless of which tenant you're logged in as. For an "
    "admin comparing model quality across cohorts. Not scoped to your own tenant's data - this "
    "is an intentional cross-tenant aggregate comparison of already-stored model metadata, "
    "never per-customer information.",
)
def cohort_comparison(current_user: CurrentUser = Depends(get_current_user)) -> Any:
    """Telco vs Banking, side by side. Pure display of already-stored
    metadata.json + already-fitted feature importances for both tenants -
    not scoped to current_user.tenant_id, since this is an intentional
    cross-tenant aggregate comparison, not per-customer data.
    """
    return {
        "telco": _cohort_stats(CLASSIFIER_MODEL_DIRS["telco"]),
        "banking": _cohort_stats(CLASSIFIER_MODEL_DIRS["banking"]),
    }


@app.get(
    "/api/priority",
    tags=["Priority & Backtest"],
    summary="Get customers ranked by retention priority",
    description="Returns customers ranked for retention outreach, highest priority first, "
    "using the given `strategy` (default `revenue_weighted`, which ranks by expected dollar "
    "impact rather than raw churn probability alone). For a retention team deciding who to "
    "call first with a limited amount of time. Returns an `{\"available\": false, ...}` shape "
    "if priority ranking hasn't been validated for your tenant.",
)
def priority(
    strategy: str = "revenue_weighted",
    limit: int = 50,
    current_user: CurrentUser = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "priority_ranking", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "priority_ranking", TENANT_PROFILES, db=db)
    ranking = get_priority_ranking(
        strategy=strategy,
        model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
    )
    return ranking.head(limit).to_dict(orient="records")


@app.get(
    "/api/backtest",
    tags=["Priority & Backtest"],
    summary="Backtest a single retention-outreach threshold",
    description="Scores what would have happened, on real historical test data, if you had "
    "acted on the top `top_pct` fraction of customers by priority - real revenue captured "
    "under revenue-weighted ranking vs. probability-only ranking vs. random selection. For "
    "showing a decision-maker that the ranking strategy actually beats acting at random, at "
    "one specific threshold. See `/api/backtest/curve` for every threshold at once.",
)
def backtest(
    top_pct: float = 0.2,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "backtest", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "backtest", TENANT_PROFILES, db=db)
    return run_backtest_module(
        top_pct=top_pct,
        model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
    )


@app.get(
    "/api/backtest/curve",
    tags=["Priority & Backtest"],
    summary="Backtest every retention-outreach threshold from min_pct to max_pct",
    description="Same computation as `/api/backtest`, scored at every percentage point from "
    "`min_pct` to `max_pct` (inclusive, step `step_pct`) in one call. Powers an interactive "
    "'what if we could act on X% of customers' slider with real, exactly-computed values at "
    "every point a user can drag to - not an interpolation between a few pre-validated "
    "thresholds. Rejects the request with 400 if `min_pct > max_pct` or `step_pct < 1`.",
)
def backtest_curve(
    min_pct: int = 1,
    max_pct: int = 100,
    step_pct: int = 1,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Same backtest, scored at every percentage-point from min_pct to
    max_pct (inclusive) - powers the Dashboard's interactive ROI slider
    with real, exactly-computed values at every point a user can drag to,
    not an interpolation between a few pre-validated thresholds. Backed by
    run_backtest_curve(), which loads the model/ranks the test set once
    and reuses that across every requested point."""
    if not feature_enabled(current_user.tenant_id, "backtest", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "backtest", TENANT_PROFILES, db=db)
    if not (0 < min_pct <= max_pct <= 100) or step_pct < 1:
        raise HTTPException(
            status_code=400, detail="require 0 < min_pct <= max_pct <= 100 and step_pct >= 1"
        )
    top_pcts = [pct / 100 for pct in range(min_pct, max_pct + 1, step_pct)]
    return run_backtest_curve_module(
        top_pcts,
        model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
    )


@app.get(
    "/api/survival/segments",
    tags=["Survival"],
    summary="Get median survival time by contract type",
    description="Returns the median time-to-churn (from a fitted Cox proportional-hazards "
    "survival model) broken down by contract type for your tenant. For understanding how "
    "contract terms relate to how long customers typically stay. Returns an "
    "`{\"available\": false, ...}` shape if survival analysis hasn't been validated for your "
    "tenant.",
)
def survival_segments(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "survival", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "survival", TENANT_PROFILES, db=db)
    return median_survival_by_contract(
        model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
    )


@app.get(
    "/api/survival/likelihood/{customer_id}",
    tags=["Survival"],
    summary="Get one customer's churn likelihood within a time window",
    description="Returns a single customer's probability of churning within a given time "
    "window, from the fitted Cox proportional-hazards survival model. For a retention rep "
    "asking 'not just whether, but roughly when might this customer churn'. Returns 404 if the "
    "customer doesn't exist in this tenant's data.",
)
def survival_likelihood(
    customer_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "survival", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "survival", TENANT_PROFILES, db=db)
    try:
        return churn_likelihood_for_customer(
            customer_id,
            model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        )
    except SurvivalCustomerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get(
    "/api/segments",
    tags=["Segments & Anomalies"],
    summary="Get customer segment (cluster) profiles",
    description="Returns the real, already-fitted K-Means customer segments for your tenant, "
    "each with its size and defining characteristics. For understanding what distinct types "
    "of customers make up your portfolio, rather than treating every customer as equivalent. "
    "Returns an `{\"available\": false, ...}` shape if segmentation hasn't been validated for "
    "your tenant.",
)
def segments(current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> Any:
    if not feature_enabled(current_user.tenant_id, "segments", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "segments", TENANT_PROFILES, db=db)
    profiles = load_cluster_profiles(
        model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
    )
    return profiles.to_dict(orient="records")


@app.get(
    "/api/anomalies",
    tags=["Segments & Anomalies"],
    summary="Get the most anomalous customers",
    description="Returns up to `limit` customers flagged as statistical outliers (via "
    "isolation forest), most anomalous first. For surfacing customers whose profile doesn't "
    "resemble the rest of the portfolio - worth a manual look before assuming the model's "
    "standard prediction applies cleanly. Returns an `{\"available\": false, ...}` shape if "
    "anomaly detection hasn't been validated for your tenant.",
)
def anomalies(
    limit: int = 5,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "anomalies", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "anomalies", TENANT_PROFILES, db=db)
    results = load_flagged_anomalies(
        model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
    )
    flagged = results[results["anomaly_flag"] == -1].sort_values("anomaly_score").head(limit)
    return flagged.to_dict(orient="records")


@app.get(
    "/api/clv/importance",
    tags=["CLV"],
    summary="Get customer-lifetime-value model metrics and feature importances",
    description="Returns the fitted CLV (customer lifetime value) model's evaluation metrics "
    "and which features most influence it, for your tenant. NOTE: there is no per-customer "
    "dollar CLV prediction anywhere in this API - only these aggregate model-level metrics. "
    "Returns an `{\"available\": false, ...}` shape if CLV modeling hasn't been validated for "
    "your tenant.",
)
def clv_importance(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "clv", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "clv", TENANT_PROFILES, db=db)
    clv_model_dir = clv_model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    importances = load_clv_importances(model_dir=clv_model_dir)
    metrics = load_clv_metrics(model_dir=clv_model_dir)
    return {
        **metrics,
        "feature_importances": [{"feature": feature, "importance": importance} for feature, importance in importances],
    }


@app.get(
    "/api/clv/estimate",
    tags=["CLV"],
    summary="Get a formula-based estimated CLV summary for tenants with no real CLV data",
    description="Returns a formula-based CLV PROXY (revenue x expected total lifetime) for tenants "
    "with no real, per-customer CLV column at all - NOT a trained regression, NOT a measured value. "
    "Mutually exclusive with /api/clv/importance: a tenant with real CLV data uses that endpoint "
    "instead. Always includes a `methodology_note` disclosing the estimate's formula and "
    "assumptions. Returns an `{\"available\": false, ...}` shape if this hasn't been validated for "
    "your tenant.",
)
def clv_estimate(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "clv_estimated", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "clv_estimated", TENANT_PROFILES, db=db)
    model_dir = model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    data_path = data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    return load_clv_estimate_summary(model_dir=model_dir, data_path=data_path)


@app.post(
    "/api/whatif",
    tags=["Business Impact"],
    summary="Simulate a hypothetical change to one customer's profile",
    description="Re-scores a single customer's churn probability under a hypothetical change "
    "to their profile (e.g. 'what if their contract were month-to-month instead of two-year'). "
    "For exploring which levers actually move risk for a specific customer before committing to "
    "an intervention. The re-scored probability is logged to that customer's real audit trail "
    "(see `/api/customer/{customer_id}/timeline`). Returns 404 if the customer doesn't exist, "
    "400 if a given override feature/value is invalid.",
)
def whatif(
    request: WhatIfRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    model_dir = model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    try:
        result = simulate_whatif(
            request.customer_id,
            request.overrides,
            model_dir=model_dir,
            data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        )
    except WhatIfCustomerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidFeatureError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "valid_features": exc.valid_features})
    except InvalidCategoryError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "valid_values": exc.valid_values})

    # A what-if run is a genuine, explicit re-scoring of this one customer -
    # log it to the real predictions audit trail (Customer 360 timeline reads
    # this back). The "-whatif" suffix on model_version distinguishes it from
    # the one-time baseline snapshot migrate_csv_to_db.py wrote for every
    # customer, so the timeline can show which model produced which number.
    db.add(
        Prediction(
            tenant_id=current_user.tenant_id,
            customer_id=request.customer_id,
            model_version=f"{model_dir.name}-whatif",
            churn_probability=result["new_probability"],
        )
    )
    db.commit()

    return result


@app.get(
    "/api/health-score/{customer_id}",
    tags=["Business Impact"],
    summary="Get one customer's composite health score",
    description="Returns a single customer's composite health score (0-100), combining churn "
    "probability, CLV percentile rank, and other signals into one number, plus its component "
    "breakdown. For a quick, at-a-glance read on customer health without interpreting several "
    "separate metrics. Returns 404 if the customer doesn't exist in this tenant's data.",
)
def health_score(
    customer_id: str, current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    try:
        return compute_health_score(
            customer_id,
            model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            clv_model_dir=clv_model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            clv_data_path=clv_data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        )
    except HealthScoreCustomerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get(
    "/api/recommend/{customer_id}",
    tags=["Business Impact"],
    summary="Get the recommended retention action for one customer",
    description="Returns the single recommended retention action for a customer (e.g. offer a "
    "discount, schedule a call) and the specific signals that triggered it. For a retention rep "
    "who needs one clear next step, not a raw probability to interpret themselves. Pass "
    "`log_event=true` only for a genuine, explicit single-customer lookup (e.g. a 'Get "
    "Recommendation' button) - it writes a permanent audit-trail row; bulk dashboard views that "
    "call this per-row should leave it `false` (the default) to avoid flooding the audit trail.",
)
def recommend(
    customer_id: str,
    log_event: bool = False,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """log_event defaults to False because this endpoint is ALSO called
    ~20x per Dashboard page load (PriorityTable fetches one recommendation
    per visible row, purely for display). Logging every one of those would
    flood recommendation_log with incidental bulk-display noise, not real
    per-customer activity - the exact flooding problem the Customer 360
    timeline is designed to avoid (see customer_timeline()'s docstring).
    Only a deliberate, explicit lookup (Customer 360's "Get Recommendation"
    button) passes log_event=true and creates a real audit-trail row.
    """
    if not feature_enabled(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    try:
        result = recommend_action_for_customer(
            customer_id,
            model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            clv_data_path=clv_data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        )
    except RecommendCustomerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (KeyError, ValueError) as exc:
        # explain_customer() raises KeyError if the customer doesn't exist at
        # all, or ValueError if they exist but aren't in the saved test split
        # (this endpoint only scores test-set customers, same as /api/priority).
        raise HTTPException(status_code=404, detail=str(exc))

    if log_event:
        db.add(
            RecommendationLog(
                tenant_id=current_user.tenant_id,
                customer_id=customer_id,
                recommended_action=result["recommended_action"],
                triggered_by_json=result["triggered_by"],
            )
        )
        db.commit()

    return result


@app.get(
    "/api/alerts",
    tags=["Business Impact"],
    summary="Get active portfolio and per-customer alerts",
    description="Returns currently active alerts - both portfolio-level (e.g. total revenue "
    "at risk exceeding a threshold) and per-customer - for your tenant. For surfacing what "
    "needs attention right now without having to scan the full customer list. Also available "
    "as an API key-authenticated endpoint for system integrations (see the API-key quick start "
    "in this API's top-level description). Returns an `{\"available\": false, ...}` shape if "
    "the business-impact module hasn't been validated for your tenant.",
)
def alerts(
    current_user: CurrentUser = Depends(get_current_user_or_api_key), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    return scan_for_alerts(
        model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        clv_data_path=clv_data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
    )


# "Customers needing action" is approximated as churn_probability >= 0.3 -
# the complement of recommend.py rule (e)'s "no action needed" threshold -
# rather than running the full recommender against all 1409 test customers
# (which would take minutes; explain_customer() alone is ~0.15s/customer).
NEEDS_ACTION_CHURN_PROBABILITY_THRESHOLD = 0.3


@app.get(
    "/api/business-impact",
    tags=["Business Impact"],
    summary="Get the full portfolio's revenue-at-risk summary",
    description="Returns every customer's revenue at risk, recoverable revenue, and "
    "opportunity score, plus portfolio-level totals - the core dollar-impact view of churn "
    "for your tenant. For a business stakeholder who wants to know 'how much revenue is at "
    "stake, and from whom' in one call. Also available as an API key-authenticated endpoint "
    "for system integrations. Returns an `{\"available\": false, ...}` shape if this module "
    "hasn't been validated for your tenant.",
)
def business_impact(
    current_user: CurrentUser = Depends(get_current_user_or_api_key), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    model_dir = model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    data_path = data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    clv_data_path = clv_data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    impact_df = compute_business_impact_bulk(model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path)
    customers = impact_df[
        ["customerID", "revenue_at_risk", "recoverable_revenue", "opportunity_score", "confidence"]
    ].to_dict(orient="records")
    needs_action_count = int((impact_df["churn_probability"] >= NEEDS_ACTION_CHURN_PROBABILITY_THRESHOLD).sum())
    return {
        "customers": customers,
        "total_revenue_at_risk": float(impact_df["revenue_at_risk"].sum()),
        "total_recoverable_revenue": float(impact_df["recoverable_revenue"].sum()),
        "customers_needing_action_count": needs_action_count,
        "needs_action_churn_probability_threshold": NEEDS_ACTION_CHURN_PROBABILITY_THRESHOLD,
        "metadata": business_impact_metadata(model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path),
    }


@app.get(
    "/api/action-queue",
    tags=["Business Impact"],
    summary="Get a ready-to-work queue of top-priority customers with recommended actions",
    description="Returns the top `limit` customers by opportunity score, each already paired "
    "with its recommended retention action and expected revenue recovery. For a retention team "
    "that wants a single worklist to execute against, rather than combining priority ranking "
    "and per-customer recommendations themselves. Returns an `{\"available\": false, ...}` "
    "shape if this module hasn't been validated for your tenant.",
)
def action_queue(
    limit: int = 50, current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    model_dir = model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    data_path = data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    clv_data_path = clv_data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    impact_df = compute_business_impact_bulk(model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path)
    top = impact_df.head(limit)

    queue = []
    for _, row in top.iterrows():
        recommendation = recommend_action_for_customer(
            row["customerID"], model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path
        )
        queue.append(
            {
                "customer_id": row["customerID"],
                "recommended_action": recommendation["recommended_action"],
                "expected_revenue_recovery": float(row["recoverable_revenue"]),
                "opportunity_score": float(row["opportunity_score"]),
            }
        )

    return {
        "action_queue": queue,
        "metadata": business_impact_metadata(model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path),
    }


@app.get(
    "/api/business-language-explanation/{customer_id}",
    tags=["Business Impact"],
    summary="Get a plain-language explanation of one customer's churn risk",
    description="Returns a plain-English (non-technical) explanation of why a customer is at "
    "risk of churning, translating the model's top features into business language. For a "
    "retention rep or account manager who isn't a data scientist and needs to understand (and "
    "explain to the customer) why they're flagged. Returns 404 if the customer doesn't exist.",
)
def business_language_explanation_endpoint(
    customer_id: str, current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "business_impact_core", TENANT_PROFILES, db=db)
    try:
        return business_language_explanation(
            customer_id,
            model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get(
    "/api/customer/{customer_id}/detail",
    tags=["Business Impact"],
    summary="Get one customer's full detail: risk, health score, and recommended action",
    description="Returns a single customer's churn probability with top contributing "
    "features, composite health score, and recommended retention action, combined into one "
    "response. For a system integration that wants everything about one customer in a single "
    "call rather than three separate ones (`/api/business-language-explanation`, "
    "`/api/health-score`, `/api/recommend`). Also available with API-key auth for server-to-"
    "server use. Returns 404 if the customer doesn't exist in this tenant's data, or an "
    "`{\"available\": false, ...}` shape if this module hasn't been validated for your tenant.",
)
def customer_detail(
    customer_id: str,
    current_user: CurrentUser = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db),
) -> Any:
    # db= is required here - without it, feature_enabled() inside
    # copilot_tools.get_customer_detail() only ever sees config.yaml's
    # static telco/banking profiles, never a self-registered tenant's real
    # Company-backed one (same missing-db= bug shape found and fixed
    # across several other endpoints in this same pass).
    result = copilot_tools.get_customer_detail(customer_id, current_user.tenant_id, db=db)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post(
    "/api/budget-optimizer",
    tags=["Budget Optimizer"],
    summary="Allocate a fixed retention budget across customers for maximum ROI",
    description="Given a total `budget_amount` and (optionally) a `cost_per_intervention`, "
    "greedily allocates the budget across customers to maximize recoverable revenue, "
    "optionally stopping once ROI turns negative (`stop_at_positive_roi`). For a manager who "
    "has a fixed retention budget and wants to know exactly which customers to spend it on. "
    "The per-intervention cost assumption is always labeled explicitly in the response - never "
    "silently assumed. Returns 400 if the budget amount is invalid.",
)
def budget_optimizer(
    request: BudgetOptimizerRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "budget_optimizer", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "budget_optimizer", TENANT_PROFILES, db=db)
    try:
        return optimize_budget(
            request.budget_amount,
            cost_per_intervention=request.cost_per_intervention,
            stop_at_positive_roi=request.stop_at_positive_roi,
            tenant_id=current_user.tenant_id,
            model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            clv_data_path=clv_data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        )
    except InvalidBudgetError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post(
    "/api/scenario",
    tags=["Scenarios"],
    summary="Run and save a portfolio-wide retention scenario",
    description="Re-scores your entire tenant's customer population under a hypothetical "
    "change (`scenario_type` + `params`, e.g. a discount offer applied portfolio-wide) and "
    "saves the result under `scenario_name` for later reference. For exploring 'what if we ran "
    "this retention campaign' before committing budget to it - every result is a real "
    "re-scoring, never a fabricated projection. Also available with API-key auth for system "
    "integrations that want to run scenarios programmatically. Returns 400 if `scenario_type` "
    "or `params` are invalid.",
)
def create_scenario(
    request: ScenarioRequest,
    current_user: CurrentUser = Depends(get_current_user_or_api_key),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "scenario_simulator", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "scenario_simulator", TENANT_PROFILES, db=db)
    try:
        result = run_scenario(
            request.scenario_type,
            request.params,
            request.scenario_name,
            tenant_id=current_user.tenant_id,
            model_dir=model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db),
            data_path=data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db),
        )
    except InvalidScenarioError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    scenario = Scenario(
        tenant_id=current_user.tenant_id,
        scenario_name=request.scenario_name,
        scenario_type=request.scenario_type,
        params_json=request.params,
        results_json=result,
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)

    return {"id": scenario.id, "created_at": scenario.created_at.isoformat(), **result}


@app.get(
    "/api/scenario/history",
    tags=["Scenarios"],
    summary="List previously run scenarios",
    description="Returns every previously saved scenario for your tenant, newest first, with "
    "a summary and headline revenue-at-risk delta for each. For reviewing what's already been "
    "explored before running a new scenario.",
)
def scenario_history(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    if not feature_enabled(current_user.tenant_id, "scenario_simulator", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "scenario_simulator", TENANT_PROFILES, db=db)
    rows = (
        get_tenant_scoped_query(db, Scenario, tenant_id=current_user.tenant_id)
        .order_by(Scenario.created_at.desc())
        .all()
    )
    return [
        {
            "id": row.id,
            "scenario_name": row.scenario_name,
            "scenario_type": row.scenario_type,
            "created_at": row.created_at.isoformat(),
            "summary": row.results_json.get("summary"),
            "headline_delta": (
                row.results_json.get("total_revenue_at_risk", {}).get("after", 0.0)
                - row.results_json.get("total_revenue_at_risk", {}).get("before", 0.0)
            ),
        }
        for row in rows
    ]


@app.get(
    "/api/scenario/{scenario_id}",
    tags=["Scenarios"],
    summary="Get one saved scenario's full detail",
    description="Returns the full saved result of one previously run scenario by id, "
    "including its parameters and complete before/after impact. Returns 404 if no scenario "
    "with that id exists for your tenant.",
)
def scenario_detail(
    scenario_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    if not feature_enabled(current_user.tenant_id, "scenario_simulator", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "scenario_simulator", TENANT_PROFILES, db=db)
    row = (
        get_tenant_scoped_query(db, Scenario, tenant_id=current_user.tenant_id)
        .filter(Scenario.id == scenario_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {
        "id": row.id,
        "scenario_name": row.scenario_name,
        "scenario_type": row.scenario_type,
        "params": row.params_json,
        "created_at": row.created_at.isoformat(),
        **row.results_json,
    }


VALID_TRACKING_STATUSES = ("not_started", "called", "emailed", "resolved")


def _tracking_to_dict(customer_id: str, row: CustomerTracking | None) -> dict[str, Any]:
    if row is None:
        return {
            "customer_id": customer_id,
            "assigned_manager": None,
            "status": "not_started",
            "call_scheduled_date": None,
            "updated_at": None,
        }
    return {
        "customer_id": row.customer_id,
        "assigned_manager": row.assigned_manager,
        "status": row.status,
        "call_scheduled_date": row.call_scheduled_date.isoformat() if row.call_scheduled_date else None,
        "updated_at": row.updated_at.isoformat(),
    }


@app.get(
    "/api/tracking/{customer_id}",
    tags=["Business Impact"],
    summary="Get one customer's manual operational tracking fields",
    description="Returns manually entered operational fields (assigned manager, outreach "
    "status, scheduled call date) for a customer - never AI-generated or model-derived, only "
    "ever whatever a human has typed in via the matching PUT endpoint. Returns a default "
    "'not_started' state (not a 404) if this customer has no tracking row yet.",
)
def get_tracking(
    customer_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Manual operational fields only - not AI-generated, never derived from
    any model. Returns a default not_started state if this customer has no
    tracking row yet, rather than 404ing (most customers won't have one)."""
    row = (
        get_tenant_scoped_query(db, CustomerTracking, tenant_id=current_user.tenant_id)
        .filter(CustomerTracking.customer_id == customer_id)
        .first()
    )
    return _tracking_to_dict(customer_id, row)


@app.put(
    "/api/tracking/{customer_id}",
    tags=["Business Impact"],
    summary="Update one customer's manual operational tracking fields",
    description="Sets the assigned manager, outreach status, and/or scheduled call date for a "
    "customer. For a retention rep logging real-world outreach progress against the model's "
    "recommendations. `status` must be one of `not_started`, `called`, `emailed`, `resolved` "
    "(400 otherwise); `call_scheduled_date` must be an ISO date/datetime string or null.",
)
def update_tracking(
    customer_id: str,
    request: TrackingUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    if request.status not in VALID_TRACKING_STATUSES:
        raise HTTPException(status_code=400, detail=f"status must be one of {VALID_TRACKING_STATUSES}")

    call_scheduled_date = None
    if request.call_scheduled_date:
        try:
            call_scheduled_date = datetime.fromisoformat(request.call_scheduled_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="call_scheduled_date must be an ISO date/datetime string")

    row = (
        get_tenant_scoped_query(db, CustomerTracking, tenant_id=current_user.tenant_id)
        .filter(CustomerTracking.customer_id == customer_id)
        .first()
    )
    if row is None:
        row = CustomerTracking(tenant_id=current_user.tenant_id, customer_id=customer_id)
        db.add(row)

    row.assigned_manager = request.assigned_manager
    row.status = request.status
    row.call_scheduled_date = call_scheduled_date
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(row)

    return _tracking_to_dict(customer_id, row)


@app.post(
    "/api/copilot/chat",
    tags=["Copilot"],
    summary="Ask the AI Copilot a question about your portfolio",
    description="Sends a natural-language question to the AI Copilot, which answers using "
    "only real tool calls against your tenant's actual data - it never fabricates a number. "
    "JWT-authenticated only (no API-key access), given the LLM cost and rate implications of "
    "opening this up to automated callers. Returns 503 if the Copilot is unavailable (e.g. no "
    "API key configured for the underlying model).",
)
def copilot_chat_endpoint(
    request: CopilotChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """tenant_id always comes from the verified JWT (current_user.tenant_id),
    never from the request body or anything the model outputs - see
    src/copilot/agent.py's module docstring for why that boundary matters.

    db is passed through to every tool call (see src/copilot/tools.py's
    module docstring) - without it, the Copilot's tools 100%-of-the-time
    report "unavailable" for a self-registered tenant regardless of their
    real feature_flags_json (same missing-db= bug shape found and fixed
    across several other endpoints in this same pass)."""
    try:
        result = copilot_chat(request.message, tenant_id=current_user.tenant_id, db=db)
    except CopilotUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Copilot is unavailable: {exc}")

    return result


@app.get(
    "/api/copilot/morning-brief",
    tags=["Copilot"],
    summary="Get a proactive AI-generated summary of today's top opportunities and alerts",
    description="Generates a fresh, unprompted brief on every call (no caching): top "
    "opportunities, the most severe active alert, and one illustrative retention scenario, "
    "using the same real-tool-calls-only guardrails as `/api/copilot/chat`. For showing a user "
    "something useful the moment they log in, without them having to ask a question first. "
    "JWT-authenticated only. Returns 503 if the Copilot is unavailable.",
)
def copilot_morning_brief_endpoint(
    current_user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> Any:
    """Proactive Copilot brief - no request body, generated fresh on every
    call (no caching). Same real-tool-calls-only guardrails as
    /api/copilot/chat, same 503-on-unavailable contract; see
    src/copilot/agent.py's generate_morning_brief()."""
    try:
        result = copilot_morning_brief(tenant_id=current_user.tenant_id, db=db)
    except CopilotUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Copilot is unavailable: {exc}")

    return result


@app.get(
    "/api/customer/{customer_id}/timeline",
    tags=["Customer Timeline"],
    summary="Get one customer's real event history",
    description="Returns a chronological timeline of real events for one customer: inferred "
    "signup/contract snapshot, logged predictions, logged recommendations, and any saved "
    "scenario that affected them. For an account manager reviewing everything that's actually "
    "happened around a customer. Deliberately excludes bulk-view side effects (e.g. loading "
    "the priority table does not create timeline events) - only genuinely explicit, single-"
    "customer actions are logged. Returns 404 if the customer doesn't exist, or an "
    "`{\"available\": false, ...}` shape if this module hasn't been validated for your tenant.",
)
def customer_timeline(
    customer_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Customer 360's real audit trail. Deliberately does NOT log (or show)
    an event for every bulk view of /api/priority, /api/business-impact,
    /api/action-queue, or /api/health-score/{id} - those recompute the whole
    test-set population (or read an in-memory cache of it) on effectively
    every dashboard page load, so logging one there would flood a customer's
    timeline with hundreds of fabricated-looking "events" that don't
    correspond to anyone actually doing anything for that customer. Only
    genuinely explicit single-customer actions - a what-if run
    (/api/whatif) or a recommendation lookup (/api/recommend/{id}) - are
    logged as real events, plus any saved scenario that happens to include
    this customer in its re-scored population.

    Also excludes the one-time database/migrate_csv_to_db.py bulk baseline
    Prediction row every customer already has (model_version == the bare
    model dir name, e.g. "v1") - that's a data-loading artifact from setup,
    not a system event that happened FOR this customer, and including it
    would mean literally every customer always shows as "active" instead of
    the thin/empty-state timeline that's the honest norm before anyone has
    actually looked at them.
    """
    # NOTE: db=db is required on every one of these three lookups below -
    # omitting it silently makes feature_enabled()/model_dir_for()/
    # data_path_for() only ever see config.yaml's static telco/banking
    # profiles (get_tenant_profile() falls through to a Company-backed
    # profile ONLY when a db session is passed - see src/tenant_registry.py's
    # module docstring) - i.e. this endpoint would 100%-of-the-time report
    # "unavailable" for every self-registered tenant regardless of their
    # real feature_flags_json, the exact missing-db= bug shape this project
    # has hit at least twice before elsewhere. Found and fixed as part of
    # generalizing this endpoint - was never actually reachable for a
    # self-registered tenant before this.
    if not feature_enabled(current_user.tenant_id, "customer_timeline", TENANT_PROFILES, db=db):
        return unavailable_response(current_user.tenant_id, "customer_timeline", TENANT_PROFILES, db=db)

    model_dir = model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    data_path = data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db)
    now = datetime.now(timezone.utc)
    try:
        events: list[dict[str, Any]] = timeline_load_customer_snapshot(
            customer_id, model_dir=model_dir, data_path=data_path, now=now
        )
    except TimelineCustomerNotFoundError:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")

    prediction_rows = (
        get_tenant_scoped_query(db, Prediction, tenant_id=current_user.tenant_id)
        .filter(Prediction.customer_id == customer_id)
        .filter(Prediction.model_version != model_dir.name)
        .order_by(Prediction.predicted_at)
        .all()
    )
    for row in prediction_rows:
        events.append(
            {
                "event_type": "prediction",
                "timestamp": row.predicted_at.isoformat(),
                "data_source": "real_system_event",
                "details": {"churn_probability": row.churn_probability, "model_version": row.model_version},
            }
        )

    recommendation_rows = (
        get_tenant_scoped_query(db, RecommendationLog, tenant_id=current_user.tenant_id)
        .filter(RecommendationLog.customer_id == customer_id)
        .order_by(RecommendationLog.created_at)
        .all()
    )
    for row in recommendation_rows:
        events.append(
            {
                "event_type": "recommendation",
                "timestamp": row.created_at.isoformat(),
                "data_source": "real_system_event",
                "details": {"recommended_action": row.recommended_action, "triggered_by": row.triggered_by_json},
            }
        )

    scenario_rows = (
        get_tenant_scoped_query(db, Scenario, tenant_id=current_user.tenant_id).order_by(Scenario.created_at).all()
    )
    for row in scenario_rows:
        impact = get_customer_scenario_impact(
            row.scenario_type, row.params_json, customer_id, model_dir=model_dir, data_path=data_path
        )
        if impact is None:
            continue
        events.append(
            {
                "event_type": "scenario",
                "timestamp": row.created_at.isoformat(),
                "data_source": "real_system_event",
                "details": {
                    "scenario_id": row.id,
                    "scenario_name": row.scenario_name,
                    "scenario_type": row.scenario_type,
                    "before_probability": impact["before_probability"],
                    "after_probability": impact["after_probability"],
                    "was_affected": impact["was_affected"],
                },
            }
        )

    events.sort(key=lambda event: event["timestamp"])
    return {"customer_id": customer_id, "events": events}


@app.get(
    "/customers",
    tags=["Admin/Cohort"],
    summary="List raw customer records for your tenant",
    description="Returns every customer record stored for your tenant (raw ingested features, "
    "as loaded at setup time). For an admin inspecting the underlying data your tenant's "
    "models were trained/scored against, not a business-facing view. Pass "
    "`recommend_eligible_only=true` to restrict the list to customers "
    "`/api/recommend/{customer_id}` (and `/api/explain/{customer_id}`) will actually accept - "
    "see that param's own note below for why the two don't otherwise match.",
)
def list_customers(
    tenant_id: str | None = None,
    recommend_eligible_only: bool = False,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    """List customers for the caller's own tenant.

    `tenant_id` is accepted as a query parameter but is never used - it
    exists only so a client attempting to request another tenant's data
    makes a well-formed request rather than hitting a validation error. The
    tenant scoping the query is always current_user.tenant_id, taken from
    the verified JWT, never from user input.

    `recommend_eligible_only`: explain_customer() (and, via it,
    recommend_action_for_customer()) only ever scores customers in this
    tenant's saved TEST split - by design, to avoid serving recommendations
    for customers the model was directly trained on - and raises "not in
    the saved test split" for the rest (typically ~80% of a tenant's full
    customer list, the size of the train split). A caller that offers a
    customer picker for recommendation lookups (Customer 360) should filter
    to this set rather than surface that error live, per option; leaves the
    unfiltered list (every other caller of this endpoint) untouched by
    default. Falls back to the unfiltered list if no model has been
    trained for this tenant yet, rather than erroring.
    """
    del tenant_id  # explicitly unused - tenant comes from the token only
    rows = get_tenant_scoped_query(db, Customer, tenant_id=current_user.tenant_id).all()
    customers = [
        {"customer_id": row.customer_id, "tenant_id": row.tenant_id, "raw_features": row.raw_features}
        for row in rows
    ]
    if recommend_eligible_only:
        model_dir = model_dir_for(current_user.tenant_id, TENANT_PROFILES, db=db)
        data_path = data_path_for(current_user.tenant_id, TENANT_PROFILES, db=db)
        if model_dir is not None and data_path is not None:
            try:
                eligible_ids = set(list_test_split_customer_ids(model_dir=model_dir, data_path=data_path))
                customers = [c for c in customers if c["customer_id"] in eligible_ids]
            except FileNotFoundError:
                pass  # no split_indices.json for this tenant yet - fall back to the unfiltered list
    return customers
