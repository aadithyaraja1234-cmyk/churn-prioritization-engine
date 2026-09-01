"""Copilot tool layer - Stage 1 of the AI Copilot subsystem.

Every function here is a THIN WRAPPER around an existing, already-tested
function from src/models/*.py - no new logic, no new data access, no new
assumptions. The Gemini agent (Stage 2) will call these; a human could
call them directly and get the exact same real data.

Each tool returns a dict that includes a "source" key naming the
underlying module:function(s) it wraps, so every answer the copilot gives
is traceable back to a concrete, already-validated computation - never a
number the LLM invented on its own.

Config-driven gating: which tenant gets which module (business_impact/
budget_optimizer/scenario/segment/alerts/survival) is read from
config.yaml's per-tenant feature_flags via src/tenant_registry.py, not
hardcoded to a tenant name here - see docs/ADDING_A_TENANT.md. Any
disabled feature returns the same {"available": False, "reason": ...}
shape, rather than raising or fabricating a result.

Every tool takes an optional `db` (SQLAlchemy Session) and threads it
through to feature_enabled()/model_dir_for()/etc. - without it, those
calls only ever see config.yaml's static telco/banking profiles
(src/tenant_registry.py's get_tenant_profile() falls through to a
Company-backed profile ONLY when a db session is passed), so every one of
these 8 tools would 100%-of-the-time report "unavailable" for a self-
registered tenant regardless of their real feature_flags_json - the same
missing-db= bug shape this project has hit and fixed across several API
endpoints. db defaults to None (not required) so a caller that genuinely
has no session - e.g. a quick script - still gets correct behavior for
Telco/Banking, the same graceful degradation every other db=None default
in this project already has; src/copilot/agent.py's chat()/
generate_morning_brief() (the only real callers, via api/main.py's
copilot endpoints) always pass a real one.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.models.alerts import scan_for_alerts
from src.models.budget_optimizer import InvalidBudgetError, optimize_budget
from src.models.business_impact import business_impact_metadata, compute_business_impact_bulk
from src.models.explain import explain_customer
from src.models.health_score import CustomerNotFoundError as HealthScoreCustomerNotFoundError, compute_health_score
from src.models.recommend import CustomerNotFoundError as RecommendCustomerNotFoundError, recommend_action_for_customer
from src.models.scenario import InvalidScenarioError, run_scenario as run_scenario_module
from src.models.segment import load_cluster_profiles
from src.models.survival import CustomerNotFoundError as SurvivalCustomerNotFoundError, churn_likelihood_for_customer
from src.tenant_registry import (
    clv_data_path_for,
    clv_model_dir_for,
    data_path_for,
    feature_enabled,
    model_dir_for,
    unavailable_response,
)


def _not_trained(tenant_id: str, feature: str, source: str, db: Session | None) -> dict[str, Any]:
    return {**unavailable_response(tenant_id, feature, db=db), "source": source}


def get_business_impact_summary(tenant_id: str, db: Session | None = None) -> dict[str, Any]:
    """Portfolio-level revenue-at-risk / recoverable-revenue summary.
    Wraps business_impact.py's compute_business_impact_bulk() +
    business_impact_metadata() - the same computation /api/business-impact
    uses, minus the full per-customer list (see get_top_opportunities() for
    that)."""
    source = "business_impact.py:compute_business_impact_bulk"
    if not feature_enabled(tenant_id, "business_impact_core", db=db):
        return _not_trained(tenant_id, "business_impact_core", source, db)

    model_dir = model_dir_for(tenant_id, db=db)
    data_path = data_path_for(tenant_id, db=db)
    impact_df = compute_business_impact_bulk(
        model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path_for(tenant_id, db=db)
    )
    needs_action_threshold = 0.3
    needs_action_count = int((impact_df["churn_probability"] >= needs_action_threshold).sum())
    return {
        "total_revenue_at_risk": float(impact_df["revenue_at_risk"].sum()),
        "total_recoverable_revenue": float(impact_df["recoverable_revenue"].sum()),
        "customers_needing_action_count": needs_action_count,
        "needs_action_churn_probability_threshold": needs_action_threshold,
        "metadata": business_impact_metadata(model_dir=model_dir, data_path=data_path),
        "source": source,
    }


def get_top_opportunities(tenant_id: str, limit: int = 10, db: Session | None = None) -> dict[str, Any]:
    """Top-N customers by opportunity_score. Wraps business_impact.py's
    compute_business_impact_bulk(), already sorted opportunity_score
    descending - the same ranking /api/business-impact and the Priority
    Table use."""
    source = "business_impact.py:compute_business_impact_bulk"
    if not feature_enabled(tenant_id, "business_impact_core", db=db):
        return _not_trained(tenant_id, "business_impact_core", source, db)

    impact_df = compute_business_impact_bulk(
        model_dir=model_dir_for(tenant_id, db=db),
        data_path=data_path_for(tenant_id, db=db),
        clv_data_path=clv_data_path_for(tenant_id, db=db),
    )
    customers = impact_df.head(limit)[
        ["customerID", "revenue_at_risk", "recoverable_revenue", "opportunity_score", "confidence"]
    ].to_dict(orient="records")
    return {"customers": customers, "source": source}


def get_customer_detail(customer_id: str, tenant_id: str, db: Session | None = None) -> dict[str, Any]:
    """Single-customer detail: explanation, health score, and recommended
    action. Wraps explain.py:explain_customer, health_score.py:
    compute_health_score, recommend.py:recommend_action_for_customer.

    CLV NOTE (read this before phrasing any answer that mentions CLV):
    there is no per-customer CLV DOLLAR prediction anywhere in this
    system - clv.py only exposes aggregate model metrics/feature
    importances, not a per-customer estimate. The closest per-customer CLV
    signal is health_score_components["clv_percentile_rank_component"]
    below, which is health_score.py's clv_component renamed here on
    purpose: it is a 0-100 PERCENTILE RANK of this customer's real CLTV
    among all customers (clv_df.rank(pct=True) * 100), NOT a dollar
    figure. Never report it as "this customer's lifetime value is $X" -
    that number does not exist in this tool's output.
    """
    source = "explain.py:explain_customer,health_score.py:compute_health_score,recommend.py:recommend_action_for_customer"
    if not feature_enabled(tenant_id, "business_impact_core", db=db):
        return _not_trained(tenant_id, "business_impact_core", source, db)

    model_dir = model_dir_for(tenant_id, db=db)
    data_path = data_path_for(tenant_id, db=db)
    clv_data_path = clv_data_path_for(tenant_id, db=db)

    try:
        explanation = explain_customer(customer_id, model_dir=model_dir, data_path=data_path)
    except (KeyError, ValueError) as exc:
        return {"error": str(exc), "source": source}

    try:
        health = compute_health_score(
            customer_id,
            model_dir=model_dir,
            data_path=data_path,
            clv_model_dir=clv_model_dir_for(tenant_id, db=db),
            clv_data_path=clv_data_path,
        )
    except HealthScoreCustomerNotFoundError as exc:
        return {"error": str(exc), "source": source}

    try:
        recommendation = recommend_action_for_customer(
            customer_id, model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path
        )
    except RecommendCustomerNotFoundError as exc:
        return {"error": str(exc), "source": source}

    # Renamed from health_score.py's "clv_component" - that name alone
    # invites an LLM to misread a 0-100 percentile rank as a dollar CLV
    # figure (see this function's docstring). Every other component name
    # is passed through unchanged; none of them carry the same
    # dollar-figure conflation risk. "clv_component" may not be present at
    # all for a tenant with no CLV data (health_score.py drops it from the
    # weighted average entirely rather than faking a value - see that
    # module's docstring) - only rename it when it's actually there.
    components = dict(health["components"])
    if "clv_component" in components:
        components["clv_percentile_rank_component"] = components.pop("clv_component")

    return {
        "customer_id": customer_id,
        "churn_probability": explanation["churn_probability"],
        "top_features": explanation["top_features"],
        "health_score": health["health_score"],
        "health_score_components": components,
        "recommended_action": recommendation["recommended_action"],
        "triggered_by": recommendation["triggered_by"],
        "source": source,
    }


def run_budget_optimization(
    tenant_id: str, budget: float, cost_override: float | None = None, db: Session | None = None
) -> dict[str, Any]:
    """Wraps budget_optimizer.py's optimize_budget() verbatim - same
    greedy allocation, same labeled cost assumption, same Hillstrom
    citation."""
    source = "budget_optimizer.py:optimize_budget"
    if not feature_enabled(tenant_id, "budget_optimizer", db=db):
        return _not_trained(tenant_id, "budget_optimizer", source, db)

    try:
        result = optimize_budget(
            budget_amount=budget,
            cost_per_intervention=cost_override,
            tenant_id=tenant_id,
            model_dir=model_dir_for(tenant_id, db=db),
            data_path=data_path_for(tenant_id, db=db),
            clv_data_path=clv_data_path_for(tenant_id, db=db),
        )
    except InvalidBudgetError as exc:
        return {"error": str(exc), "source": source}

    return {**result, "source": source}


def run_scenario(tenant_id: str, scenario_type: str, params: dict[str, Any], db: Session | None = None) -> dict[str, Any]:
    """Wraps scenario.py's run_scenario() verbatim - re-scores the real
    test-set population under the hypothetical change. Does not persist to
    the scenario history table (that's a main.py/DB-layer concern for the
    Scenario Simulator UI, not part of the underlying scenario logic)."""
    source = "scenario.py:run_scenario"
    if not feature_enabled(tenant_id, "scenario_simulator", db=db):
        return _not_trained(tenant_id, "scenario_simulator", source, db)

    try:
        result = run_scenario_module(
            scenario_type,
            params,
            scenario_name=f"copilot: {scenario_type}",
            tenant_id=tenant_id,
            model_dir=model_dir_for(tenant_id, db=db),
            data_path=data_path_for(tenant_id, db=db),
        )
    except InvalidScenarioError as exc:
        return {"error": str(exc), "source": source}

    return {**result, "source": source}


def get_segment_summary(tenant_id: str, db: Session | None = None) -> dict[str, Any]:
    """Wraps segment.py's load_cluster_profiles() - the real, already-
    fitted K-Means clusters and their labels (same data /api/segments
    returns)."""
    source = "segment.py:load_cluster_profiles"
    if not feature_enabled(tenant_id, "segments", db=db):
        return _not_trained(tenant_id, "segments", source, db)

    profiles = load_cluster_profiles(model_dir=model_dir_for(tenant_id, db=db), data_path=data_path_for(tenant_id, db=db))
    return {"segments": profiles.to_dict(orient="records"), "source": source}


def get_alerts(tenant_id: str, db: Session | None = None) -> dict[str, Any]:
    """Wraps alerts.py's scan_for_alerts() - the real, already-computed
    per-customer and portfolio-level alerts (same data /api/alerts
    returns)."""
    source = "alerts.py:scan_for_alerts"
    if not feature_enabled(tenant_id, "business_impact_core", db=db):
        return _not_trained(tenant_id, "business_impact_core", source, db)

    alerts = scan_for_alerts(
        model_dir=model_dir_for(tenant_id, db=db),
        data_path=data_path_for(tenant_id, db=db),
        clv_data_path=clv_data_path_for(tenant_id, db=db),
    )
    return {"alerts": alerts, "source": source}


def get_survival_likelihood(customer_id: str, tenant_id: str, db: Session | None = None) -> dict[str, Any]:
    """Wraps survival.py's churn_likelihood_for_customer() (built directly
    on top of churn_likelihood_within_window()'s real Cox PH survival-
    function output - see that module for the conditional_after math) -
    same data /api/survival/likelihood/{customer_id} returns."""
    source = "survival.py:churn_likelihood_for_customer"
    if not feature_enabled(tenant_id, "survival", db=db):
        return _not_trained(tenant_id, "survival", source, db)

    try:
        likelihood = churn_likelihood_for_customer(
            customer_id, model_dir=model_dir_for(tenant_id, db=db), data_path=data_path_for(tenant_id, db=db)
        )
    except SurvivalCustomerNotFoundError as exc:
        return {"error": str(exc), "source": source}

    return {"customer_id": customer_id, "churn_likelihood": likelihood, "source": source}
