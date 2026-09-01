"""Retention Budget Optimizer - greedy allocation of a fixed retention
budget across customers. No new modeling, no retraining: reuses
compute_business_impact_bulk()'s already-computed opportunity_score and
recoverable_revenue (which itself already bakes in the labeled
INTERVENTION_SUCCESS_RATE assumption) and recommend.py's existing
recommender. This module only ranks and greedily selects.

Every number here is either:
  (a) a direct computation from real, validated outputs (opportunity_score,
      recoverable_revenue, churn_probability, all reused as-is), or
  (b) an explicitly-labeled assumption (DEFAULT_COST_PER_INTERVENTION,
      and by extension INTERVENTION_SUCCESS_RATE baked into
      recoverable_revenue) - documented here AND surfaced in the API
      response metadata, same honesty pattern as business_impact.py.

Generalized to accept any tenant's model_dir/data_path/clv_data_path (same
pattern as every other module this project's generalization pass touched).
This module never actually needed its own tenant_config threading: it only
ever calls compute_business_impact_bulk()/recommend_action_for_customer(), both
already fully tenant-generic, so once those two were generalized this
module was mechanically tenant-generic too - the real, separate bug was
api/main.py's /api/budget-optimizer endpoint never passing db= to
feature_enabled()/model_dir_for()/etc. (same missing-db= bug shape found
and fixed across several other endpoints), which meant this endpoint
100%-of-the-time reported "unavailable" for every self-registered tenant
regardless of their real feature_flags_json - and "budget_optimizer" was
never wired into src/models/tenant_training.py's auto-enable gating at
all, so no self-registered tenant ever had the flag set to try anyway.

SANITY GATE: budget_optimizer_sanity_check() below - structural, not
statistical (same category as timeline.py's/scenario.py's own gates,
neither of which impose a numeric threshold this module has no equivalent
of): this module's only real dependency is compute_business_impact_bulk()/
recommend_action_for_customer(), both already gated by business_impact_core
passing - there's no NEW statistical risk this module introduces on top of
that. The gate exists to catch a different, real risk: running the actual
allocation end-to-end and confirming it selects a real, non-empty set of
customers for a small, always-affordable probe budget, rather than trusting
"business_impact_core passed" to imply "the allocation loop itself runs
cleanly for this tenant's population."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.business_impact import business_impact_metadata, compute_business_impact_bulk
from src.models.recommend import recommend_action_for_customer

# Assumed average intervention cost - no real cost data exists in this
# system; adjust based on actual campaign costs if known. Same honesty
# category as INTERVENTION_SUCCESS_RATE: a clearly labeled placeholder, not
# a measured fact.
DEFAULT_COST_PER_INTERVENTION = 75.0

# See module docstring's SANITY GATE note. Small enough that every tenant's
# test population (the smallest real one seen so far, Aurora's, has 800
# customers) can always afford this many interventions at the default
# per-intervention cost, so a real shortfall here means something is
# actually broken, not just a small population.
SANITY_CHECK_PROBE_N_CUSTOMERS = 10


class InvalidBudgetError(ValueError):
    pass


def optimize_budget(
    budget_amount: float,
    cost_per_intervention: float | None = None,
    stop_at_positive_roi: bool = False,
    tenant_id: str = "telco",
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    clv_data_path: str | Path = "data/raw/telco_enriched.csv",
) -> dict[str, Any]:
    """Rank all customers by opportunity_score descending (reused, not
    recomputed) and greedily fund one intervention per customer,
    highest-opportunity first, until budget_amount is exhausted.

    Greedy-by-opportunity_score is optimal for this specific problem shape
    (every selected customer costs the same fixed amount, so maximizing
    total recoverable_revenue under a budget cap reduces to taking the
    highest-value items first - no knapsack tradeoff since there's no
    per-customer cost variation to trade off against).

    stop_at_positive_roi: if True, skip any customer whose recoverable_revenue
    doesn't cover its own cost_per_intervention, rather than counting them
    towards the budget - selection continues down the opportunity-ranked
    list until either the budget is exhausted or no remaining candidate
    clears the cost bar. Since the ranking is by opportunity_score, not by
    recoverable_revenue, an uneconomical customer can appear ahead of many
    economical ones; skipping (rather than halting on) the first failure
    avoids one early low-value customer zeroing out an otherwise-healthy
    result, even though budget may go unspent if the qualifying pool runs
    out before the budget does.
    """
    del tenant_id  # not used for any column/data-access decision - model_dir/data_path/clv_data_path carry all of that now

    if budget_amount is None or budget_amount <= 0:
        raise InvalidBudgetError("budget_amount must be a positive number")

    cost_is_default = cost_per_intervention is None
    cost = DEFAULT_COST_PER_INTERVENTION if cost_is_default else cost_per_intervention
    if cost <= 0:
        raise InvalidBudgetError("cost_per_intervention must be a positive number")

    impact_df = compute_business_impact_bulk(model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path)
    total_at_risk_revenue = float(impact_df["revenue_at_risk"].sum())

    n_affordable = max(int(budget_amount // cost), 0)
    candidates = impact_df[impact_df["recoverable_revenue"] > cost] if stop_at_positive_roi else impact_df
    selected_df = candidates.head(n_affordable)

    selected_customers = []
    for _, row in selected_df.iterrows():
        recommendation = recommend_action_for_customer(
            row["customerID"], model_dir=model_dir, data_path=data_path, clv_data_path=clv_data_path
        )
        selected_customers.append(
            {
                "customer_id": row["customerID"],
                "opportunity_score": float(row["opportunity_score"]),
                "recoverable_revenue": float(row["recoverable_revenue"]),
                "churn_probability": float(row["churn_probability"]),
                "recommended_action": recommendation["recommended_action"],
            }
        )

    n_customers_covered = len(selected_customers)
    total_cost = n_customers_covered * cost
    total_recoverable_revenue = float(selected_df["recoverable_revenue"].sum())
    roi = (total_recoverable_revenue - total_cost) / total_cost if total_cost > 0 else 0.0
    pct_of_at_risk_revenue_covered = (
        float(selected_df["revenue_at_risk"].sum()) / total_at_risk_revenue if total_at_risk_revenue else 0.0
    )
    unspent_budget = float(budget_amount) - float(total_cost)
    # "Fully utilized" = every affordable slot was filled. In
    # stop_at_positive_roi mode, running out of qualifying (cost-effective)
    # candidates before the budget cap means the remainder was deliberately
    # left unspent, not that we ran out of budget.
    budget_fully_utilized = (not stop_at_positive_roi) or (n_customers_covered == n_affordable)

    metadata = business_impact_metadata(model_dir=model_dir, data_path=data_path)
    metadata["cost_per_intervention_note"] = (
        "Assumed average intervention cost — no real cost data exists in this system; "
        "adjust based on actual campaign costs if known."
    )

    return {
        "budget_amount": float(budget_amount),
        "cost_per_intervention": float(cost),
        "cost_per_intervention_is_default": cost_is_default,
        "stop_at_positive_roi": stop_at_positive_roi,
        "selected_customers": selected_customers,
        "n_customers_covered": n_customers_covered,
        "total_cost": float(total_cost),
        "total_recoverable_revenue": total_recoverable_revenue,
        "roi": float(roi),
        "pct_of_at_risk_revenue_covered": float(pct_of_at_risk_revenue_covered),
        "budget_fully_utilized": budget_fully_utilized,
        "unspent_budget": unspent_budget,
        "metadata": metadata,
    }


def budget_optimizer_sanity_check(
    model_dir: str | Path,
    data_path: str | Path,
    clv_data_path: str | Path | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Returns (failure_reason_or_None, metrics) - see module docstring's
    SANITY GATE note. Runs a real, small, always-affordable probe
    allocation (SANITY_CHECK_PROBE_N_CUSTOMERS customers at the default
    per-intervention cost) against this tenant's own real data and confirms
    it selects that many customers without crashing."""
    probe_budget = DEFAULT_COST_PER_INTERVENTION * SANITY_CHECK_PROBE_N_CUSTOMERS
    try:
        result = optimize_budget(
            probe_budget,
            tenant_id="_sanity_check_probe",
            model_dir=model_dir,
            data_path=data_path,
            clv_data_path=clv_data_path,
        )
    except Exception as exc:  # noqa: BLE001 - report as a gate failure, not a crash
        return f"probe allocation ({SANITY_CHECK_PROBE_N_CUSTOMERS} customers at the default cost) failed to run: {exc}", {}

    metrics = {
        "probe_n_customers_covered": result["n_customers_covered"],
        "probe_total_recoverable_revenue": result["total_recoverable_revenue"],
    }
    if result["n_customers_covered"] < SANITY_CHECK_PROBE_N_CUSTOMERS:
        return (
            f"probe allocation only covered {result['n_customers_covered']} of "
            f"{SANITY_CHECK_PROBE_N_CUSTOMERS} customers it should always be able to afford - this "
            "tenant's test population is smaller than this gate expects, or the underlying "
            "business-impact computation returned fewer real candidates than requested.",
            metrics,
        )
    return None, metrics
