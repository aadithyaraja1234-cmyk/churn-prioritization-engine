"""Business Impact Layer — translates already-validated model outputs into
revenue-facing numbers. No new modeling, no retraining: every function here
is a pure arithmetic transform of numbers already produced elsewhere
(get_priority_ranking()'s churn_probability/revenue_at_risk, the real CLTV
column's percentile rank, and survival.py's already-fitted Cox PH model).

Every number here is either:
  (a) a direct computation from real, validated outputs (confidence,
      ease_of_saving, and the underlying revenue_at_risk/churn_probability
      themselves), or
  (b) an explicitly-labeled assumption/heuristic (INTERVENTION_SUCCESS_RATE,
      ease_of_saving's boundary-distance proxy, the inestimable-contract
      survival fallback) - documented here AND surfaced in the API response
      metadata, never silently baked into a number with no explanation.

recoverable_revenue reflects expected revenue over the customer's remaining
tenure (estimated via survival analysis by contract type), not a single
month - this was corrected from an earlier single-month calculation that
significantly understated retention value (a customer retained today keeps
paying for their whole remaining relationship, not just the next 30 days).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_tenant_config
from src.models.explain import explain_customer
from src.models.intervention_benchmark import compute_hillstrom_benchmark
from src.models.prioritize import get_priority_ranking
from src.models.survival import median_survival_by_contract, resolve_segment_feature_column

_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}

# Conservative placeholder for contract segments where the Cox model's
# predict_median() returns inf (the segment's survival curve never drops
# below 0.5 within the observed data range - e.g. Two year contracts churn
# so rarely that no true median is observable). Not a measured or fitted
# value - a documented cap standing in for "we can't estimate this, so
# don't let it default to an unbounded/undefined remaining tenure."
FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT = 36.0

# Every customer has *some* future, even one the survival model estimates
# as already "overdue" to churn (median total tenure < current tenure) -
# 1 month keeps recoverable_revenue from going to zero or negative for
# those customers instead of just clamping to a small positive floor.
MIN_REMAINING_TENURE_MONTHS = 1.0

# Maps every real Telco feature column to a plain-language business
# description, so business-language-explanation never leaks a raw column
# name. Covers all 19 classifier feature columns.
DRIVER_BUSINESS_LABELS: dict[str, str] = {
    "Contract": "their contract type",
    "tenure": "how long they've been a customer",
    "MonthlyCharges": "their monthly bill amount",
    "TotalCharges": "their total spend with us",
    "OnlineSecurity": "whether they have online security",
    "OnlineBackup": "whether they have online backup",
    "DeviceProtection": "whether they have device protection",
    "TechSupport": "whether they have tech support",
    "InternetService": "their internet service type",
    "PaymentMethod": "how they pay their bill",
    "PaperlessBilling": "their paperless billing preference",
    "SeniorCitizen": "senior citizen status",
    "Partner": "having a partner",
    "Dependents": "having dependents",
    "PhoneService": "having phone service",
    "MultipleLines": "having multiple phone lines",
    "StreamingTV": "TV streaming usage",
    "StreamingMovies": "movie streaming usage",
    "gender": "gender",
}

# Assumed value, not adjusted to match Hillstrom directly. Industry-benchmark
# assumption, not measured from this system's own outcome data - no real
# intervention-outcome data exists yet to derive this empirically (same
# honesty category as the Phase 5 feedback loop and the health-score/recommend
# heuristics). Retention-offer success rates in telecom churn literature
# commonly cluster around 20-40%; 30% is a reasonable illustrative midpoint,
# not a fitted or measured value.
#
# For reference, the Hillstrom email-marketing RCT (2008) showed an absolute
# lift of ~0.5 percentage points (0.6% base rate) and a relative uplift of
# ~54-119% (~86% combined) - see src/models/intervention_benchmark.py. Cited
# as the closest available real-world benchmark, not as validation of this
# number, since the domains (impulse retail purchase vs. proactive churn
# retention) and base rates differ substantially. See
# business_impact_metadata()'s "hillstrom_benchmark" key for the real figures,
# reported alongside this assumption rather than merged into it.
INTERVENTION_SUCCESS_RATE = 0.30


def recoverable_revenue(revenue_at_risk: float, expected_remaining_tenure_months: float) -> float:
    """Expected recoverable revenue if an intervention is attempted, over the
    customer's expected remaining tenure - not a single month.

    revenue_at_risk (churn_probability * MonthlyCharges) is inherently a
    one-month figure. Multiplying it only by INTERVENTION_SUCCESS_RATE, as
    this function used to do, silently treated "recoverable revenue" as one
    month of saved billing, which understates what a successful retention
    is actually worth (the customer keeps paying for their whole remaining
    relationship, not just the next 30 days). expected_remaining_tenure_months
    corrects for this - see expected_remaining_tenure_months() below.

    ASSUMPTION: multiplies by INTERVENTION_SUCCESS_RATE (0.30), an
    industry-benchmark placeholder, not a measured success rate.
    """
    return revenue_at_risk * INTERVENTION_SUCCESS_RATE * expected_remaining_tenure_months


def expected_remaining_tenure_months(
    contract: str, tenure_months: float, median_survival_by_contract_map: dict[str, float | None]
) -> float:
    """Expected number of additional months this customer keeps paying,
    estimated as (this contract type's median total survival time) minus
    (their tenure so far), floored at MIN_REMAINING_TENURE_MONTHS.

    median_survival_by_contract_map comes from survival.py's already-fitted
    Cox PH model (src/models/survival.py's median_survival_by_contract()) -
    no new model, no refitting. Some contract segments (observed: "Two year")
    churn so rarely that the model can't produce a true median within the
    observed data range and reports None for them; those fall back to
    FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT, a documented
    placeholder, not a measured or fitted value.
    """
    median_total_tenure = median_survival_by_contract_map.get(contract)
    if median_total_tenure is None:
        median_total_tenure = FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT
    return max(median_total_tenure - tenure_months, MIN_REMAINING_TENURE_MONTHS)


def confidence(churn_probability: float) -> float:
    """Distance from the model's 0.5 decision boundary, normalized 0-1.

    A real property of the model's own output (not invented): 1.0 means the
    model is maximally certain (probability near 0 or 1), 0.0 means it's
    right at the boundary (probability near 0.5).
    """
    return abs(churn_probability - 0.5) * 2.0


def ease_of_saving(churn_probability: float) -> float:
    """HEURISTIC PROXY, not a measured quantity: customers near the 0.5
    decision boundary are assumed more movable by an intervention than
    those the model is already near-certain will stay or leave. This is
    the inverse of confidence() by construction (1 - confidence), not an
    independently derived signal.
    """
    return 1.0 - confidence(churn_probability)


def opportunity_score(
    revenue_at_risk: float, ease_of_saving_value: float, clv_percentile_weight: float | None
) -> float:
    """Composite ranking score: bigger, more-movable, more-valuable
    customers rank higher. clv_percentile_weight is the customer's CLTV
    percentile rank normalized to 0-1 (reused from the real CLTV column,
    same convention as health_score.py/alerts.py).

    clv_percentile_weight is None when this tenant has no CLV data at all
    (clv_data_path=None) - the CLV term is dropped from the product
    entirely rather than standing in a 0.0 weight, which would zero out
    the WHOLE score for every customer alike and make the resulting
    "ranking" arbitrary (all zeros, tie-broken by whatever order pandas
    happened to preserve) rather than a real, if reduced-signal, ordering
    by revenue_at_risk * ease_of_saving. A customer missing from an
    otherwise-present CLV file for THIS tenant is a different, genuine
    per-customer data gap (see compute_business_impact_bulk) - that case
    still multiplies by 0.0, since dropping the term for only some
    customers would make cross-customer scores incomparable."""
    if clv_percentile_weight is None:
        return revenue_at_risk * ease_of_saving_value
    return revenue_at_risk * ease_of_saving_value * clv_percentile_weight


def compute_business_impact_bulk(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    clv_data_path: str | Path | None = "data/raw/telco_enriched.csv",
    tenant_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Business impact numbers for every test-set customer, sorted by
    opportunity_score descending. Vectorized, no per-customer model
    inference beyond what get_priority_ranking() already computed - but
    get_priority_ranking() itself re-scores the whole test set (~2s), so
    the result is cached in-memory per (model_dir, data_path, clv_data_path)
    since multiple endpoints (business-impact, action-queue) reuse this.

    clv_data_path may be None, and model_dir may not have a
    survival_model.pkl - both true for a tenant that's only had the core
    classifier trained (business_impact_core doesn't require CLV or
    survival to have been separately trained and validated; see
    config.yaml's tenants: docstring). Neither degrades the numbers for a
    tenant that DOES have them (telco/banking always will) - this only
    changes behavior for a tenant missing one or both.

    Generalized to accept tenant_config (same pattern as prioritize.py -
    read back from model_dir/split_indices.json via
    src/data/split.py's load_tenant_config() when omitted). Two column
    roles beyond id/revenue need special handling since neither is
    guaranteed to exist for a self-registered tenant the way Telco's
    "Contract"/"tenure" always do:
      - duration_column (tenure-equivalent): if this tenant never mapped
        one (no "duration" onboarding role chosen), there is no real
        per-customer tenure value to offset the survival estimate by -
        every customer's expected_remaining_tenure_months falls back to
        FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT uniformly
        (equivalent to the existing no-survival-model fallback below,
        just for a different reason), rather than crashing on a missing
        column. Surfaced via business_impact_metadata()'s
        "expected_remaining_tenure_note" when this applies.
      - segment_feature_column: not a real onboarding role at all (see
        survival.py's docstring - it's auto-derived from the fitted Cox
        model's own hazard ratios and saved into survival_model.pkl's own
        tenant_config) - read from THAT pickle when it exists, not from
        the classifier's tenant_config, since that's the only place the
        resolved value actually lives.
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)
    clv_data_path = Path(clv_data_path) if clv_data_path is not None else None

    cache_key = (str(model_dir), str(data_path), str(clv_data_path))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    duration_column = tenant_config.get("duration_column")
    clv_column = tenant_config.get("clv_column")

    ranking = get_priority_ranking(
        strategy="revenue_weighted", model_dir=model_dir, data_path=data_path, tenant_config=tenant_config
    )

    has_clv = clv_data_path is not None and clv_column is not None
    if has_clv:
        clv_df = pd.read_csv(clv_data_path)
        clv_by_id = clv_df.set_index(id_column)[clv_column]
        clv_percentile_by_id = clv_by_id.rank(pct=True)  # already 0-1, used directly as the weight
    else:
        # No CLV data for this tenant at all - clv_percentile_weight below
        # is set to None (not 0.0) for every customer, so opportunity_score()
        # drops the CLV term from the product instead of zeroing the whole
        # score out uniformly.
        clv_percentile_by_id = pd.Series(dtype=float)

    has_duration = duration_column is not None
    segment_feature_column = resolve_segment_feature_column(model_dir)
    if (model_dir / "survival_model.pkl").exists():
        median_survival = median_survival_by_contract(model_dir=model_dir, data_path=data_path)
    else:
        # No survival model trained for this tenant - expected_remaining_
        # tenure_months() below already falls back to
        # FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT for any
        # contract type missing from this map, so an empty map reuses that
        # exact same, already-documented fallback for every contract
        # instead of crashing. (segment_feature_column is already None in
        # this branch too - resolve_segment_feature_column() returns None
        # for exactly the same "no survival_model.pkl" condition.)
        median_survival = {}

    raw_df = clean_data(load_raw(data_path, tenant_config), tenant_config).set_index(id_column)
    select_columns: dict[str, str] = {}  # canonical name -> this tenant's real column name
    if segment_feature_column and segment_feature_column in raw_df.columns:
        select_columns["Contract"] = segment_feature_column
    if has_duration and duration_column in raw_df.columns:
        select_columns["tenure"] = duration_column
    contract_and_tenure_by_id = raw_df[list(select_columns.values())].rename(
        columns={v: k for k, v in select_columns.items()}
    )
    if "Contract" not in contract_and_tenure_by_id.columns:
        # No resolved segment column available (no survival model, or one
        # exists but somehow has no categorical covariate) - a placeholder
        # constant value for every row. Only ever read via median_survival's
        # dict lookup below, which is {} in exactly this situation, so the
        # placeholder's actual value never affects the result - it only
        # exists so the row-wise apply() has a value to pass.
        contract_and_tenure_by_id["Contract"] = "unknown"
    if "tenure" not in contract_and_tenure_by_id.columns:
        # No duration_column mapped for this tenant at all - 0.0 for every
        # customer, which (combined with median_survival being real or {})
        # makes expected_remaining_tenure_months() fall back to a flat
        # per-tenant constant instead of crashing on a missing column. See
        # this function's own docstring.
        contract_and_tenure_by_id["tenure"] = 0.0

    result = ranking[["customerID", "churn_probability", "expected_revenue_at_risk"]].copy()
    result = result.rename(columns={"expected_revenue_at_risk": "revenue_at_risk"})
    result = result.join(contract_and_tenure_by_id, on="customerID")
    result["expected_remaining_tenure_months"] = result.apply(
        lambda row: expected_remaining_tenure_months(row["Contract"], row["tenure"], median_survival), axis=1
    )
    result["recoverable_revenue"] = result.apply(
        lambda row: recoverable_revenue(row["revenue_at_risk"], row["expected_remaining_tenure_months"]), axis=1
    )
    result["confidence"] = result["churn_probability"].apply(confidence)
    result["ease_of_saving"] = result["churn_probability"].apply(ease_of_saving)
    if has_clv:
        # A customer missing from an otherwise-present CLV file is a real
        # per-customer data gap - fall back to a 0.0 weight for THAT
        # customer only (see opportunity_score()'s docstring for why this
        # differs from the no-CLV-at-all case below).
        result["clv_percentile_weight"] = result["customerID"].map(clv_percentile_by_id).fillna(0.0)
    else:
        # None (not 0.0) for every customer - opportunity_score() reads
        # this as "drop the CLV term", not "CLV weight of zero".
        result["clv_percentile_weight"] = None
    result["opportunity_score"] = result.apply(
        lambda row: opportunity_score(row["revenue_at_risk"], row["ease_of_saving"], row["clv_percentile_weight"]),
        axis=1,
    )

    result = result.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    _CACHE[cache_key] = result
    return result


def business_impact_metadata(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    clv_data_path: str | Path | None = "data/raw/telco_enriched.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assumption disclosure block - always attached to API responses that
    use recoverable_revenue/opportunity_score, so a labeled assumption is
    never presented as a measured fact.

    "hillstrom_benchmark" is reported alongside "intervention_success_rate"
    as separate, real, cited external evidence - NOT merged into it or used
    to justify/derive the 30% figure. See intervention_benchmark.py's module
    docstring for why the two measure different quantities.

    "median_survival_months_by_contract" is the real, already-fitted Cox PH
    figures used to compute expected_remaining_tenure_months - reported so
    the remaining-tenure multiplier is inspectable, not just asserted.

    "opportunity_score_note" is only present when clv_data_path is None -
    see opportunity_score()'s docstring for why the CLV term is dropped
    from the formula entirely for this tenant, rather than reported as a
    normal, full-signal ranking.

    "expected_remaining_tenure_note" is only present when this tenant has
    no duration_column mapped at all (distinct from just missing a
    survival model - see compute_business_impact_bulk()'s docstring) -
    every customer gets the same flat fallback remaining-tenure estimate
    since there's no real per-customer tenure to offset it by.
    """
    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    metadata = {
        "intervention_success_rate": INTERVENTION_SUCCESS_RATE,
        "intervention_success_rate_note": (
            "Industry-benchmark assumption, not measured from this system's own outcome data — "
            "no real intervention-outcome data exists yet to derive this empirically. Assumed value, "
            "not adjusted to match the Hillstrom benchmark below directly."
        ),
        "ease_of_saving_note": (
            "Heuristic proxy: customers near the model's 0.5 decision boundary are assumed to be "
            "more movable than those already near-certain to stay or leave. Not an independently "
            "measured quantity."
        ),
        "recoverable_revenue_note": (
            "Recoverable revenue reflects expected revenue over the customer's remaining tenure "
            "(estimated via survival analysis by contract type), not a single month — this was "
            "corrected from an earlier single-month calculation that significantly understated "
            "retention value."
        ),
        "median_survival_months_by_contract": (
            median_survival_by_contract(model_dir=model_dir, data_path=data_path)
            if (Path(model_dir) / "survival_model.pkl").exists()
            else {}  # no survival model trained for this tenant - see compute_business_impact_bulk()
        ),
        "median_survival_fallback_months_for_inestimable_contract": (
            FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT
        ),
        "hillstrom_benchmark": compute_hillstrom_benchmark(),
    }
    if clv_data_path is None:
        metadata["opportunity_score_note"] = (
            "computed without CLV weighting; CLV model not yet trained for this tenant."
        )
    if tenant_config.get("duration_column") is None:
        metadata["expected_remaining_tenure_note"] = (
            "computed using a flat fallback remaining-tenure estimate for every customer; this "
            "tenant has no tenure/duration-equivalent column mapped, so there's no real "
            "per-customer value to offset the estimate by."
        )
    return metadata


def to_business_language(feature_name: str) -> str:
    return DRIVER_BUSINESS_LABELS.get(feature_name, feature_name)


def business_language_explanation(
    customer_id: str,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
) -> dict[str, Any]:
    """Templates explain_customer()'s top drivers into a plain-English
    sentence - reuses the existing explainability output, no new modeling."""
    explanation = explain_customer(customer_id, model_dir=model_dir, data_path=data_path)
    contributions = explanation["top_features"]

    increasing = [c for c in contributions if c["direction"] == "up"]
    decreasing = [c for c in contributions if c["direction"] == "down"]

    if increasing:
        labels = [to_business_language(c["feature"]) for c in increasing[:2]]
        if len(labels) == 1:
            sentence = f"Risk is elevated primarily because of {labels[0]}."
        else:
            sentence = f"Risk is elevated primarily because of {labels[0]} and {labels[1]}."
        if decreasing:
            sentence += f" This is partially offset by {to_business_language(decreasing[0]['feature'])}."
    elif decreasing:
        offsetting = to_business_language(decreasing[0]["feature"])
        sentence = f"No single factor strongly elevates risk; {offsetting} is helping keep it low."
    else:
        sentence = "No dominant factor identified for this customer."

    return {
        "customer_id": customer_id,
        "explanation": sentence,
        "churn_probability": explanation["churn_probability"],
    }
