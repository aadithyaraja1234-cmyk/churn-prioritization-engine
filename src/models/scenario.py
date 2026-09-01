"""Portfolio-level scenario simulator.

Re-scores the same test-set population used everywhere else in this app
(prioritize.py / business_impact.py) under a hypothetical portfolio-wide
change, using the already-trained classifier - no retraining, no new model.

Four scenario types are supported:
  - "uniform_charge_change": shift MonthlyCharges by a percentage for every
    customer.
  - "contract_migration": move a random fraction of customers off one
    contract type onto another (fixed seed, reproducible).
  - "discount_offer": reduce MonthlyCharges by a percentage for one segment
    (or everyone, if target_segment is null).
  - "loyalty_program": a random fraction of customers (fixed seed) get a
    flat, clearly-labeled-as-illustrative churn_probability reduction - not
    a feature change, since this system has no loyalty-program feature or
    outcome data to model a real effect from (see LOYALTY_PROGRAM_REDUCTION_
    FACTOR below and its metadata note in run_scenario()).

A "support staffing" scenario type was considered and deliberately NOT
implemented: this system has no support-ticket or staffing-related feature
in its data, so any staffing effect would be fabricated, not modeled -
that would violate this project's no-fabricated-numbers discipline. If
asked, the honest answer is "not implementable without a support-
interaction dataset."

Segment membership (per_segment_breakdown's grouping key) is each
customer's ORIGINAL K-Means cluster, computed from their unmodified
features. Segments are a customer-identity grouping used to see which
existing group absorbs the impact - they are intentionally not re-drawn
just because a hypothetical price/contract change shifted a feature.

net_change_in_recoverable_revenue reuses business_impact.py's
recoverable_revenue()/expected_remaining_tenure_months() and their labeled
INTERVENTION_SUCCESS_RATE/survival-based remaining-tenure assumptions
verbatim - the same honesty discipline applies here (see
business_impact.py's module docstring). Remaining tenure is re-derived per
scenario (not reused from business_impact.py's cached table) because
contract_migration changes "Contract" itself, which the survival-based
estimate depends on.

Generalized to accept tenant_config (same pattern as prioritize.py/
business_impact.py/explain.py/recommend.py - read back from
model_dir/split_indices.json via src/data/split.py's load_tenant_config()
when omitted). Only uniform_charge_change and contract_migration are
explicitly covered by this generalization pass and its sanity gate (see
scenario_simulator_sanity_check() below) - discount_offer/loyalty_program
share the exact same underlying plumbing (_load_test_population()/
_compute_scenario_scores()) so they're mechanically fixed too, but neither
was specifically re-verified against a non-Telco tenant here.

SANITY GATE: scenario_simulator_sanity_check() below. Two real, structural
requirements this module has regardless of tenant, both discovered by
reading _compute_scenario_scores() itself rather than assumed:
  1. EVERY scenario type (not just contract_migration) calls
     _original_segment_assignment(), which unconditionally loads
     segment_model.pkl - so scenario_simulator cannot run AT ALL for a
     tenant whose segments module hasn't itself passed its own sanity
     gate. This is a hard prerequisite, not a per-scenario-type detail.
  2. Beyond "does it run," a scenario that changes a feature but produces
     no real shift in churn_probability for any test customer would be
     technically-not-crashing but practically useless (a demo running the
     canonical -10% uniform_charge_change scenario and seeing literally
     nobody's risk move is indistinguishable, to a viewer, from the
     module being silently broken) - the gate runs that exact canonical
     scenario against this tenant's own real test population and requires
     a genuine, non-zero-noise fraction of customers to show a real
     probability change.

contract_migration's "Contract"-equivalent column is segment_feature_column
- not a real onboarding-mappable role (see survival.py's docstring): it's
auto-derived from the fitted Cox model's own hazard ratios and only
resolved once survival_model.pkl exists. This means, structurally,
contract_migration (and therefore the whole scenario_simulator flag, which
gates the module as a whole, not per scenario-type) cannot work for a
tenant whose survival module hasn't passed its own sanity gate - see
run_scenario()'s NoSegmentFeatureColumnError and this module's sanity-gate
docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_tenant_config
from src.features.encode import transform_categorical_features
from src.models.business_impact import business_impact_metadata, expected_remaining_tenure_months, recoverable_revenue
from src.models.segment import CLUSTER_LABELS
from src.models.survival import median_survival_by_contract, resolve_segment_feature_column

HIGH_RISK_THRESHOLD = 0.5
CONTRACT_MIGRATION_SEED = 42
LOYALTY_PROGRAM_SEED = 43
# Illustrative placeholder, not a measured effect - see the loyalty_program_note
# added to this scenario's metadata in run_scenario(). 0.85 = a 15% relative
# reduction in churn_probability for adopters.
LOYALTY_PROGRAM_REDUCTION_FACTOR = 0.85
SCENARIO_TYPES = ("uniform_charge_change", "contract_migration", "discount_offer", "loyalty_program")

# See module docstring's SANITY GATE note. The canonical probe scenario run
# by scenario_simulator_sanity_check() - a plain -10% uniform_charge_change,
# the simplest scenario type and the one every tenant with a revenue_column
# can always run (no segment_feature_column/survival-model dependency the
# way contract_migration has).
SANITY_CHECK_PROBE_PERCENT_CHANGE = -10.0

# A probability re-score is floating-point, not literally identical, even
# for a customer the model barely cares about revenue for - >1e-9 separates
# "genuinely repriced" from "identical before/after, GBM just returned the
# same float twice." Below MIN_SANE_SCENARIO_AFFECTED_FRACTION of the test
# population showing a real shift means the classifier isn't meaningfully
# sensitive to this tenant's revenue_column at all, which makes the whole
# module (built entirely around "reprice and see risk move") a no-op for
# this tenant, not a genuine feature - same "no-op flag" concern as
# priority_ranking's Spearman gate.
SCENARIO_PROBABILITY_CHANGE_EPSILON = 1e-9
MIN_SANE_SCENARIO_AFFECTED_FRACTION = 0.01


class InvalidScenarioError(ValueError):
    pass


class NoSegmentFeatureColumnError(InvalidScenarioError):
    """contract_migration needs a resolved segment_feature_column (the
    "Contract"-equivalent categorical) - only available once this tenant's
    survival module has actually run and auto-derived one (see
    survival.py's docstring). Raised instead of a bare KeyError so the
    caller gets an honest, specific reason rather than a stack trace."""

    def __init__(self):
        super().__init__(
            "contract_migration isn't available for this tenant: no segment/contract-equivalent "
            "column has been resolved yet. This is auto-derived by the survival module from the "
            "fitted model's own hazard ratios, so it requires survival analysis to have already "
            "passed its own sanity gate for this tenant."
        )


def _load_test_population(
    model_dir: Path, data_path: Path, tenant_config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], dict[str, dict[object, int]], list[str]]:
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]

    encoders = joblib.load(model_dir / "encoders.pkl")
    with open(model_dir / "split_indices.json", encoding="utf-8") as f:
        split_info = json.load(f)
    test_idx = split_info["test_idx"]
    feature_names = split_info["feature_names"]

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    test_df = df.loc[test_idx].reset_index(drop=True)
    feature_columns = [c for c in test_df.columns if c not in (id_column, target_column)]
    return test_df, feature_columns, encoders, feature_names


def _score(
    df: pd.DataFrame,
    feature_columns: list[str],
    encoders: dict[str, dict[object, int]],
    feature_names: list[str],
    model: Any,
    positive_index: int = 1,
) -> np.ndarray:
    X = transform_categorical_features(df[feature_columns], encoders).reindex(columns=feature_names, fill_value=0)
    return model.predict_proba(X)[:, positive_index]


def _original_segment_assignment(df: pd.DataFrame, model_dir: Path, encoders: dict[str, dict[object, int]]) -> np.ndarray:
    saved = joblib.load(model_dir / "segment_model.pkl")
    seg_model = saved["model"]
    scaler = saved["scaler"]
    seg_feature_columns = saved["feature_columns"]
    X = transform_categorical_features(df[seg_feature_columns], encoders)
    X_scaled = scaler.transform(X)
    return seg_model.predict(X_scaled)


def _apply_uniform_charge_change(
    test_df: pd.DataFrame, params: dict[str, Any], id_column: str, revenue_column: str
) -> tuple[pd.DataFrame, set[str]]:
    if "percent_change" not in params:
        raise InvalidScenarioError("params.percent_change is required for uniform_charge_change")
    try:
        percent_change = float(params["percent_change"])
    except (TypeError, ValueError):
        raise InvalidScenarioError("params.percent_change must be a number")
    if percent_change <= -100:
        raise InvalidScenarioError("params.percent_change must be greater than -100 (charges can't go negative)")

    modified_df = test_df.copy()
    modified_df[revenue_column] = modified_df[revenue_column] * (1 + percent_change / 100.0)
    affected_customer_ids = set(test_df[id_column])  # uniform - applies to the whole population
    return modified_df, affected_customer_ids


def _apply_contract_migration(
    test_df: pd.DataFrame,
    params: dict[str, Any],
    encoders: dict[str, dict[object, int]],
    id_column: str,
    segment_feature_column: str,
) -> tuple[pd.DataFrame, set[str]]:
    for key in ("from_contract", "to_contract", "migration_rate"):
        if key not in params:
            raise InvalidScenarioError(f"params.{key} is required for contract_migration")

    from_contract = params["from_contract"]
    to_contract = params["to_contract"]
    try:
        migration_rate = float(params["migration_rate"])
    except (TypeError, ValueError):
        raise InvalidScenarioError("params.migration_rate must be a number")
    if not 0.0 <= migration_rate <= 1.0:
        raise InvalidScenarioError("params.migration_rate must be between 0 and 1")

    valid_contracts = (
        sorted(encoders[segment_feature_column].keys())
        if segment_feature_column in encoders
        else sorted(test_df[segment_feature_column].unique())
    )
    if from_contract not in valid_contracts or to_contract not in valid_contracts:
        raise InvalidScenarioError(f"from_contract/to_contract must be one of {valid_contracts}")

    modified_df = test_df.copy()
    eligible_idx = modified_df.index[modified_df[segment_feature_column] == from_contract].to_numpy()
    rng = np.random.default_rng(CONTRACT_MIGRATION_SEED)
    n_migrate = int(round(len(eligible_idx) * migration_rate))
    migrate_positions = (
        rng.choice(eligible_idx, size=n_migrate, replace=False) if n_migrate > 0 else np.array([], dtype=int)
    )
    modified_df.loc[migrate_positions, segment_feature_column] = to_contract
    affected_customer_ids = set(test_df.loc[migrate_positions, id_column])
    return modified_df, affected_customer_ids


def _apply_discount_offer(
    test_df: pd.DataFrame, params: dict[str, Any], segment_ids: np.ndarray, id_column: str, revenue_column: str
) -> tuple[pd.DataFrame, set[str]]:
    if "discount_percent" not in params:
        raise InvalidScenarioError("params.discount_percent is required for discount_offer")
    try:
        discount_percent = float(params["discount_percent"])
    except (TypeError, ValueError):
        raise InvalidScenarioError("params.discount_percent must be a number")
    if not 0 < discount_percent <= 100:
        raise InvalidScenarioError("params.discount_percent must be greater than 0 and at most 100")

    target_segment = params.get("target_segment")
    modified_df = test_df.copy()
    if target_segment is None:
        eligible_mask = np.ones(len(test_df), dtype=bool)
    else:
        valid_segments = sorted(CLUSTER_LABELS.keys())
        try:
            target_segment_id = int(target_segment)
        except (TypeError, ValueError):
            raise InvalidScenarioError("params.target_segment must be an integer segment id or null")
        if target_segment_id not in valid_segments:
            raise InvalidScenarioError(f"params.target_segment must be one of {valid_segments} or null")
        eligible_mask = segment_ids == target_segment_id

    modified_df.loc[eligible_mask, revenue_column] = modified_df.loc[eligible_mask, revenue_column] * (
        1 - discount_percent / 100.0
    )
    affected_customer_ids = set(test_df.loc[eligible_mask, id_column])
    return modified_df, affected_customer_ids


def _apply_loyalty_program(
    test_df: pd.DataFrame, params: dict[str, Any], id_column: str
) -> tuple[pd.DataFrame, set[str]]:
    """No feature is changed here - loyalty_program's effect is a direct,
    clearly-labeled-illustrative churn_probability adjustment applied in
    _compute_scenario_scores() after scoring, not a modeled feature effect.
    modified_df is an unchanged copy of test_df, returned only so this
    scenario type fits the same (modified_df, affected_customer_ids) shape
    as the other three."""
    if "adoption_rate" not in params:
        raise InvalidScenarioError("params.adoption_rate is required for loyalty_program")
    try:
        adoption_rate = float(params["adoption_rate"])
    except (TypeError, ValueError):
        raise InvalidScenarioError("params.adoption_rate must be a number")
    if not 0.0 <= adoption_rate <= 1.0:
        raise InvalidScenarioError("params.adoption_rate must be between 0 and 1")

    modified_df = test_df.copy()
    rng = np.random.default_rng(LOYALTY_PROGRAM_SEED)
    n_adopt = int(round(len(test_df) * adoption_rate))
    adopter_positions = (
        rng.choice(test_df.index.to_numpy(), size=n_adopt, replace=False) if n_adopt > 0 else np.array([], dtype=int)
    )
    affected_customer_ids = set(test_df.loc[adopter_positions, id_column])
    return modified_df, affected_customer_ids


def _build_summary(
    scenario_type: str,
    params: dict[str, Any],
    delta_high_risk: int,
    delta_revenue: float,
    dominant_segment: dict[str, Any],
) -> str:
    if scenario_type == "uniform_charge_change":
        percent_change = float(params["percent_change"])
        direction = "increase" if percent_change > 0 else "decrease"
        change_desc = f"A {abs(percent_change):.0f}% price {direction}"
    elif scenario_type == "contract_migration":
        change_desc = (
            f"Migrating {float(params['migration_rate']) * 100:.0f}% of {params['from_contract']} customers "
            f"to {params['to_contract']}"
        )
    elif scenario_type == "discount_offer":
        target_segment = params.get("target_segment")
        segment_desc = "all customers" if target_segment is None else f"segment {target_segment}"
        change_desc = f"A {float(params['discount_percent']):.0f}% discount offer to {segment_desc}"
    else:  # loyalty_program
        change_desc = f"A loyalty program with {float(params['adoption_rate']) * 100:.0f}% adoption"

    if delta_high_risk > 0:
        risk_clause = f"would push {delta_high_risk} additional customer{'s' if delta_high_risk != 1 else ''} into high-risk status"
    elif delta_high_risk < 0:
        risk_clause = f"would move {abs(delta_high_risk)} customer{'s' if delta_high_risk != -1 else ''} out of high-risk status"
    else:
        risk_clause = "would not change the number of high-risk customers"

    revenue_direction = "increasing" if delta_revenue >= 0 else "decreasing"
    return (
        f"{change_desc} {risk_clause}, concentrated primarily in the '{dominant_segment['label']}' segment, "
        f"{revenue_direction} total revenue at risk by ${abs(delta_revenue):,.0f}."
    )


def _compute_scenario_scores(
    scenario_type: str,
    params: dict[str, Any],
    model_dir: Path,
    data_path: Path,
    tenant_config: dict[str, Any],
) -> tuple[pd.DataFrame, set[str]]:
    """The expensive part, shared by run_scenario() and
    get_customer_scenario_impact(): re-scores the whole test-set population
    before/after the hypothetical change and returns the full per-customer
    DataFrame (not just aggregates), plus the set of customerIDs whose
    features were actually changed by this scenario.

    contract_migration uses a fixed seed, so calling this again later with
    the same stored params reproduces the exact same per-customer result -
    that determinism is what makes get_customer_scenario_impact() safe to
    recompute on demand instead of persisting a full per-customer table for
    every saved scenario.

    result_df's own column names ("customerID", "before_contract", "tenure",
    etc.) are internal implementation detail, not a public contract like
    prioritize.py's output - kept as fixed canonical labels regardless of
    tenant (renamed from this tenant's real id_column/segment_feature_column/
    duration_column) purely so the rest of this module's logic doesn't need
    to thread tenant-specific names through every downstream computation.
    """
    if scenario_type not in SCENARIO_TYPES:
        raise InvalidScenarioError(f"scenario_type must be one of {SCENARIO_TYPES}, got {scenario_type!r}")

    id_column = tenant_config["id_column"]
    target_positive_value = tenant_config["target_positive_value"]
    revenue_column = tenant_config["revenue_column"]
    duration_column = tenant_config.get("duration_column")

    segment_feature_column = resolve_segment_feature_column(model_dir)
    if scenario_type == "contract_migration" and segment_feature_column is None:
        raise NoSegmentFeatureColumnError()

    model = joblib.load(model_dir / "model.pkl")
    positive_index = list(model.classes_).index(target_positive_value)
    test_df, feature_columns, encoders, feature_names = _load_test_population(model_dir, data_path, tenant_config)
    segment_ids = _original_segment_assignment(test_df, model_dir, encoders)

    if scenario_type == "uniform_charge_change":
        modified_df, affected_customer_ids = _apply_uniform_charge_change(test_df, params, id_column, revenue_column)
    elif scenario_type == "contract_migration":
        modified_df, affected_customer_ids = _apply_contract_migration(
            test_df, params, encoders, id_column, segment_feature_column
        )
    elif scenario_type == "discount_offer":
        modified_df, affected_customer_ids = _apply_discount_offer(
            test_df, params, segment_ids, id_column, revenue_column
        )
    else:  # loyalty_program
        modified_df, affected_customer_ids = _apply_loyalty_program(test_df, params, id_column)

    before_probs = _score(test_df, feature_columns, encoders, feature_names, model, positive_index)
    if scenario_type == "loyalty_program":
        # No feature changed, so re-scoring modified_df would just reproduce
        # before_probs - the effect is this direct, labeled-illustrative
        # adjustment for adopters instead (see _apply_loyalty_program()).
        is_adopter = test_df[id_column].isin(affected_customer_ids).to_numpy()
        after_probs = np.where(is_adopter, before_probs * LOYALTY_PROGRAM_REDUCTION_FACTOR, before_probs)
    else:
        after_probs = _score(modified_df, feature_columns, encoders, feature_names, model, positive_index)

    # "contract" placeholder when this scenario type never touches the
    # segment column at all (discount_offer/loyalty_program) or none is
    # resolved (uniform_charge_change for a tenant with no survival model) -
    # expected_remaining_tenure_months() below only ever reads it through
    # median_survival, which is {} in exactly that situation (see below), so
    # the placeholder value itself never affects a real number.
    if segment_feature_column and segment_feature_column in test_df.columns:
        before_contract = test_df[segment_feature_column].values
        after_contract = modified_df[segment_feature_column].values
    else:
        before_contract = after_contract = np.full(len(test_df), "unknown", dtype=object)

    # No real per-customer tenure to offset the survival estimate by if this
    # tenant never mapped a duration column - same documented fallback as
    # business_impact.py's compute_business_impact_bulk().
    tenure_values = test_df[duration_column].values if duration_column and duration_column in test_df.columns else 0.0

    result_df = pd.DataFrame(
        {
            "customerID": test_df[id_column].values,
            "before_probability": before_probs,
            "after_probability": after_probs,
            "before_monthly_charges": test_df[revenue_column].values,
            "after_monthly_charges": modified_df[revenue_column].values,
            "before_contract": before_contract,
            "after_contract": after_contract,
            "tenure": tenure_values,
            "segment": segment_ids,
        }
    )
    result_df["before_revenue_at_risk"] = result_df["before_probability"] * result_df["before_monthly_charges"]
    result_df["after_revenue_at_risk"] = result_df["after_probability"] * result_df["after_monthly_charges"]
    result_df["before_high_risk"] = result_df["before_probability"] > HIGH_RISK_THRESHOLD
    result_df["after_high_risk"] = result_df["after_probability"] > HIGH_RISK_THRESHOLD

    # Remaining tenure is re-derived per-scenario (not reused from
    # business_impact.py's cached table) because contract_migration changes
    # "Contract" itself - the "after" state's remaining tenure must reflect
    # the new contract's survival curve, not the customer's original one.
    # {} (not a crash) when this tenant has no survival model at all - same
    # fallback business_impact.py's compute_business_impact_bulk() already
    # uses, for the same reason.
    median_survival = (
        median_survival_by_contract(model_dir=model_dir, data_path=data_path)
        if (model_dir / "survival_model.pkl").exists()
        else {}
    )
    result_df["before_expected_remaining_tenure_months"] = result_df.apply(
        lambda row: expected_remaining_tenure_months(row["before_contract"], row["tenure"], median_survival), axis=1
    )
    result_df["after_expected_remaining_tenure_months"] = result_df.apply(
        lambda row: expected_remaining_tenure_months(row["after_contract"], row["tenure"], median_survival), axis=1
    )

    return result_df, affected_customer_ids


def run_scenario(
    scenario_type: str,
    params: dict[str, Any],
    scenario_name: str,
    tenant_id: str = "telco",
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del tenant_id  # not used for any column/data-access decision - tenant_config (below) carries all of that now

    model_dir = Path(model_dir)
    data_path = Path(data_path)
    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)

    result_df, affected_customer_ids = _compute_scenario_scores(scenario_type, params, model_dir, data_path, tenant_config)

    crossed_into_high_risk = result_df.loc[
        (~result_df["before_high_risk"]) & result_df["after_high_risk"], "customerID"
    ].tolist()
    crossed_out_of_high_risk = result_df.loc[
        result_df["before_high_risk"] & (~result_df["after_high_risk"]), "customerID"
    ].tolist()

    before_recoverable = result_df.apply(
        lambda row: recoverable_revenue(row["before_revenue_at_risk"], row["before_expected_remaining_tenure_months"]),
        axis=1,
    ).sum()
    after_recoverable = result_df.apply(
        lambda row: recoverable_revenue(row["after_revenue_at_risk"], row["after_expected_remaining_tenure_months"]),
        axis=1,
    ).sum()

    segment_breakdown = []
    for segment_id, group in result_df.groupby("segment"):
        segment_breakdown.append(
            {
                "segment": int(segment_id),
                "label": CLUSTER_LABELS.get(int(segment_id), "unlabeled"),
                "customer_count": int(len(group)),
                "before_revenue_at_risk": float(group["before_revenue_at_risk"].sum()),
                "after_revenue_at_risk": float(group["after_revenue_at_risk"].sum()),
                "before_high_risk_count": int(group["before_high_risk"].sum()),
                "after_high_risk_count": int(group["after_high_risk"].sum()),
            }
        )
    segment_breakdown.sort(key=lambda s: s["segment"])

    total_before_revenue_at_risk = float(result_df["before_revenue_at_risk"].sum())
    total_after_revenue_at_risk = float(result_df["after_revenue_at_risk"].sum())
    before_high_risk_count = int(result_df["before_high_risk"].sum())
    after_high_risk_count = int(result_df["after_high_risk"].sum())
    delta_high_risk = after_high_risk_count - before_high_risk_count
    delta_revenue = total_after_revenue_at_risk - total_before_revenue_at_risk

    dominant_segment = max(
        segment_breakdown, key=lambda s: abs(s["after_revenue_at_risk"] - s["before_revenue_at_risk"])
    )
    summary = _build_summary(scenario_type, params, delta_high_risk, delta_revenue, dominant_segment)

    metadata = business_impact_metadata(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    if scenario_type == "loyalty_program":
        reduction_pct = (1 - LOYALTY_PROGRAM_REDUCTION_FACTOR) * 100
        metadata["loyalty_program_note"] = (
            "Illustrative assumption — no real loyalty-program outcome data exists in this system; "
            f"the {reduction_pct:.0f}% reduction is a placeholder for demonstration, not a measured effect."
        )

    return {
        "scenario_type": scenario_type,
        "scenario_name": scenario_name,
        "params": params,
        "summary": summary,
        "total_revenue_at_risk": {"before": total_before_revenue_at_risk, "after": total_after_revenue_at_risk},
        "customers_high_risk": {
            "before": before_high_risk_count,
            "after": after_high_risk_count,
            "delta": delta_high_risk,
            "crossed_into_high_risk": crossed_into_high_risk,
            "crossed_out_of_high_risk": crossed_out_of_high_risk,
        },
        "net_change_in_recoverable_revenue": float(after_recoverable - before_recoverable),
        "per_segment_breakdown": segment_breakdown,
        "affected_customer_count": len(affected_customer_ids),
        # The full re-scored population size - distinct from
        # affected_customer_count (the subset whose features actually
        # changed) - so a caller can see exactly how large a population
        # this scenario re-scored, not just how many it touched.
        "customers_scored": int(len(result_df)),
        "metadata": metadata,
    }


def get_customer_scenario_impact(
    scenario_type: str,
    params: dict[str, Any],
    customer_id: str,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Re-derive one customer's before/after probability from a previously
    saved scenario's (scenario_type, params) - used by the Customer 360
    timeline to show real scenario impact per customer without persisting a
    ~1400-row customer-level table for every saved scenario. Deterministic:
    contract_migration's fixed seed reproduces the exact same result. Returns
    None if the customer isn't in this scenario's scored population."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)
    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)

    result_df, affected_customer_ids = _compute_scenario_scores(scenario_type, params, model_dir, data_path, tenant_config)
    match = result_df.loc[result_df["customerID"] == customer_id]
    if match.empty:
        return None

    row = match.iloc[0]
    return {
        "before_probability": float(row["before_probability"]),
        "after_probability": float(row["after_probability"]),
        "before_revenue_at_risk": float(row["before_revenue_at_risk"]),
        "after_revenue_at_risk": float(row["after_revenue_at_risk"]),
        "was_affected": customer_id in affected_customer_ids,
    }


def scenario_simulator_sanity_check(
    model_dir: str | Path,
    data_path: str | Path,
    tenant_config: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Returns (failure_reason_or_None, metrics) - see module docstring's
    SANITY GATE note. Structural check first (segments must have passed,
    or every scenario type crashes outright), then a real run of the
    canonical probe scenario against this tenant's own test population to
    confirm the module isn't a silent no-op for them."""
    model_dir = Path(model_dir)
    if not (model_dir / "segment_model.pkl").exists():
        return (
            "scenario_simulator requires segment_model.pkl - every scenario type (not just "
            "contract_migration) re-derives each customer's original segment via the fitted "
            "segmentation model, so this module cannot run at all until segments has passed its "
            "own sanity gate for this tenant.",
            {},
        )

    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)

    try:
        result = run_scenario(
            scenario_type="uniform_charge_change",
            params={"percent_change": SANITY_CHECK_PROBE_PERCENT_CHANGE},
            scenario_name="_sanity_check_probe",
            model_dir=model_dir,
            data_path=data_path,
            tenant_config=tenant_config,
        )
    except Exception as exc:  # noqa: BLE001 - report as a gate failure, not a crash
        return f"canonical probe scenario ({SANITY_CHECK_PROBE_PERCENT_CHANGE:.0f}% uniform_charge_change) failed to run: {exc}", {}

    result_df, _ = _compute_scenario_scores(
        "uniform_charge_change",
        {"percent_change": SANITY_CHECK_PROBE_PERCENT_CHANGE},
        model_dir,
        Path(data_path),
        tenant_config,
    )
    changed = (result_df["after_probability"] - result_df["before_probability"]).abs() > SCENARIO_PROBABILITY_CHANGE_EPSILON
    affected_fraction = float(changed.mean())
    metrics = {
        "probe_affected_fraction": affected_fraction,
        "probe_net_change_in_recoverable_revenue": result["net_change_in_recoverable_revenue"],
    }
    if affected_fraction < MIN_SANE_SCENARIO_AFFECTED_FRACTION:
        return (
            f"only {affected_fraction:.1%} of the test population showed any real churn_probability "
            f"shift under the canonical {SANITY_CHECK_PROBE_PERCENT_CHANGE:.0f}% probe scenario - below "
            f"the {MIN_SANE_SCENARIO_AFFECTED_FRACTION:.0%} floor this needs to be a genuine, working "
            "scenario tool rather than a no-op for this tenant's classifier.",
            metrics,
        )
    return None, metrics
