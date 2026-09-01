"""Customer Lifetime Value regression on a real, per-customer CLV column — independent of the classifier pipeline.

Accepts tenant_config the same way src/models/train.py does (see
src/data/load.py's DEFAULT_TENANT_CONFIG for the full key list).
tenant_config["clv_column"] (Telco: CLTV, from telco_enriched.csv - not the
TotalCharges proxy used elsewhere) is the real per-customer target
regressed on. For Telco specifically, CLTV's correlation with
MonthlyCharges (0.099), tenure (0.396), and TotalCharges (0.342) is well
below the 0.9 leakage threshold used for survival.py's TotalCharges
exclusion, so no feature is excluded on those grounds there. Features are
restricted to the same 19-column set the classifier uses; the other
Telco-enriched-only columns (City, State, Zip Code, Latitude, Longitude,
Churn Score, Churn Reason - tenant_config["clv_excluded_columns"]) are
excluded because they are either high-cardinality geo identifiers or
derived from/biased toward the Churn label, not meant to double as CLV
predictors here. A tenant with no clv_column present simply can't use this
module - train_clv_model() raises a clear error rather than guessing.

Also home to estimate_clv_bulk() (and its sanity gate,
clv_estimate_sanity_check(), and API-facing summary,
load_clv_estimate_summary()) - a SEPARATE, deliberately less confident
capability for a tenant with NO clv_column at all: a formula-based CLV
PROXY (revenue x expected total lifetime), never a trained regression, and
never run for the same tenant a real clv_column already covers above. See
estimate_clv_bulk()'s own docstring for the exact formula.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.clean import clean_data
from src.data.load import DEFAULT_TENANT_CONFIG, load_raw
from src.data.split import load_split_indices, load_tenant_config
from src.features.encode import fit_categorical_encoders, transform_categorical_features
from src.models.business_impact import (
    FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT,
    expected_remaining_tenure_months,
)
from src.models.survival import median_survival_by_contract, resolve_segment_feature_column


def train_clv_model(
    model_dir: str | Path = "models/v1_enriched",
    data_path: str | Path = "data/raw/telco_enriched.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    clv_column = tenant_config["clv_column"]
    clv_excluded_columns = tenant_config.get("clv_excluded_columns", [])

    model_dir = Path(model_dir)
    data_path = Path(data_path)

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    if clv_column not in df.columns:
        raise ValueError(
            f"clv_column '{clv_column}' not found in data - CLV regression needs a real, "
            "per-customer lifetime-value column (see tenant_config['clv_column'])."
        )
    train_idx, test_idx = load_split_indices(model_dir)

    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    non_feature_columns = {id_column, target_column, clv_column, *clv_excluded_columns}
    feature_columns = [col for col in df.columns if col not in non_feature_columns]
    categorical_columns = [col for col in feature_columns if df_train[col].dtype == "object"]
    encoders = fit_categorical_encoders(df_train[feature_columns], categorical_columns)

    X_train = transform_categorical_features(df_train[feature_columns], encoders)
    X_test = transform_categorical_features(df_test[feature_columns], encoders)

    y_train = df_train[clv_column]
    y_test = df_test[clv_column]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(
        {"model": model, "encoders": encoders, "feature_columns": feature_columns, "tenant_config": tenant_config},
        model_dir / "clv_model.pkl",
    )

    feature_importances = sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    (model_dir / "clv_metadata.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {**metrics, "feature_importances": feature_importances}


def load_feature_importances(model_dir: str | Path = "models/v1_enriched") -> list[tuple[str, float]]:
    """Feature importances from the already-fitted CLV model (no refit)."""
    model_dir = Path(model_dir)
    saved = joblib.load(model_dir / "clv_model.pkl")
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    return sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )


def load_clv_metrics(model_dir: str | Path = "models/v1_enriched") -> dict[str, float]:
    """RMSE/MAE/R2 from the last training run, persisted alongside clv_model.pkl."""
    model_dir = Path(model_dir)
    metadata_path = model_dir / "clv_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


# --- Formula-based CLV ESTIMATE, for a tenant with no real clv_column at
# all - see estimate_clv_bulk()'s docstring for the exact formula and why
# it's a deliberately different, less confident thing from train_clv_model()
# above. Deliberately very low: the point is "customers are meaningfully
# differentiated from each other", not a claim about the estimate's
# accuracy (there's no ground truth to score accuracy against here at
# all - that's exactly why this isn't R²-gated the way train_clv_model()
# is).
MIN_SANE_CLV_ESTIMATE_COEFFICIENT_OF_VARIATION = 0.05

CLV_ESTIMATE_METHODOLOGY_NOTE = (
    "estimated_clv is a formula-based PROXY, not a trained regression and not a measured value: "
    "revenue_column x (tenure so far + expected remaining tenure). Expected remaining tenure reuses "
    "this tenant's own already-fitted survival model (business_impact.py's "
    "expected_remaining_tenure_months(), the same math recoverable_revenue already relies on) when "
    "one exists for this tenant, falling back to a flat industry-placeholder otherwise "
    f"({FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT} months, shared with business_impact.py). "
    "Shown only because this tenant has no real, per-customer CLV column to train a genuine "
    "regression against - see clv_estimate_sanity_check()."
)


def estimate_clv_bulk(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Formula-based CLV ESTIMATE for every customer in data_path - NOT a
    trained regression (that's train_clv_model() above, for a tenant WITH a
    real clv_column) and NOT a measured value. Only meaningful for a tenant
    with no clv_column mapped at all; see tenant_training.py's
    _run_optional_modules() for the gate that decides which of the two this
    project actually runs for a given tenant.

    estimated_clv = revenue_column * (tenure_so_far + expected_remaining_tenure_months)

    Reuses exactly the same building blocks business_impact.py's
    compute_business_impact_bulk() already relies on for its own
    expected_remaining_tenure_months column - no new modeling:
      - tenure_so_far: tenant_config["duration_column"] when mapped, else
        0.0 for every customer (the estimate then reflects only the
        forward-looking term - see CLV_ESTIMATE_METHODOLOGY_NOTE).
      - expected_remaining_tenure_months: survival-informed per resolved
        segment (resolve_segment_feature_column()/median_survival_by_contract())
        when a survival model exists for this tenant, else every customer
        falls back to the same FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT
        business_impact.py already uses for the identical reason.

    Deliberately does NOT multiply by INTERVENTION_SUCCESS_RATE the way
    recoverable_revenue() does - that factor answers "how much of at-risk
    revenue survives an intervention", a different question from "what is
    this customer worth in total".
    """
    model_dir = Path(model_dir)
    # Defaults to reading tenant_config back from THIS model_dir's own
    # split_indices.json (same convention as business_impact.py's
    # compute_business_impact_bulk()), not DEFAULT_TENANT_CONFIG (Telco's
    # shape) - unlike train_clv_model() above, this function is meant to be
    # called generically for any tenant, most often without an explicit
    # tenant_config, so defaulting to Telco's column names would silently
    # break for every other tenant's own model_dir.
    tenant_config = tenant_config or load_tenant_config(model_dir)
    data_path = Path(data_path)

    id_column = tenant_config["id_column"]
    revenue_column = tenant_config["revenue_column"]
    duration_column = tenant_config.get("duration_column")

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)

    segment_feature_column = resolve_segment_feature_column(model_dir)
    if (model_dir / "survival_model.pkl").exists():
        median_survival = median_survival_by_contract(model_dir=model_dir, data_path=data_path)
    else:
        median_survival = {}

    revenue = pd.to_numeric(df[revenue_column], errors="coerce").fillna(0.0)
    if duration_column and duration_column in df.columns:
        tenure_so_far = pd.to_numeric(df[duration_column], errors="coerce").fillna(0.0)
    else:
        tenure_so_far = pd.Series(0.0, index=df.index)

    if segment_feature_column and segment_feature_column in df.columns:
        contract_series = df[segment_feature_column]
    else:
        # No resolved segment column (no survival model, or a tenant with
        # no duration_column at all) - a placeholder constant. Only ever
        # read via median_survival's dict lookup below, which is {} in
        # exactly this situation, so this placeholder's actual value never
        # affects the result (same pattern business_impact.py's
        # compute_business_impact_bulk() already uses).
        contract_series = pd.Series("unknown", index=df.index)

    expected_remaining = pd.Series(
        [
            expected_remaining_tenure_months(contract, tenure, median_survival)
            for contract, tenure in zip(contract_series, tenure_so_far)
        ],
        index=df.index,
    )

    result = pd.DataFrame({id_column: df[id_column]})
    result["estimated_clv"] = revenue * (tenure_so_far + expected_remaining)
    return result


def clv_estimate_sanity_check(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Structural gate for estimate_clv_bulk() - there is no real CLV
    column to score an R² against (that's exactly why this proxy exists in
    the first place), so this checks the estimate is well-formed and
    actually differentiates customers, not a degenerate/flat number:
      - every value finite and >= 0 (a bad revenue_column value, or a
        formula edge case, would otherwise silently produce a garbage
        negative or infinite "lifetime value")
      - coefficient of variation (std/mean) above a low bar, so a tenant
        whose formula collapses to (near-)the same number for everyone
        (e.g. every customer shares the same revenue_column value) is
        reported unavailable rather than shown as a fake ranking signal.

    Returns (reason, metrics) - reason is None when it passes, matching
    priority_ranking_sanity_check()/scenario_simulator_sanity_check()'s
    convention (real metrics reported either way, not just on failure).
    """
    tenant_config = tenant_config or load_tenant_config(model_dir)
    result = estimate_clv_bulk(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    values = result["estimated_clv"]

    n_non_finite = int((~np.isfinite(values)).sum())
    finite_values = values[np.isfinite(values)]
    n_negative = int((finite_values < 0).sum())
    mean = float(finite_values.mean()) if len(finite_values) else 0.0
    std = float(finite_values.std()) if len(finite_values) else 0.0
    coefficient_of_variation = (std / mean) if mean > 0 else 0.0

    metrics = {
        "n_customers": int(len(values)),
        "n_non_finite": n_non_finite,
        "n_negative": n_negative,
        "mean": mean,
        "coefficient_of_variation": coefficient_of_variation,
    }

    if n_non_finite > 0:
        return (
            f"{n_non_finite} of {len(values)} customers produced a non-finite estimated CLV value - "
            "formula-based CLV proxy is not usable for this tenant.",
            metrics,
        )
    if n_negative > 0:
        return (
            f"{n_negative} of {len(values)} customers produced a negative estimated CLV value - "
            "formula-based CLV proxy is not usable for this tenant.",
            metrics,
        )
    if mean <= 0:
        return (
            "estimated CLV has a non-positive mean across all customers - formula-based CLV proxy is "
            "not usable for this tenant.",
            metrics,
        )
    if coefficient_of_variation < MIN_SANE_CLV_ESTIMATE_COEFFICIENT_OF_VARIATION:
        return (
            f"estimated CLV does not meaningfully differentiate customers (coefficient of variation "
            f"{coefficient_of_variation:.4f} < {MIN_SANE_CLV_ESTIMATE_COEFFICIENT_OF_VARIATION}) - "
            "formula-based CLV proxy is not usable for this tenant.",
            metrics,
        )
    return None, metrics


def load_clv_estimate_summary(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """API/frontend-facing summary of estimate_clv_bulk() - aggregate
    stats plus the top_n highest-estimated customers, not the full
    per-customer table (unbounded size, and not what a summary display
    needs). Always includes CLV_ESTIMATE_METHODOLOGY_NOTE, so this is
    never presented indistinguishably from a real, trained CLV figure."""
    tenant_config = tenant_config or load_tenant_config(model_dir)
    id_column = tenant_config["id_column"]
    result = estimate_clv_bulk(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    values = result["estimated_clv"]
    top = result.sort_values("estimated_clv", ascending=False).head(top_n)

    return {
        "estimated": True,
        "methodology_note": CLV_ESTIMATE_METHODOLOGY_NOTE,
        "count": int(len(result)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
        "top_customers": [
            {"customer_id": row[id_column], "estimated_clv": float(row["estimated_clv"])}
            for _, row in top.iterrows()
        ],
    }
