"""Cox Proportional Hazards survival model — independent of the classifier pipeline.

Accepts tenant_config the same way src/models/train.py does (see
src/data/load.py's DEFAULT_TENANT_CONFIG for the full key list, including
duration_column/duration_leakage_column/segment_feature_column, the three
keys specific to this module).

tenant_config["duration_leakage_column"] (Telco: TotalCharges) is excluded
as a covariate alongside tenant_config["duration_column"] (Telco: tenure)
when present: it is approximately duration * revenue, so it leaks duration
information back into the model. For Telco specifically, including
TotalCharges inflated the test c-index from 0.868 to 0.931 - well above
published benchmarks (0.83-0.89) for this dataset.

When tenant_config has no duration_leakage_column at all (true for any
self-registered tenant who didn't map one via the onboarding "duration_
leakage_column" role - see src/onboarding/schema_fields.py's
suggest_duration_leakage_column()), train_survival_model() runs the same
check automatically against every remaining numeric covariate: correlation
with duration * revenue (the product relationship - Telco's real
TotalCharges/tenure/MonthlyCharges), AND correlation with duration alone.
Both are checked, not just one - verified against this project's own real
data: TotalCharges' correlation with tenure ALONE is only 0.826 (it would
slip past a duration-only check at any reasonable threshold), but its
correlation with tenure * MonthlyCharges is 0.9996 - the product
relationship is what actually identifies this exact leak shape reliably;
correlation-with-duration-alone is kept as a second, complementary check
(catches a column that's duration itself under another name, without
needing revenue at all) but is not sufficient on its own. Either signal
meeting DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD excludes that column and
reports it in the result's "duration_leakage_warning", rather than
silently trusting it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lifelines import CoxPHFitter

from src.data.clean import clean_data
from src.data.load import DEFAULT_TENANT_CONFIG, load_raw
from src.data.split import load_split_indices
from src.features.encode import fit_categorical_encoders, transform_categorical_features

# Same "how correlated is too correlated to be a coincidence" bar as
# src/onboarding/schema_fields.py's DURATION_LEAKAGE_SUGGESTION_CORRELATION_
# THRESHOLD - applied here as an automatic training-time safety net rather
# than a human-reviewed suggestion.
DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD = 0.9


def train_survival_model(
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]
    target_positive_value = tenant_config["target_positive_value"]
    revenue_column = tenant_config["revenue_column"]
    duration_column = tenant_config["duration_column"]
    duration_leakage_column = tenant_config.get("duration_leakage_column")

    model_dir = Path(model_dir)
    data_path = Path(data_path)

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    if duration_column not in df.columns:
        raise ValueError(
            f"duration_column '{duration_column}' not found in data - survival analysis needs a "
            "tenure/duration-equivalent column (see tenant_config['duration_column'])."
        )
    train_idx, test_idx = load_split_indices(model_dir)

    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    # duration_column is the survival duration, so it must not also appear as
    # a covariate. duration_leakage_column (Telco: TotalCharges) is excluded
    # too when present: it is approximately duration * revenue, so it leaks
    # duration information back in (see module docstring).
    excluded_columns = {id_column, target_column, duration_column}
    auto_detected_leakage_columns: list[tuple[str, float]] = []
    if duration_leakage_column and duration_leakage_column in df.columns:
        excluded_columns.add(duration_leakage_column)
    else:
        # No explicit duration_leakage_column mapped/configured - fall back
        # to the automatic correlation check (see module docstring: both
        # duration*revenue-product correlation AND duration-alone
        # correlation are checked - the product is what actually catches
        # Telco's own real leak shape reliably). Computed on the TRAINING
        # split only, same discipline as the encoders/model fit below - the
        # held-out test split is never touched before c_index is scored on
        # it.
        duration_train_values = pd.to_numeric(df_train[duration_column], errors="coerce")
        revenue_train_values = (
            pd.to_numeric(df_train[revenue_column], errors="coerce") if revenue_column in df_train.columns else None
        )
        duration_times_revenue = (
            duration_train_values * revenue_train_values if revenue_train_values is not None else None
        )
        for column in df.columns:
            if column in excluded_columns or not pd.api.types.is_numeric_dtype(df_train[column]):
                continue
            candidate_values = df_train[column]
            duration_correlation = duration_train_values.corr(candidate_values)
            product_correlation = (
                duration_times_revenue.corr(candidate_values) if duration_times_revenue is not None else None
            )
            strongest = max(
                (abs(c) for c in (duration_correlation, product_correlation) if c is not None), default=0.0
            )
            if strongest >= DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD:
                auto_detected_leakage_columns.append((column, float(strongest)))
        excluded_columns.update(column for column, _ in auto_detected_leakage_columns)
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    categorical_columns = [col for col in feature_columns if df_train[col].dtype == "object"]
    encoders = fit_categorical_encoders(df_train[feature_columns], categorical_columns)

    X_train = transform_categorical_features(df_train[feature_columns], encoders)
    X_test = transform_categorical_features(df_test[feature_columns], encoders)

    cox_train = X_train.copy()
    cox_train["duration"] = df_train[duration_column].values
    cox_train["event"] = df_train[target_column].eq(target_positive_value).astype(int).values

    cox_test = X_test.copy()
    cox_test["duration"] = df_test[duration_column].values
    cox_test["event"] = df_test[target_column].eq(target_positive_value).astype(int).values

    cph = CoxPHFitter()
    cph.fit(cox_train, duration_col="duration", event_col="event")

    c_index = cph.score(cox_test, scoring_method="concordance_index")

    hazard_ratios = cph.hazard_ratios_.reindex(
        cph.hazard_ratios_.sub(1.0).abs().sort_values(ascending=False).index
    )
    coefficients = {feature: float(hr) for feature, hr in hazard_ratios.items()}

    # segment_feature_column drives median_survival_by_contract()/
    # churn_likelihood_within_window()/churn_likelihood_for_customer() - if
    # tenant_config doesn't name a real categorical column for it (true for
    # every self-registered tenant today: there's no dedicated onboarding
    # role for "the Contract-equivalent column", only duration/clv), auto-
    # pick the categorical covariate whose hazard ratio deviates furthest
    # from 1.0 - i.e. the one this fitted model itself found to matter most,
    # not a column-name guess. coefficients is already sorted by that exact
    # deviation, so the first categorical entry in it IS that column. This
    # is what saves alongside the model (not what the caller passed in), so
    # every downstream reader (which all load tenant_config from this same
    # pickle) sees a real, resolved value instead of KeyError-ing on a
    # tenant_config that only ever specified duration_column.
    configured_segment_feature_column = tenant_config.get("segment_feature_column")
    if configured_segment_feature_column in categorical_columns:
        segment_feature_column = configured_segment_feature_column
    else:
        segment_feature_column = next((f for f in coefficients if f in categorical_columns), None)
    resolved_tenant_config = {**tenant_config, "segment_feature_column": segment_feature_column}

    joblib.dump(
        {
            "model": cph,
            "encoders": encoders,
            "feature_columns": feature_columns,
            "tenant_config": resolved_tenant_config,
        },
        model_dir / "survival_model.pkl",
    )

    result: dict[str, Any] = {
        "c_index": float(c_index),
        "coefficients": coefficients,
    }
    if auto_detected_leakage_columns:
        details = ", ".join(f"'{col}' (r={corr:.3f})" for col, corr in auto_detected_leakage_columns)
        result["duration_leakage_warning"] = (
            f"Automatically excluded {len(auto_detected_leakage_columns)} column(s) from survival "
            f"covariates: correlation with the duration column met or exceeded the "
            f"{DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD} leakage threshold (same signature as Telco's "
            f"TotalCharges/tenure leak - see this module's docstring), so they were excluded rather than "
            f"silently trusted. No explicit duration_leakage_column was mapped for this tenant. {details}."
        )
    return result


class CustomerNotFoundError(Exception):
    pass


def resolve_segment_feature_column(model_dir: str | Path) -> str | None:
    """The "Contract"-equivalent column other modules (business_impact.py,
    scenario.py, timeline.py) need for a per-segment breakdown - not a real
    onboarding-mappable role (see this module's docstring): it's
    auto-derived here, at training time, from the fitted Cox model's own
    hazard ratios, and saved into survival_model.pkl's own tenant_config -
    never the classifier's tenant_config directly, which never has this key
    resolved. None when no survival model exists for this tenant (never
    trained, or it failed its own sanity gate) - every caller treats that
    as "no segment breakdown available", not an error, the same fallback
    business_impact.py's compute_business_impact_bulk() already established
    before this was extracted into one shared function."""
    survival_model_path = Path(model_dir) / "survival_model.pkl"
    if not survival_model_path.exists():
        return None
    saved = joblib.load(survival_model_path)
    return saved.get("tenant_config", {}).get("segment_feature_column")


# Dataset tenure is recorded in whole months, but the more actionable framing
# ("X% likely to churn within 7/30/90 days") is in days - this is a fixed
# 30-day-month conversion, a documented approximation, not a calendar-exact
# one.
DAYS_PER_MONTH = 30.0


def churn_likelihood_within_window(
    contract: str,
    current_tenure: float,
    window_days: int,
    model_dir: str | Path = "models/v1",
    data_path: str | Path = "data/raw/telco.csv",
) -> float:
    """Probability of churning within `window_days` days from now, given
    contract type and current tenure - reuses the already-fitted Cox PH
    model's own survival function (not just its median), so this is a real
    model output, not a new assumption. Every feature besides
    segment_feature_column (Telco: Contract) is held at the training-set
    median, same simplification as median_survival_by_contract() (holds
    every other feature at its training-set median and varies only
    segment_feature_column).

    Uses lifelines' conditional_after to get the standard survival-analysis
    conditional-survival identity: given the customer has already survived
    to current_tenure, P(churn in the next window_months) =
    1 - S(current_tenure + window_months | survived to current_tenure).

    tenant_config isn't a parameter here - the fitted model's own
    tenant_config (saved alongside it by train_survival_model()) is reused,
    so this always matches whatever tenant that survival_model.pkl was
    actually trained for, not whatever tenant happens to call this.
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    saved = joblib.load(model_dir / "survival_model.pkl")
    cph = saved["model"]
    encoders = saved["encoders"]
    feature_columns = saved["feature_columns"]
    tenant_config = saved.get("tenant_config", DEFAULT_TENANT_CONFIG)
    segment_feature_column = tenant_config.get("segment_feature_column")
    if not segment_feature_column:
        # Only reachable if this tenant's data had zero categorical
        # covariates at all - train_survival_model() auto-derives a real
        # one from the fitted hazard ratios otherwise, so this is a "no
        # such breakdown exists for this tenant" case, not a config bug.
        raise ValueError(
            "This tenant's survival model has no segment_feature_column (no categorical covariate "
            "was available to auto-derive one from) - a by-category breakdown isn't possible."
        )

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    train_idx, _ = load_split_indices(model_dir)
    X_train = transform_categorical_features(df.loc[train_idx, feature_columns], encoders)
    baseline_row = X_train.median()
    baseline_row[segment_feature_column] = encoders[segment_feature_column][contract]

    window_months = window_days / DAYS_PER_MONTH
    survival_fn = cph.predict_survival_function(
        pd.DataFrame([baseline_row]), times=[window_months], conditional_after=[max(current_tenure, 0.0)]
    )
    conditional_survival_probability = float(survival_fn.to_numpy().flatten()[0])
    return 1.0 - conditional_survival_probability


def churn_likelihood_for_customer(
    customer_id: str, model_dir: str | Path = "models/v1", data_path: str | Path = "data/raw/telco.csv"
) -> dict[str, float]:
    """7/30/90-day churn likelihood for one real customer's actual
    segment_feature_column (Telco: Contract) and duration - every other
    feature still held at the training-set median (see
    churn_likelihood_within_window()'s docstring), so this is a
    contract-and-tenure-conditioned estimate, not a full per-customer
    feature-conditioned one."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    saved = joblib.load(model_dir / "survival_model.pkl")
    tenant_config = saved.get("tenant_config", DEFAULT_TENANT_CONFIG)
    id_column = tenant_config["id_column"]
    duration_column = tenant_config["duration_column"]
    segment_feature_column = tenant_config.get("segment_feature_column")
    if not segment_feature_column:
        raise ValueError(
            "This tenant's survival model has no segment_feature_column (no categorical covariate "
            "was available to auto-derive one from) - per-customer likelihood isn't possible."
        )

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    matches = df.loc[df[id_column] == customer_id]
    if matches.empty:
        raise CustomerNotFoundError(customer_id)
    customer_row = matches.iloc[0]

    return {
        f"{window}d": churn_likelihood_within_window(
            customer_row[segment_feature_column],
            float(customer_row[duration_column]),
            window,
            model_dir=model_dir,
            data_path=data_path,
        )
        for window in (7, 30, 90)
    }


def median_survival_by_contract(
    model_dir: str | Path = "models/v1", data_path: str | Path = "data/raw/telco.csv"
) -> dict[str, float]:
    """Median predicted survival time (duration units, e.g. months) per
    segment_feature_column category (Telco: Contract).

    Holds every other feature at its training-set median and varies only
    segment_feature_column, using the already-fitted model - no refitting.
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    saved = joblib.load(model_dir / "survival_model.pkl")
    cph = saved["model"]
    encoders = saved["encoders"]
    feature_columns = saved["feature_columns"]
    tenant_config = saved.get("tenant_config", DEFAULT_TENANT_CONFIG)
    segment_feature_column = tenant_config.get("segment_feature_column")
    if not segment_feature_column:
        # No categorical covariate existed to auto-derive one from (see
        # train_survival_model()) - an empty map, same shape callers
        # already treat as "no breakdown available" (business_impact.py's
        # own no-survival-model fallback uses the same {} convention).
        return {}

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    train_idx, _ = load_split_indices(model_dir)
    X_train = transform_categorical_features(df.loc[train_idx, feature_columns], encoders)
    baseline_row = X_train.median()

    segments = {}
    for contract_name, code in encoders[segment_feature_column].items():
        row = baseline_row.copy()
        row[segment_feature_column] = code
        segments[contract_name] = row

    segment_df = pd.DataFrame(segments.values(), index=list(segments.keys()))
    medians = cph.predict_median(segment_df)
    # predict_median() returns inf when the survival curve never drops below
    # 0.5 within the observed data range (e.g. Two year contracts churn so
    # rarely that the median lies beyond what was observed) - inf isn't valid
    # JSON, and it also isn't a real time estimate, so it's reported as None.
    return {name: (None if medians[name] == float("inf") else float(medians[name])) for name in segments}
