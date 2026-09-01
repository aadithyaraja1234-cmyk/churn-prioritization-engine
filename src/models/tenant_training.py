"""Stage 2b: wires real training into the async job infrastructure
(api/training.py) proven in Stage 2a. Prepares a self-registered tenant's
confirmed, validated upload into the shape src.models.train.train_model()
already expects, calls that EXISTING function (same one used for Telco/
Banking - no new training logic, no new hyperparameters), and applies a
sanity check to its result.

Stage 2b (this revision): after the classifier, ALSO sequentially runs
survival.py/segment.py/anomaly.py/clv.py - the same four modules Telco
uses, now generalized to accept tenant_config (see src/data/load.py's
DEFAULT_TENANT_CONFIG) - against this tenant's own confirmed data, each
gated by its own sanity check (see _run_optional_modules()). A module that
fails its gate is trained but not enabled - the real metrics and failure
reason are still reported, never hidden (same pattern as
_sanity_check()'s classifier gate below, now applied per-module).

Deliberately has no DB session and makes no decision about job status
itself - it returns a TrainingOutcome describing what actually happened,
factually. api/training.py's _run_training() is what translates that into
training_jobs row updates and Company.feature_flags_json changes; keeping
that split is what keeps db-session/threading concerns out of the actual
data-prep/training logic.

Stage 2c: business_impact.py, prioritize.py, backtest.py, scenario.py,
recommend.py, whatif.py, and timeline.py have all since been generalized
the same way (tenant_config-driven, no hardcoded Telco column literals -
see each module's own docstring) - the "NOT wired to any of this" scope
note that used to live here is stale and has been removed. priority_ranking,
backtest, and customer_timeline are now run and gated here exactly like
survival/segments/anomalies/clv, each on its own real sanity check
(priority_ranking_sanity_check()/backtest_sanity_check()/
timeline_sanity_check()). scenario_simulator is gated the same way, with
one added wrinkle: it has a real structural dependency on segments (every
scenario type loads segment_model.pkl - see scenario.py's
scenario_simulator_sanity_check() docstring), so it's only attempted when
segments already passed for this tenant - a tenant whose segmentation
didn't clear its own bar never even gets a scenario_simulator attempt,
rather than a guaranteed structural failure being reported as if it were
this module's own finding. business_impact_core itself stays gated purely
on the classifier's own is_sane verdict (api/training.py) - Part A's fix
made it tenant-generic, not something with a separate statistical gate of
its own the way these have.

Stage 2d: budget_optimizer is now wired here too, on its own real sanity
check (budget_optimizer_sanity_check()) - the earlier claim that it
"calls into scenario.py" and was therefore out of scope was checked against
the actual module and found incorrect: budget_optimizer.py only ever calls
compute_business_impact_bulk()/recommend_action_for_customer(), both
already fully tenant-generic, so it needed no generalization of its own -
just wiring, plus a real bug fix in api/main.py's /api/budget-optimizer
endpoint, which never passed db= to feature_enabled()/model_dir_for()/etc.
and so 100%-of-the-time reported "unavailable" for every self-registered
tenant regardless of feature_flags_json (same missing-db= bug shape found
and fixed across several other endpoints in this same pass). whatif.py has
no per-tenant training gate of its own (it's gated purely by
business_impact_core in the API layer, the same
"core classifier passed" bar every business-impact-derived feature uses) -
its own generalization is a data-access fix, not a new module_results entry.

Stage 2e: clv_estimated is a new, separate module_results entry (never both
this and clv at once for the same tenant) - for a tenant with NO real
clv_column mapped, src/models/clv.py's estimate_clv_bulk() computes a
formula-based CLV PROXY (revenue x expected total lifetime, survival-
informed when available) instead of leaving CLV a flat "unavailable". Gated
on its own structural/variance sanity check (clv_estimate_sanity_check()),
not clv.py's R² gate - there is no real CLV column to score an R² against,
which is exactly the situation this proxy exists for.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.anomaly import detect_anomalies
from src.models.backtest import backtest_sanity_check, count_churned_test_customers
from src.models.budget_optimizer import budget_optimizer_sanity_check
from src.models.clv import clv_estimate_sanity_check, train_clv_model
from src.models.prioritize import priority_ranking_sanity_check
from src.models.scenario import scenario_simulator_sanity_check
from src.models.segment import run_segmentation
from src.models.survival import resolve_segment_feature_column, train_survival_model
from src.models.timeline import timeline_sanity_check
from src.models.train import train_model

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"
TENANT_UPLOADS_DIR = ROOT / "data" / "tenant_uploads"

# "e.g. 0.55 - barely better than random" / "suspiciously above 0.98
# (likely leakage)" - both floor and ceiling per the task brief, grounded
# in this project's own real precedent: src/models/survival.py excludes
# TotalCharges specifically because including it leaked duration
# information back in and inflated a metric well past realistic
# benchmarks (0.868 -> 0.931 c-index, docstring has the full story) - the
# same "too good to be true" pattern this ceiling exists to catch again.
MIN_SANE_ROC_AUC = 0.55
MAX_SANE_ROC_AUC = 0.98

# Per-module sanity gates, each mirroring the classifier's floor-and-ceiling
# (or floor-and-degeneracy) pattern above rather than inventing a new shape
# of check per module.
#
# Survival: same leakage-guard logic as the classifier's ceiling - a
# c-index this high almost always means a duration-leaking covariate
# slipped through (this project's own real precedent: TotalCharges, see
# src/models/survival.py's docstring). 0.95 leaves a little more headroom
# than the classifier's 0.98 since c-index's usable range (0.5-1.0) is
# narrower than ROC-AUC's.
MIN_SANE_SURVIVAL_C_INDEX = 0.55
MAX_SANE_SURVIVAL_C_INDEX = 0.95

# Segmentation: 0.05 is a low but real floor - silhouette scores near 0
# mean clusters barely more separated than random partitions, but genuine,
# useful customer segmentations (including Telco's own, finalized at 0.157)
# routinely score well below the >0.5 textbook "strong structure" bar, so a
# stricter floor would reject real, usable segmentations. A single cluster
# holding >90% of customers is checked separately: a low-but-passing
# silhouette can still hide a degenerate 99%/1% split that isn't a
# meaningful segmentation in practice.
MIN_SANE_SILHOUETTE = 0.05
MAX_SANE_DOMINANT_CLUSTER_SHARE = 0.90

# Anomaly detection: contamination is a chosen INPUT (5% here, same as
# Telco's own finalized choice - see tests/test_anomaly.py), so in a
# healthy fit the flagged fraction should land close to it. Outside
# 1%-20% signals a degenerate fit (isolation forest either can't find
# meaningfully distinctive outliers, or is flagging a huge chunk of the
# population as "anomalous", which isn't a meaningful outlier signal
# anymore) rather than a real distribution's shape.
MIN_SANE_ANOMALY_FLAGGED_FRACTION = 0.01
MAX_SANE_ANOMALY_FLAGGED_FRACTION = 0.20

# CLV: a deliberately very low bar (Telco's own real, validated R² is
# 0.226 - see src/models/clv.py's docstring on CLTV's large unexplained/
# proprietary component) - the point is "meaningfully better than
# predicting the mean for every customer alike" (R²<=0 territory), not a
# high-accuracy claim.
MIN_SANE_CLV_R2 = 0.05


class TrainingDataError(Exception):
    """Raised when the confirmed upload/mapping can't be turned into a
    trainable dataset at all (e.g. no column mapped to a required role, or
    the target has fewer than two classes) - an expected, honestly-
    reported failure mode, not a bug. api/training.py catches this exactly
    like any other training exception."""


@dataclass
class TrainingOutcome:
    roc_auc: float
    pr_auc: float
    n_train: int
    n_test: int
    churn_rate_train: float
    churn_rate_test: float
    model_dir: str  # relative to repo root, e.g. "models/meridian-wireless"
    data_path: str  # relative to repo root - the FILTERED training CSV actually used
    inferred_target_positive_value: Any
    is_sane: bool
    sanity_warning: str | None
    # Hyperparameter search results (src/models/train.py's
    # _tune_hyperparameters()) - cv_score_during_tuning is the training-
    # split-only CV estimate used to PICK hyperparameters; final_test_roc_auc
    # (== roc_auc above) is the real, once-computed held-out number. Never
    # conflate the two - see train_model()'s metadata dict comment.
    cv_score_during_tuning: float | None  # None specifically when tuning was skipped - see src/models/train.py
    final_test_roc_auc: float
    best_hyperparameters: dict[str, Any]
    tuning_seconds: float
    tuning_skipped_reason: str | None  # non-None only when the training set was too small to tune reliably
    # {"survival": {"ran": bool, "passed": bool, "metrics": dict|None, "reason": str|None}, "segments": ..., "anomalies": ..., "clv": ...}
    # ran=False means never attempted (no duration/clv column mapped) -
    # distinct from ran=True, passed=False (attempted, real result, just
    # didn't clear its sanity gate). metrics is always the real, honest
    # result when ran=True, whether or not it passed.
    module_results: dict[str, dict[str, Any]] = field(default_factory=dict)


def _tenant_config_from_mapping(column_mapping: dict[str, str], df: pd.DataFrame) -> dict[str, Any]:
    id_columns = [c for c, r in column_mapping.items() if r == "customer_id"]
    target_columns = [c for c, r in column_mapping.items() if r == "target"]
    revenue_columns = [c for c, r in column_mapping.items() if r == "revenue"]
    duration_columns = [c for c, r in column_mapping.items() if r == "duration"]
    clv_columns = [c for c, r in column_mapping.items() if r == "clv"]
    duration_leakage_columns = [c for c, r in column_mapping.items() if r == "duration_leakage_column"]

    if len(id_columns) != 1:
        raise TrainingDataError("Exactly one column must be mapped to customer_id to train a model.")
    if len(target_columns) != 1:
        raise TrainingDataError("Exactly one column must be mapped to target to train a model.")
    if len(revenue_columns) != 1:
        raise TrainingDataError("Exactly one column must be mapped to revenue to train a model.")
    # duration/clv/duration_leakage_column are optional (unlike id/target/
    # revenue above) - a tenant with none of them mapped can still train
    # the classifier and run segment.py/anomaly.py; survival.py/clv.py are
    # just skipped, honestly, rather than blocking training entirely (see
    # _run_optional_modules()), and survival.py falls back to its own
    # automatic correlation-based leakage check when no
    # duration_leakage_column is mapped (see its DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD).
    if len(duration_columns) > 1:
        raise TrainingDataError("At most one column may be mapped to duration.")
    if len(clv_columns) > 1:
        raise TrainingDataError("At most one column may be mapped to clv.")
    if len(duration_leakage_columns) > 1:
        raise TrainingDataError("At most one column may be mapped to duration_leakage_column.")

    target_column = target_columns[0]
    target_series = df[target_column].dropna()
    if target_series.nunique() < 2:
        raise TrainingDataError(f"Target column '{target_column}' has fewer than two distinct values.")

    # Which literal value in the target column means "churned"? config.yaml's
    # hand-curated demo tenants specify target_positive_value explicitly;
    # nothing in the onboarding wizard's column_mapping captures this for a
    # self-registered tenant. Inferred as the minority class - churn is
    # virtually always the rarer outcome in a real portfolio - and reported
    # back in TrainingOutcome so this assumption is visible, never silent.
    target_positive_value = target_series.value_counts().idxmin()

    tenant_config: dict[str, Any] = {
        "id_column": id_columns[0],
        "target_column": target_column,
        "revenue_column": revenue_columns[0],
        "target_positive_value": target_positive_value,
    }
    if duration_columns:
        tenant_config["duration_column"] = duration_columns[0]
    if clv_columns:
        tenant_config["clv_column"] = clv_columns[0]
    if duration_leakage_columns:
        tenant_config["duration_leakage_column"] = duration_leakage_columns[0]
    # No onboarding role for segment_feature_column (Telco: "Contract") -
    # auto-derived by survival.py itself from the fitted model's own hazard
    # ratios (see its docstring). duration_leakage_column DOES have a role
    # now (see src/onboarding/schema_fields.py's suggest_duration_leakage_column())
    # - if the company didn't map one, survival.py still runs its own
    # automatic correlation-based check rather than trusting every
    # covariate (see DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD).
    return tenant_config


def _filtered_training_frame(df: pd.DataFrame, column_mapping: dict[str, str], tenant_config: dict[str, Any]) -> pd.DataFrame:
    """Only id/target/revenue/feature/duration/clv/duration_leakage_column-
    mapped columns - a column the user mapped to "ignore" during onboarding
    must never leak into the model as a feature just because it happened to
    be in the uploaded CSV. duration/clv/duration_leakage_column-mapped
    columns are kept alongside "feature" ones (not dropped) for two
    reasons: survival.py/clv.py need the raw column present in this same
    filtered CSV to train on later (survival.py explicitly EXCLUDES
    duration_leakage_column from its own covariates - see its docstring -
    but the classifier still gets it as an ordinary feature), and -
    mirroring Telco's own real behavior, where tenure/TotalCharges are both
    survival-relevant AND ordinary classifier features - there's no reason
    to withhold a genuinely predictive column from the classifier just
    because it also has a second role. This is the one thing
    src.data.split.split_data() can't do itself: it turns "every column
    except id/target" into X, so filtering has to happen
    before the file ever reaches train_model()."""
    feature_columns = [
        c for c, r in column_mapping.items() if r in ("feature", "duration", "clv", "duration_leakage_column")
    ]
    keep_columns = list(
        dict.fromkeys(
            [tenant_config["id_column"], tenant_config["target_column"], tenant_config["revenue_column"], *feature_columns]
        )
    )
    return df[keep_columns].copy()


def _sanity_check(roc_auc: float) -> str | None:
    if roc_auc < MIN_SANE_ROC_AUC:
        return (
            f"ROC-AUC ({roc_auc:.3f}) is at or below {MIN_SANE_ROC_AUC} - barely better than random. This "
            "usually means the mapped feature columns don't actually carry predictive signal for this "
            "target (or the column mapping itself is wrong), not something more training data alone would "
            "fix. Feature flags were NOT auto-enabled for this tenant - review before enabling manually."
        )
    if roc_auc > MAX_SANE_ROC_AUC:
        return (
            f"ROC-AUC ({roc_auc:.3f}) is suspiciously close to a perfect score (above {MAX_SANE_ROC_AUC}). "
            "This project has hit this exact failure mode before (src/models/survival.py's docstring - "
            "including a duration-leaking feature inflated a metric well past realistic benchmarks) - it "
            "almost always means one of the mapped feature columns leaks the target, not a genuinely "
            "excellent model. Feature flags were NOT auto-enabled for this tenant - review the mapped "
            "feature columns for leakage before enabling manually."
        )
    return None


def _passed(metrics: dict[str, Any]) -> dict[str, Any]:
    return {"ran": True, "passed": True, "metrics": metrics, "reason": None}


def _failed(metrics: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"ran": True, "passed": False, "metrics": metrics, "reason": reason}


def _skipped(reason: str) -> dict[str, Any]:
    return {"ran": False, "passed": False, "metrics": None, "reason": reason}


def _errored(exc: Exception) -> dict[str, Any]:
    return {"ran": True, "passed": False, "metrics": None, "reason": f"training crashed: {exc}"}


def _run_optional_modules(tenant_config: dict[str, Any], model_dir: Path, data_path: Path) -> dict[str, dict[str, Any]]:
    """Runs survival.py/segment.py/anomaly.py/clv.py against this tenant's
    own confirmed data (the exact same generalized functions Telco uses -
    see each module's own docstring), gates each on its own sanity check,
    and returns a real, honest outcome for every one of the four -
    including the ones that fail their gate or can't run at all. Never
    raises: an unexpected crash inside any one module is caught and
    reported as that module's own failure (_errored), not a fatal error for
    the whole training job - one module's bug shouldn't take down the
    classifier result the job already has in hand.
    """
    results: dict[str, dict[str, Any]] = {}

    if "duration_column" not in tenant_config:
        results["survival"] = _skipped(
            "unavailable: no duration/tenure-equivalent column provided (map a column to the "
            "'duration' role to enable survival analysis)"
        )
    else:
        try:
            survival_metrics = train_survival_model(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
            c_index = survival_metrics["c_index"]
            if c_index <= MIN_SANE_SURVIVAL_C_INDEX:
                results["survival"] = _failed(
                    survival_metrics,
                    f"c-index ({c_index:.3f}) is at or below {MIN_SANE_SURVIVAL_C_INDEX} - barely better "
                    "than random at ranking who churns sooner.",
                )
            elif c_index > MAX_SANE_SURVIVAL_C_INDEX:
                results["survival"] = _failed(
                    survival_metrics,
                    f"c-index ({c_index:.3f}) is suspiciously close to a perfect score (above "
                    f"{MAX_SANE_SURVIVAL_C_INDEX}) - likely a covariate leaking the duration column back in.",
                )
            else:
                results["survival"] = _passed(survival_metrics)
        except Exception as exc:  # noqa: BLE001 - see docstring: one module's crash must not fail the job
            results["survival"] = _errored(exc)

    try:
        segment_result = run_segmentation(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
        chosen_k = segment_result["chosen_k"]
        silhouette = next(r["silhouette"] for r in segment_result["k_search_results"] if r["k"] == chosen_k)
        cluster_shares = segment_result["cluster_assignments"]["cluster"].value_counts(normalize=True)
        dominant_cluster_share = float(cluster_shares.max())
        metrics = {"chosen_k": chosen_k, "silhouette": silhouette, "dominant_cluster_share": dominant_cluster_share}
        if silhouette <= MIN_SANE_SILHOUETTE:
            results["segments"] = _failed(
                metrics,
                f"silhouette score ({silhouette:.3f}) is at or below {MIN_SANE_SILHOUETTE} - clusters "
                "aren't meaningfully separated from each other.",
            )
        elif dominant_cluster_share > MAX_SANE_DOMINANT_CLUSTER_SHARE:
            results["segments"] = _failed(
                metrics,
                f"{dominant_cluster_share:.0%} of customers landed in a single cluster - a degenerate "
                "split, not a genuine segmentation.",
            )
        else:
            results["segments"] = _passed(metrics)
    except Exception as exc:  # noqa: BLE001
        results["segments"] = _errored(exc)

    try:
        anomaly_result = detect_anomalies(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
        flagged_fraction = float((anomaly_result["results"]["anomaly_flag"] == -1).mean())
        metrics = {"contamination": anomaly_result["contamination"], "flagged_fraction": flagged_fraction}
        if not (MIN_SANE_ANOMALY_FLAGGED_FRACTION <= flagged_fraction <= MAX_SANE_ANOMALY_FLAGGED_FRACTION):
            results["anomalies"] = _failed(
                metrics,
                f"{flagged_fraction:.1%} of customers were flagged as anomalous - outside the "
                f"{MIN_SANE_ANOMALY_FLAGGED_FRACTION:.0%}-{MAX_SANE_ANOMALY_FLAGGED_FRACTION:.0%} sane range "
                "(a degenerate all-or-nothing result, not a meaningful outlier signal).",
            )
        else:
            results["anomalies"] = _passed(metrics)
    except Exception as exc:  # noqa: BLE001
        results["anomalies"] = _errored(exc)

    if "clv_column" not in tenant_config:
        results["clv"] = _skipped(
            "unavailable: no CLV-equivalent column provided (map a column to the 'clv' role to enable "
            "CLV modeling)"
        )
        # clv_estimated: the formula-based proxy (src/models/clv.py's
        # estimate_clv_bulk()) only ever makes sense for a tenant with NO
        # real clv_column at all - a tenant that already has one gets the
        # real, trained clv model above instead, never both. Structural/
        # variance gate, not R² (there's no ground truth to score against -
        # see clv_estimate_sanity_check()'s docstring).
        try:
            reason, metrics = clv_estimate_sanity_check(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
            results["clv_estimated"] = _failed(metrics, reason) if reason else _passed(metrics)
        except Exception as exc:  # noqa: BLE001
            results["clv_estimated"] = _errored(exc)
    else:
        try:
            clv_metrics = train_clv_model(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
            r2 = clv_metrics["r2"]
            metrics = {"r2": r2, "rmse": clv_metrics["rmse"], "mae": clv_metrics["mae"]}
            if r2 <= MIN_SANE_CLV_R2:
                results["clv"] = _failed(
                    metrics,
                    f"R² ({r2:.3f}) is at or below {MIN_SANE_CLV_R2} - not meaningfully better than "
                    "predicting the same average value for every customer.",
                )
            else:
                results["clv"] = _passed(metrics)
        except Exception as exc:  # noqa: BLE001
            results["clv"] = _errored(exc)
        results["clv_estimated"] = _skipped(
            "unavailable: this tenant has a real, mapped CLV column - the formula-based estimate is "
            "only offered when no real CLV data exists at all"
        )

    # priority_ranking/backtest/customer_timeline: same treatment as the
    # four modules above (own real gate, own honest pass/fail/error), none
    # of them structurally dependent on survival/segments/anomalies/clv
    # having passed - see each module's own sanity-check docstring for its
    # specific threshold and reasoning.
    try:
        reason, metrics = priority_ranking_sanity_check(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
        results["priority_ranking"] = _failed(metrics, reason) if reason else _passed(metrics)
    except Exception as exc:  # noqa: BLE001
        results["priority_ranking"] = _errored(exc)

    try:
        n_churned_test = count_churned_test_customers(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
        metrics = {"n_churned_test": n_churned_test}
        reason = backtest_sanity_check(n_churned_test)
        results["backtest"] = _failed(metrics, reason) if reason else _passed(metrics)
    except Exception as exc:  # noqa: BLE001
        results["backtest"] = _errored(exc)

    try:
        reason = timeline_sanity_check(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
        results["customer_timeline"] = _failed({}, reason) if reason else _passed({})
    except Exception as exc:  # noqa: BLE001
        results["customer_timeline"] = _errored(exc)

    # scenario_simulator: real structural dependency on segments (see this
    # function's own docstring and scenario.py's scenario_simulator_sanity_
    # check() docstring) - skipped, not attempted-and-failed, when segments
    # itself didn't pass, since the guaranteed crash that would follow is a
    # segments-level finding, not a new one of this module's own.
    if not results["segments"]["passed"]:
        results["scenario_simulator"] = _skipped(
            "unavailable: requires segments to have passed its own sanity gate first - every scenario "
            "type re-derives each customer's segment via the fitted segmentation model."
        )
    else:
        try:
            reason, metrics = scenario_simulator_sanity_check(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
            results["scenario_simulator"] = _failed(metrics, reason) if reason else _passed(metrics)
        except Exception as exc:  # noqa: BLE001
            results["scenario_simulator"] = _errored(exc)

    # budget_optimizer: no structural dependency on any of the modules
    # above (it only ever calls compute_business_impact_bulk()/
    # recommend_action_for_customer(), gated already by business_impact_core
    # passing at the classifier level, not by anything gated here) - always
    # attempted. clv_data_path is always data_path itself for a self-
    # registered tenant (same convention business_impact.py's
    # compute_business_impact_bulk() docstring documents: no separate
    # enriched dataset the way Telco has).
    try:
        reason, metrics = budget_optimizer_sanity_check(model_dir=model_dir, data_path=data_path, clv_data_path=data_path)
        results["budget_optimizer"] = _failed(metrics, reason) if reason else _passed(metrics)
    except Exception as exc:  # noqa: BLE001
        results["budget_optimizer"] = _errored(exc)

    return results


def prepare_and_train(
    tenant_id: str,
    csv_text: str,
    column_mapping: dict[str, str] | None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> TrainingOutcome:
    if not column_mapping:
        raise TrainingDataError(
            "No confirmed column mapping found for this upload - it must be validated via "
            "POST /api/onboarding/validate before training can start."
        )

    df = pd.read_csv(io.StringIO(csv_text))
    tenant_config = _tenant_config_from_mapping(column_mapping, df)
    filtered = _filtered_training_frame(df, column_mapping, tenant_config)

    TENANT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    data_path = TENANT_UPLOADS_DIR / f"{tenant_id}.csv"
    filtered.to_csv(data_path, index=False)

    model_dir = ROOT / "models" / tenant_id
    metadata = train_model(
        data_path=data_path,
        config_path=config_path,
        output_dir=model_dir,
        tenant_config=tenant_config,
    )

    roc_auc = float(metadata["roc_auc"])
    sanity_warning = _sanity_check(roc_auc)

    # Runs regardless of the classifier's own sanity verdict: survival/
    # segments/anomalies/clv are independent, separately-fit analyses (none
    # of them use the classifier's predictions as an input), each gated on
    # its own real result - a borderline classifier doesn't make these
    # other four modules' own numbers any less real.
    module_results = _run_optional_modules(tenant_config, model_dir, data_path)

    return TrainingOutcome(
        roc_auc=roc_auc,
        pr_auc=float(metadata["pr_auc"]),
        n_train=int(metadata["n_train"]),
        n_test=int(metadata["n_test"]),
        churn_rate_train=float(metadata["churn_rate_train"]),
        churn_rate_test=float(metadata["churn_rate_test"]),
        model_dir=str(model_dir.relative_to(ROOT)).replace("\\", "/"),
        data_path=str(data_path.relative_to(ROOT)).replace("\\", "/"),
        inferred_target_positive_value=tenant_config["target_positive_value"],
        is_sane=sanity_warning is None,
        sanity_warning=sanity_warning,
        cv_score_during_tuning=(
            float(metadata["cv_score_during_tuning"]) if metadata["cv_score_during_tuning"] is not None else None
        ),
        final_test_roc_auc=float(metadata["final_test_roc_auc"]),
        best_hyperparameters=metadata["best_hyperparameters"],
        tuning_seconds=float(metadata["tuning_seconds"]),
        tuning_skipped_reason=metadata.get("tuning_skipped_reason"),
        module_results=module_results,
    )
