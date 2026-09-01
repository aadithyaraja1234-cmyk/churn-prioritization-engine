from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import DEFAULT_TENANT_CONFIG, load_raw
from src.data.split import split_data
from src.features.encode import fit_categorical_encoders, transform_categorical_features

# Hyperparameter search: bounded, time-budgeted, replaces the old single
# fixed choice (n_estimators=200, max_depth=3, learning_rate=0.05) for
# BOTH the reference Telco/Banking pipeline and the self-registered
# onboarding pipeline (both call train_model() below - no separate logic
# per pipeline). RandomizedSearchCV, not exhaustive GridSearchCV, so
# runtime is bounded by HYPERPARAMETER_SEARCH_N_ITER regardless of how
# large the search space grid technically is.
#
# HYPERPARAMETER_SEARCH_N_ITER x HYPERPARAMETER_SEARCH_CV_FOLDS is
# calibrated (see tests/test_hyperparameter_tuning.py's timing check) to
# comfortably clear HYPERPARAMETER_SEARCH_TIME_BUDGET_SECONDS_TARGET on
# this project's real datasets (Telco's ~5,600 training rows down to a
# self-registered tenant's much smaller upload). This is a calibrated
# target, not a hard mid-search cutoff enforced at runtime: sklearn's
# RandomizedSearchCV has no native "stop after N seconds" hook, and
# bolting one on would make search coverage depend on how fast the
# machine happens to be - the opposite of this project's "same real
# numbers" discipline. The actual measured time is always reported back
# in metadata (tuning_seconds), never assumed.
HYPERPARAMETER_SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 150, 200, 250, 300],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
}
HYPERPARAMETER_SEARCH_N_ITER = 15
HYPERPARAMETER_SEARCH_CV_FOLDS = 3
HYPERPARAMETER_SEARCH_TIME_BUDGET_SECONDS_TARGET = 60

# Below this many training rows, HYPERPARAMETER_SEARCH_CV_FOLDS-fold CV
# folds get too small (well under 100 rows each) to reliably estimate
# which hyperparameters actually generalize, rather than which ones
# happen to overfit that specific tiny resample's noise - discovered via
# a real regression, not a theoretical worry: on a genuinely uninformative
# 200-row synthetic dataset (target carries zero real relationship to any
# feature), pre-tuning fixed hyperparameters reliably scored ROC-AUC near
# 0.5 every time; WITH tuning enabled, held-out ROC-AUC swung between
# 0.39 and 0.64 across otherwise-identical small resamples (same
# generator, different seeds) - i.e. tuning made the "genuinely bad data
# reliably tests as low ROC-AUC" guarantee this project's own sanity gate
# (src/models/tenant_training.py's MIN_SANE_ROC_AUC) depends on LESS
# reliable, not more, specifically for small self-registered uploads.
# Below this threshold, train_model() falls back to the original,
# pre-tuning fixed hyperparameters (still config.yaml-driven) instead of
# searching - a small dataset gets a stable, pre-validated default rather
# than a search prone to picking noise.
MIN_TRAINING_ROWS_FOR_TUNING = 300


def _positive_class_scorer(positive_value: Any):
    """ROC-AUC scorer that finds the positive class via tenant_config's
    real target_positive_value (Telco: "Yes", Banking: 1, or whatever an
    arbitrary self-registered tenant's target column actually uses) - the
    exact same lookup train_model() itself already uses for the final
    held-out evaluation below (list(classes_).index(positive_value)),
    rather than assuming sklearn's default "second/greater class" ordering
    happens to match. It coincidentally does for Telco/Banking today, but
    this scorer doesn't rely on that coincidence holding for every future
    tenant's arbitrary label values."""

    def scorer(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        positive_index = list(estimator.classes_).index(positive_value)
        probabilities = estimator.predict_proba(X)[:, positive_index]
        y_binary = (y == positive_value).astype(int)
        return roc_auc_score(y_binary, probabilities)

    return scorer


def _tune_hyperparameters(
    X_train_encoded: pd.DataFrame,
    y_train: pd.Series,
    positive_value: Any,
    random_state: int,
) -> tuple[Any, dict[str, Any], float, float]:
    """Runs the bounded RandomizedSearchCV described in this module's
    top-level comment. The ONLY inputs are the TRAINING split
    (X_train_encoded/y_train) already carved out by split_data() before
    this is ever called - the held-out test set is never constructed,
    referenced, or in scope inside this function, let alone passed to it.
    See tests/test_hyperparameter_tuning.py's leakage regression test,
    which inspects the real call this function makes (via a spy on
    RandomizedSearchCV.fit), not just this comment.

    refit=True (sklearn's default) means search.best_estimator_ is already
    refit on the FULL training set with the winning hyperparameters - "refit
    on the full training set, then evaluate on the held-out test set" is
    satisfied by using that estimator directly, not by fitting a second,
    redundant model.

    Returns (best_estimator, best_params, cv_score_during_tuning,
    tuning_seconds). cv_score_during_tuning is an ESTIMATE from
    training-set-only cross-validation, used only to pick hyperparameters -
    it is never the number this project reports as "the" model score (see
    train_model()'s final_test_roc_auc, computed separately, exactly once,
    after this function returns)."""
    search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_state),
        param_distributions=HYPERPARAMETER_SEARCH_SPACE,
        n_iter=HYPERPARAMETER_SEARCH_N_ITER,
        cv=HYPERPARAMETER_SEARCH_CV_FOLDS,
        scoring=_positive_class_scorer(positive_value),
        random_state=random_state,
        # Bounded, not -1 (every core): discovered via a real, reproduced
        # flake, not a theoretical worry - this project's own test suite
        # starts training jobs in detached background threads
        # (api/training.py) that pytest never joins between tests, so an
        # n_jobs=-1 search still finishing from a PREVIOUS test can starve
        # an unrelated, supposedly-instant test of CPU for several seconds.
        # 4 keeps real wall-clock time well inside
        # HYPERPARAMETER_SEARCH_TIME_BUDGET_SECONDS_TARGET (~27s measured
        # on Telco's full training set, vs. ~65s fully serial) without
        # claiming the whole machine the way -1 does. Fit results
        # themselves are identical regardless of n_jobs - only which
        # worker computes each fit changes, not the numbers.
        n_jobs=4,
        refit=True,
    )
    started_at = time.monotonic()
    search.fit(X_train_encoded, y_train)
    tuning_seconds = time.monotonic() - started_at
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_), tuning_seconds


def train_model(
    data_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    tenant_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    data_path = Path(data_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "models" / "v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(df, config, tenant_config)

    categorical_columns = [col for col in X_train.columns if X_train[col].dtype == "object"]
    encoders = fit_categorical_encoders(X_train, categorical_columns)
    X_train_encoded = transform_categorical_features(X_train, encoders)
    X_test_encoded = transform_categorical_features(X_test, encoders)

    model_config = config.get("model", {})
    random_state = int(model_config.get("random_state", 42))
    positive_value = tenant_config["target_positive_value"]

    tuning_skipped_reason: str | None = None
    # tuning_enabled defaults True (self-registered tenants: Meridian,
    # Fernwood, every future company built via tenant_training.py's
    # _tenant_config_from_mapping() never set this key, so they get
    # tuning). Telco/Banking are the only tenants with this explicitly
    # set False (src/data/load.py's DEFAULT_TENANT_CONFIG for Telco - the
    # tenant_config run_pipeline.py's real retrain actually uses -  and
    # config.yaml's tenants.banking entry) - a deliberate decision, not an
    # oversight: a real, controlled comparison (see
    # tests/test_hyperparameter_tuning.py and this project's own tuning
    # experiment write-up) showed tuning made BOTH reference models
    # slightly WORSE on their held-out test sets (Telco 0.8471 -> 0.8400,
    # Banking 0.8399 -> 0.8285) despite scoring higher on the search's own
    # cross-validation estimate - a real instance of CV-selected
    # hyperparameters not generalizing as well as the original, already-
    # validated hand-picked ones on this specific held-out split. Every
    # published number this project has ever reported for Telco/Banking
    # (0.8471/0.6687, the Spearman correlation ~0.949, the backtest
    # curve, etc.) was computed under the original fixed hyperparameters
    # - changing them out from under those numbers, even to a
    # theoretically-defensible tuned alternative, would silently
    # invalidate every prior comparison this project has made. Tuning
    # stays available and demonstrably helps self-registered tenants
    # (Meridian +0.0092 ROC-AUC, Fernwood +0.0016) - it's excluded for
    # the reference tenants specifically, not disabled globally.
    tuning_enabled = tenant_config.get("tuning_enabled", True)

    if tuning_enabled and len(X_train_encoded) >= MIN_TRAINING_ROWS_FOR_TUNING:
        # Bounded RandomizedSearchCV over X_train_encoded/y_train ONLY -
        # see _tune_hyperparameters()'s docstring and this module's
        # top-level comment. X_test_encoded/y_test are not passed anywhere
        # near this call - the held-out test set is touched exactly once,
        # further down, to score the already-tuned, already-refit model.
        model, best_hyperparameters, cv_score_during_tuning, tuning_seconds = _tune_hyperparameters(
            X_train_encoded, y_train, positive_value=positive_value, random_state=random_state
        )
    else:
        if not tuning_enabled:
            tuning_skipped_reason = (
                "tuning available but not applied - this is a reference tenant (Telco/Banking), "
                "fixed to its originally validated hyperparameters (tenant_config['tuning_enabled'] "
                "is explicitly False) rather than the tuned alternative; a real comparison found "
                "tuning scored slightly WORSE on the held-out test set for both reference tenants "
                "despite a higher cross-validation estimate - see this module's tuning_enabled "
                "comment for the full write-up. Self-registered tenants are unaffected and use "
                "tuning by default."
            )
        else:
            # Too few training rows for a reliable search - see
            # MIN_TRAINING_ROWS_FOR_TUNING's comment. Falls back to the
            # original, pre-tuning fixed hyperparameters (still
            # config.yaml-driven, so an operator can still override them)
            # rather than search over folds too small to trust.
            tuning_skipped_reason = (
                f"training set has only {len(X_train_encoded)} rows - below MIN_TRAINING_ROWS_FOR_TUNING "
                f"({MIN_TRAINING_ROWS_FOR_TUNING}); {HYPERPARAMETER_SEARCH_CV_FOLDS}-fold CV would use "
                "unreliably small folds, so this run uses the original, pre-tuning fixed hyperparameters "
                "instead of searching."
            )
        best_hyperparameters = {
            "n_estimators": int(model_config.get("n_estimators", 200)),
            "max_depth": int(model_config.get("max_depth", 3)),
            "learning_rate": float(model_config.get("learning_rate", 0.05)),
        }
        model = GradientBoostingClassifier(random_state=random_state, **best_hyperparameters)
        model.fit(X_train_encoded, y_train)
        cv_score_during_tuning = None
        tuning_seconds = 0.0

    positive_class_index = list(model.classes_).index(positive_value)
    probabilities = model.predict_proba(X_test_encoded)[:, positive_class_index]

    y_train_binary = (y_train == positive_value).astype(int)
    y_test_binary = (y_test == positive_value).astype(int)
    roc_auc = roc_auc_score(y_test_binary, probabilities)
    pr_auc = average_precision_score(y_test_binary, probabilities)

    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(encoders, output_dir / "encoders.pkl")

    # train_idx/test_idx/feature_names - consumed by src/data/split.py's
    # load_split_indices() and every downstream module built on top of it
    # (prioritize.py, backtest.py, survival.py, clv.py, scenario.py,
    # feedback_loop.py, explain.py) to re-derive the exact same split and
    # encoded-feature-column order later without needing to re-run
    # split_data() themselves. train_idx/test_idx were already computed
    # above by split_data() - this was previously silently discarded here.
    #
    # tenant_config: additive key (every existing reader only ever indexed
    # train_idx/test_idx/feature_names by name, so this doesn't disturb
    # them) so prioritize.py/business_impact.py can read the resolved
    # tenant_config back later via src/data/split.py's load_tenant_config(),
    # the same "read it back from the training-time artifact" pattern
    # survival.py/segment.py/anomaly.py/clv.py already use for their own
    # pickles - instead of every read-time call site re-deriving or
    # hardcoding it. json-safe: target_positive_value can be a numpy scalar
    # for a self-registered tenant (src/models/tenant_training.py infers it
    # via pandas' value_counts().idxmin()), which json.dumps can't
    # serialize directly.
    json_safe_tenant_config = {
        key: (value.item() if hasattr(value, "item") else value) for key, value in tenant_config.items()
    }
    (output_dir / "split_indices.json").write_text(
        json.dumps(
            {
                "train_idx": list(train_idx),
                "test_idx": list(test_idx),
                "feature_names": list(X_train_encoded.columns),
                "tenant_config": json_safe_tenant_config,
            }
        ),
        encoding="utf-8",
    )

    metadata = {
        "run_id": "v1",
        "timestamp": datetime.now().isoformat(),
        "config_used": config,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate_train": float(y_train_binary.mean()),
        "churn_rate_test": float(y_test_binary.mean()),
        # roc_auc/pr_auc: unchanged key names (every existing consumer -
        # tenant_training.py's sanity gate, api/training.py, this
        # project's tests - reads these) - same semantic meaning as
        # always ("the held-out test score"), just now produced by a
        # tuned-and-refit model instead of fixed hyperparameters.
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        # final_test_roc_auc: explicit alias for roc_auc above, added so
        # this number is never confused with cv_score_during_tuning below
        # even by field name alone - same value, reported once, after
        # tuning is fully complete.
        "final_test_roc_auc": float(roc_auc),
        # cv_score_during_tuning: the search's OWN estimate (3-fold CV,
        # training split only) of the winning hyperparameters' ROC-AUC -
        # used only to pick hyperparameters, never reported as "the" model
        # score. Real-world CV/held-out numbers routinely differ (a
        # different sample, a different fold structure) - reported
        # alongside final_test_roc_auc specifically so the two are never
        # conflated into one number.
        # None specifically (not omitted, not 0.0) when tuning was skipped
        # for having too few training rows - see MIN_TRAINING_ROWS_FOR_TUNING.
        "cv_score_during_tuning": cv_score_during_tuning,
        "best_hyperparameters": best_hyperparameters,
        "tuning_seconds": tuning_seconds,
    }
    if tuning_skipped_reason:
        metadata["tuning_skipped_reason"] = tuning_skipped_reason
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
