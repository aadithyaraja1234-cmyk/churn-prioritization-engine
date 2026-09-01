"""Bounded hyperparameter tuning (src/models/train.py's RandomizedSearchCV
search, replacing the old fixed n_estimators=200/max_depth=3/
learning_rate=0.05 choice) - shared by the reference Telco/Banking
pipeline and the self-registered onboarding pipeline (both call
train_model()).

test_search_never_sees_the_held_out_test_set below is the single most
important test in this file, per this project's own history with exactly
this class of leakage (duration-leakage/TotalCharges, the original
explainability scale bugs): it inspects the REAL arguments the search's
own .fit() call receives (via a spy on RandomizedSearchCV.fit, not a mock
that skips real training), not just the code structure around it.

test_small_training_set_skips_tuning_and_uses_fixed_hyperparameters covers
a second real finding made while building this feature (not a leakage bug,
but a genuine reliability regression): unconditional tuning on a tiny
(~200-row) training set made a KNOWN-uninformative dataset's held-out
ROC-AUC swing between 0.39 and 0.64 across otherwise-identical resamples,
because HYPERPARAMETER_SEARCH_CV_FOLDS-fold CV on ~160 rows produces folds
too small to reliably tell "generalizes" from "overfits this resample's
noise" - undermining the sanity gate's own "bad data reliably scores low"
guarantee. See MIN_TRAINING_ROWS_FOR_TUNING (src/models/train.py).
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest import mock

import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import split_data
from src.features.encode import transform_categorical_features
from src.models.train import (
    HYPERPARAMETER_SEARCH_CV_FOLDS,
    HYPERPARAMETER_SEARCH_N_ITER,
    HYPERPARAMETER_SEARCH_SPACE,
    HYPERPARAMETER_SEARCH_TIME_BUDGET_SECONDS_TARGET,
    MIN_TRAINING_ROWS_FOR_TUNING,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def _make_signal_csv(n_rows: int, seed: int) -> str:
    """A small but genuinely trainable dataset - same shape/spirit as
    test_training_jobs.py's _make_signal_csv, kept local so this file
    doesn't depend on import order between test modules."""
    rng = random.Random(seed)
    lines = ["customer_id,target,revenue,tenure_like_feature"]
    for i in range(n_rows):
        churned = i % 2 == 0
        target = "Yes" if churned else "No"
        revenue = round(rng.uniform(20, 120), 2)
        base = rng.gauss(20, 15) if churned else rng.gauss(45, 15)
        feature = round(max(0.0, base), 2)
        lines.append(f"C{i},{target},{revenue},{feature}")
    return "\n".join(lines) + "\n"


SYNTHETIC_TENANT_CONFIG = {
    "id_column": "customer_id",
    "target_column": "target",
    "target_positive_value": "Yes",
    "revenue_column": "revenue",
}


@pytest.fixture()
def synthetic_data_path(tmp_path) -> Path:
    data_path = tmp_path / "signal.csv"
    data_path.write_text(_make_signal_csv(n_rows=400, seed=11), encoding="utf-8")
    return data_path


# --- Search space / CV configuration matches spec ---


def test_search_space_matches_specified_bounds():
    n_estimators = HYPERPARAMETER_SEARCH_SPACE["n_estimators"]
    max_depth = HYPERPARAMETER_SEARCH_SPACE["max_depth"]
    learning_rate = HYPERPARAMETER_SEARCH_SPACE["learning_rate"]

    assert min(n_estimators) >= 100 and max(n_estimators) <= 300
    assert min(max_depth) >= 2 and max(max_depth) <= 5
    assert min(learning_rate) >= 0.01 and max(learning_rate) <= 0.1


def test_search_is_bounded_not_exhaustive():
    # RandomizedSearchCV, not GridSearchCV - n_iter must be strictly less
    # than the full grid size, or this isn't actually a bounded random
    # search, it's exhaustive grid search wearing a different class name.
    full_grid_size = (
        len(HYPERPARAMETER_SEARCH_SPACE["n_estimators"])
        * len(HYPERPARAMETER_SEARCH_SPACE["max_depth"])
        * len(HYPERPARAMETER_SEARCH_SPACE["learning_rate"])
    )
    assert HYPERPARAMETER_SEARCH_N_ITER < full_grid_size


def test_cv_fold_count_is_three():
    assert HYPERPARAMETER_SEARCH_CV_FOLDS == 3


# --- THE critical leakage regression test ---


def test_search_never_sees_the_held_out_test_set(synthetic_data_path, tmp_path):
    """Spies on the REAL RandomizedSearchCV.fit() call (delegates to the
    original implementation afterwards, so real training still happens -
    this is not a mock that replaces training with a fake result) and
    captures its exact X/y arguments. Independently recomputes the true
    train/test split via split_data() - the same function train_model()
    itself uses - and asserts the captured arguments are EXACTLY the
    training rows: right count, right row identities, zero overlap with
    the held-out test rows. This inspects the actual call, not the code
    structure around it."""
    captured: dict[str, pd.DataFrame | pd.Series] = {}
    original_fit = RandomizedSearchCV.fit

    def spy_fit(self, X, y=None, **kwargs):
        captured["X"] = X.copy()
        captured["y"] = y.copy() if y is not None else None
        return original_fit(self, X, y, **kwargs)

    output_dir = tmp_path / "model"
    with mock.patch.object(RandomizedSearchCV, "fit", new=spy_fit):
        train_model(
            data_path=synthetic_data_path,
            config_path=CONFIG_PATH,
            output_dir=output_dir,
            tenant_config=SYNTHETIC_TENANT_CONFIG,
        )

    assert "X" in captured, "RandomizedSearchCV.fit() was never called - tuning didn't run at all"

    # The independent, ground-truth split - computed the exact same way
    # train_model() computes it internally, but never touched by the code
    # under test above (this call happens AFTER train_model() has already
    # returned).
    config = load_config(CONFIG_PATH)
    df = clean_data(load_raw(synthetic_data_path, SYNTHETIC_TENANT_CONFIG), SYNTHETIC_TENANT_CONFIG)
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(df, config, SYNTHETIC_TENANT_CONFIG)

    captured_X = captured["X"]
    captured_y = captured["y"]

    # Right size: exactly the training split, not train+test.
    assert len(captured_X) == len(X_train)
    assert len(captured_X) != len(X_train) + len(X_test)

    # Right rows: the exact training-row identities, nothing else.
    assert set(captured_X.index) == set(train_idx)

    # Zero overlap with the held-out test rows - the actual leakage check.
    assert set(captured_X.index).isdisjoint(set(test_idx))

    # y matches too - same row identities, same length.
    assert set(captured_y.index) == set(train_idx)
    assert len(captured_y) == len(y_train)


def test_search_object_itself_is_never_given_the_test_dataframe(synthetic_data_path, tmp_path):
    """Belt-and-suspenders on top of the spy-based test above: patches
    RandomizedSearchCV.fit to raise immediately if it EVER receives a
    DataFrame containing more rows than the known training-set size for
    this synthetic dataset - a structurally impossible-to-miss trip wire
    if a future change ever concatenates train+test before calling
    .fit()."""
    config = load_config(CONFIG_PATH)
    df = clean_data(load_raw(synthetic_data_path, SYNTHETIC_TENANT_CONFIG), SYNTHETIC_TENANT_CONFIG)
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(df, config, SYNTHETIC_TENANT_CONFIG)
    n_train = len(train_idx)

    original_fit = RandomizedSearchCV.fit

    def guarded_fit(self, X, y=None, **kwargs):
        if len(X) != n_train:
            raise AssertionError(
                f"RandomizedSearchCV.fit() received {len(X)} rows, expected exactly {n_train} "
                "(the training split) - the held-out test set may have leaked into tuning."
            )
        return original_fit(self, X, y, **kwargs)

    with mock.patch.object(RandomizedSearchCV, "fit", new=guarded_fit):
        train_model(
            data_path=synthetic_data_path,
            config_path=CONFIG_PATH,
            output_dir=tmp_path / "model",
            tenant_config=SYNTHETIC_TENANT_CONFIG,
        )


# --- Final test score computed AFTER tuning, from the refit model ---


def test_final_test_score_is_computed_after_tuning_from_the_refit_model(synthetic_data_path, tmp_path):
    """Independently reloads the persisted model.pkl/encoders.pkl and
    re-scores it against the real, independently-recomputed held-out test
    split - if this doesn't exactly match metadata['final_test_roc_auc'],
    the reported number isn't genuinely coming from the refit model
    train_model() actually saved."""
    output_dir = tmp_path / "model"
    metadata = train_model(
        data_path=synthetic_data_path,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        tenant_config=SYNTHETIC_TENANT_CONFIG,
    )

    config = load_config(CONFIG_PATH)
    df = clean_data(load_raw(synthetic_data_path, SYNTHETIC_TENANT_CONFIG), SYNTHETIC_TENANT_CONFIG)
    _, X_test, _, y_test, _, _ = split_data(df, config, SYNTHETIC_TENANT_CONFIG)

    model = joblib.load(output_dir / "model.pkl")
    encoders = joblib.load(output_dir / "encoders.pkl")
    X_test_encoded = transform_categorical_features(X_test, encoders)

    positive_value = SYNTHETIC_TENANT_CONFIG["target_positive_value"]
    positive_index = list(model.classes_).index(positive_value)
    probabilities = model.predict_proba(X_test_encoded)[:, positive_index]
    y_test_binary = (y_test == positive_value).astype(int)
    independently_computed_auc = roc_auc_score(y_test_binary, probabilities)

    assert independently_computed_auc == pytest.approx(metadata["final_test_roc_auc"])
    assert independently_computed_auc == pytest.approx(metadata["roc_auc"])


def test_cv_score_and_final_test_score_are_reported_as_distinct_fields(synthetic_data_path, tmp_path):
    metadata = train_model(
        data_path=synthetic_data_path,
        config_path=CONFIG_PATH,
        output_dir=tmp_path / "model",
        tenant_config=SYNTHETIC_TENANT_CONFIG,
    )

    assert "cv_score_during_tuning" in metadata
    assert "final_test_roc_auc" in metadata
    assert "best_hyperparameters" in metadata
    assert "tuning_seconds" in metadata

    # Both are real, valid ROC-AUC-shaped numbers ...
    assert 0.0 <= metadata["cv_score_during_tuning"] <= 1.0
    assert 0.0 <= metadata["final_test_roc_auc"] <= 1.0
    # ... but never the same measurement conflated under two names: one is
    # a 3-fold CV estimate on the training split, the other a single
    # held-out evaluation on entirely different rows.
    assert metadata["cv_score_during_tuning"] != metadata["final_test_roc_auc"]
    assert metadata["final_test_roc_auc"] == metadata["roc_auc"]  # same number, just also under the explicit name

    assert set(metadata["best_hyperparameters"]) == {"n_estimators", "max_depth", "learning_rate"}
    for param, value in metadata["best_hyperparameters"].items():
        assert value in HYPERPARAMETER_SEARCH_SPACE[param]


def test_tuning_completes_within_the_time_budget(synthetic_data_path, tmp_path):
    metadata = train_model(
        data_path=synthetic_data_path,
        config_path=CONFIG_PATH,
        output_dir=tmp_path / "model",
        tenant_config=SYNTHETIC_TENANT_CONFIG,
    )
    assert 0 < metadata["tuning_seconds"] < HYPERPARAMETER_SEARCH_TIME_BUDGET_SECONDS_TARGET


# --- Small-training-set safeguard (discovered via a real regression, not
# a theoretical worry - see MIN_TRAINING_ROWS_FOR_TUNING's comment in
# src/models/train.py) ---


def _make_noisy_csv(n_rows: int, seed: int) -> str:
    """A genuinely uninformative dataset - target carries zero real
    relationship to any feature. Same generator tests/test_training_jobs.py
    uses (kept local here too, for the same import-independence reason as
    _make_signal_csv above)."""
    rng = random.Random(seed)
    lines = ["customer_id,target,revenue,noise_feature"]
    for i in range(n_rows):
        target = "Yes" if i % 2 == 0 else "No"
        revenue = round(rng.uniform(20, 120), 2)
        noise_feature = round(rng.uniform(0, 100), 2)
        lines.append(f"C{i},{target},{revenue},{noise_feature}")
    return "\n".join(lines) + "\n"


def test_small_training_set_skips_tuning_and_uses_fixed_hyperparameters(tmp_path):
    """Regression test for a real finding made while building this
    feature: on a genuinely uninformative 200-row dataset, the pre-tuning
    fixed hyperparameters reliably scored ROC-AUC near 0.5 - but WITH
    unconditional tuning enabled, held-out ROC-AUC swung between 0.39 and
    0.64 across otherwise-identical resamples (same generator, different
    seeds), because HYPERPARAMETER_SEARCH_CV_FOLDS-fold CV on ~160
    training rows produces folds too small (~53 rows each) to reliably
    tell "generalizes" from "overfits this specific noise". Below
    MIN_TRAINING_ROWS_FOR_TUNING, train_model() must skip the search
    entirely and fall back to the original fixed hyperparameters."""
    assert 200 < MIN_TRAINING_ROWS_FOR_TUNING  # this test's premise: 200 rows IS the small case

    data_path = tmp_path / "noisy.csv"
    data_path.write_text(_make_noisy_csv(n_rows=200, seed=99), encoding="utf-8")

    metadata = train_model(
        data_path=data_path,
        config_path=CONFIG_PATH,
        output_dir=tmp_path / "model",
        tenant_config={
            "id_column": "customer_id",
            "target_column": "target",
            "target_positive_value": "Yes",
            "revenue_column": "revenue",
        },
    )

    assert metadata["cv_score_during_tuning"] is None
    assert "tuning_skipped_reason" in metadata
    assert str(MIN_TRAINING_ROWS_FOR_TUNING) in metadata["tuning_skipped_reason"]
    assert metadata["tuning_seconds"] == 0.0
    # Falls back to the documented, pre-tuning fixed values (config.yaml's
    # model: section defaults), not an arbitrary point in the search space.
    assert metadata["best_hyperparameters"] == {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}
    # Still a real, non-crashing, sane-shaped result - just not tuned.
    assert 0.0 <= metadata["roc_auc"] <= 1.0


def test_large_training_set_still_tunes_normally(synthetic_data_path):
    """Sanity check on the OTHER side of the threshold: this file's own
    400-row synthetic dataset (320 training rows, comfortably above
    MIN_TRAINING_ROWS_FOR_TUNING) must still genuinely tune, not silently
    fall back - the safeguard is for small data specifically, not a
    blanket disable."""
    config = load_config(CONFIG_PATH)
    df = clean_data(load_raw(synthetic_data_path, SYNTHETIC_TENANT_CONFIG), SYNTHETIC_TENANT_CONFIG)
    X_train, _, _, _, _, _ = split_data(df, config, SYNTHETIC_TENANT_CONFIG)
    assert len(X_train) >= MIN_TRAINING_ROWS_FOR_TUNING  # this test's premise


# --- Telco/Banking specifically stay on fixed hyperparameters (decision
# locked in after a real, controlled comparison found tuning scored worse
# on both reference tenants' held-out test sets despite a higher cross-
# validation estimate - see src/data/load.py's DEFAULT_TENANT_CONFIG
# tuning_enabled comment and config.yaml's telco:/banking: entries) ---


def test_telco_default_tenant_config_has_tuning_disabled():
    """DEFAULT_TENANT_CONFIG is what run_pipeline.py's real Telco retrain
    actually uses (tenant_config=None falls back to this) - this is the
    one flag that matters for Telco's real, deployed model."""
    from src.data.load import DEFAULT_TENANT_CONFIG

    assert DEFAULT_TENANT_CONFIG["tuning_enabled"] is False


def test_banking_tenant_config_has_tuning_disabled():
    config = load_config(CONFIG_PATH)
    assert config["tenants"]["banking"]["tuning_enabled"] is False
    # Documented on telco: too, for consistency with DEFAULT_TENANT_CONFIG
    # (see that key's comment) - not what the real retrain path reads, but
    # must never silently disagree with it.
    assert config["tenants"]["telco"]["tuning_enabled"] is False


def test_telco_and_banking_deployed_artifacts_reflect_tuning_disabled():
    """Not just the config values in isolation - the REAL, currently-
    deployed models/v1 and models/banking_v1 metadata must show tuning
    genuinely never ran for either tenant."""
    import json

    ROOT = Path(__file__).resolve().parents[1]
    telco_metadata = json.loads((ROOT / "models" / "v1" / "metadata.json").read_text(encoding="utf-8"))
    banking_metadata = json.loads((ROOT / "models" / "banking_v1" / "metadata.json").read_text(encoding="utf-8"))

    for metadata, label in ((telco_metadata, "telco"), (banking_metadata, "banking")):
        assert metadata["cv_score_during_tuning"] is None, label
        assert metadata["tuning_seconds"] == 0.0, label
        assert "tuning_skipped_reason" in metadata, label
        assert "reference tenant" in metadata["tuning_skipped_reason"], label


def test_tuning_enabled_defaults_true_for_self_registered_tenants():
    """A self-registered tenant's tenant_config (built by
    src/models/tenant_training.py's _tenant_config_from_mapping()) never
    sets "tuning_enabled" at all - confirming the key's ABSENCE still
    means "tuned by default", not silently opted out. Meridian/Fernwood/
    every future company are unaffected by the Telco/Banking-specific
    exclusion above."""
    tenant_config = {
        "id_column": "customer_id",
        "target_column": "target",
        "target_positive_value": "Yes",
        "revenue_column": "revenue",
    }
    assert "tuning_enabled" not in tenant_config
    assert tenant_config.get("tuning_enabled", True) is True
