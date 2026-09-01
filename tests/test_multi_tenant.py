import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.data.clean import clean_data
from src.data.load import DEFAULT_TENANT_CONFIG, load_raw
from src.data.split import split_data
from src.models.train import train_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def test_both_tenant_configs_load_without_error():
    config = load_config(CONFIG_PATH)
    tenants = config["tenants"]

    assert "telco" in tenants
    assert "banking" in tenants

    required_keys = {"data_path", "target_column", "target_positive_value", "revenue_column", "id_column"}
    for tenant_name in ("telco", "banking"):
        tenant_config = tenants[tenant_name]
        assert required_keys.issubset(tenant_config.keys())
        assert (ROOT / tenant_config["data_path"]).exists()


def test_v1_model_metadata_matches_the_documented_baseline():
    """models/v1/metadata.json's numbers are Telco's official, documented
    baseline as of this session's hyperparameter-tuning investigation:
    0.8421956134232348 ROC-AUC / 0.6607384136110945 PR-AUC - the ORIGINAL
    hand-picked hyperparameters (n_estimators=200, max_depth=3,
    learning_rate=0.05; tuning explicitly disabled - src/models/train.py's
    tuning_enabled), computed under the CURRENT, correct customerID-based
    split.

    This replaces an earlier-cited 0.8471014492753624/0.6686762670776025,
    which turned out to be unreproducible: it was computed under a
    since-fixed, position-dependent split, and models/v1 was never
    actually regenerated after that fix landed until this investigation
    retrained it - see test_v1_split_indices_match_a_fresh_split_data_call
    below, the permanent guard against that exact staleness recurring
    silently. A real, controlled comparison (tests/test_hyperparameter_
    tuning.py) also found automated tuning made this number slightly
    WORSE (0.8400), which is why tuning stays off for this tenant - see
    src/data/load.py's DEFAULT_TENANT_CONFIG tuning_enabled comment."""
    metadata_path = ROOT / "models" / "v1" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["roc_auc"] == pytest.approx(0.8421956134232348)
    assert metadata["pr_auc"] == pytest.approx(0.6607384136110945)
    assert metadata["best_hyperparameters"] == {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}

    # final_test_roc_auc is the SAME held-out number as roc_auc, just
    # under an unambiguous name - never conflated with the CV estimate.
    assert metadata["final_test_roc_auc"] == metadata["roc_auc"]
    # Tuning genuinely never ran for this tenant - cv_score_during_tuning
    # is None (not just "different"), and the reason is recorded, not
    # silent.
    assert metadata["cv_score_during_tuning"] is None
    assert "tuning_skipped_reason" in metadata


def test_v1_split_indices_match_a_fresh_split_data_call():
    """Permanent regression guard for the exact failure this session's
    Telco hyperparameter-tuning investigation uncovered: models/v1's
    on-disk split_indices.json silently drifted out of sync with
    src/data/split.py's actual (customerID-based, deterministic) split
    algorithm - apparently for about a month - before anyone noticed,
    because nothing ever re-checked the deployed artifact against a fresh
    computation. Every number this project has ever cited for "Telco's
    test set" (classifier metrics, the priority-ranking correlation,
    backtest lift, specific customer-ID examples) is only trustworthy if
    this holds. If a future change to split.py/clean.py/telco.csv ever
    makes this fail, it means models/v1 needs to be regenerated
    (run_pipeline.py --output-dir models/v1) before trusting any
    test-set-derived number again - this test exists so that's caught
    immediately, not discovered a month later by accident."""
    split_info = json.loads((ROOT / "models" / "v1" / "split_indices.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / "data" / "raw" / "telco.csv")
    on_disk_test_ids = set(df.loc[split_info["test_idx"], "customerID"])
    on_disk_train_ids = set(df.loc[split_info["train_idx"], "customerID"])

    config = load_config(CONFIG_PATH)
    fresh_df = clean_data(load_raw(ROOT / "data" / "raw" / "telco.csv", DEFAULT_TENANT_CONFIG), DEFAULT_TENANT_CONFIG)
    _, _, _, _, fresh_train_idx, fresh_test_idx = split_data(fresh_df, config, DEFAULT_TENANT_CONFIG)
    fresh_test_ids = set(fresh_df.loc[fresh_test_idx, "customerID"])
    fresh_train_ids = set(fresh_df.loc[fresh_train_idx, "customerID"])

    assert on_disk_test_ids == fresh_test_ids, (
        "models/v1/split_indices.json's test-set customer IDs no longer match what "
        "split_data() produces fresh - the deployed model_dir has silently drifted from the "
        "current split algorithm/data. Regenerate models/v1 (run_pipeline.py --output-dir "
        "models/v1) before trusting any test-set-derived number (backtest, correlation, "
        "customer examples, etc.)."
    )
    assert on_disk_train_ids == fresh_train_ids


def test_split_fix_produces_different_but_valid_split(tmp_path):
    """config.yaml's telco: entry has tuning_enabled: false too (same
    decision as DEFAULT_TENANT_CONFIG - see that key's comment in
    src/data/load.py), so a fresh retrain through THIS tenant_config is
    fully deterministic (same data, same seed, same fixed hyperparameters)
    and reproduces models/v1's own documented baseline exactly - not an
    approximate range. This used to only assert a sanity range because the
    frozen models/v1 predated the customerID-based split fix and couldn't
    be reproduced; that's no longer true (see
    test_v1_model_metadata_matches_the_documented_baseline)."""
    config = load_config(CONFIG_PATH)
    telco_config = config["tenants"]["telco"]

    result = train_model(
        ROOT / telco_config["data_path"],
        CONFIG_PATH,
        output_dir=tmp_path / "telco_resplit_check",
        tenant_config=telco_config,
    )

    assert result["roc_auc"] == pytest.approx(0.8421956134232348)
    assert result["pr_auc"] == pytest.approx(0.6607384136110945)


def test_banking_tenant_beats_random_baseline(tmp_path):
    config = load_config(CONFIG_PATH)
    banking_config = config["tenants"]["banking"]

    result = train_model(
        ROOT / banking_config["data_path"],
        CONFIG_PATH,
        output_dir=tmp_path / "banking_check",
        tenant_config=banking_config,
    )

    assert result["roc_auc"] > 0.65
