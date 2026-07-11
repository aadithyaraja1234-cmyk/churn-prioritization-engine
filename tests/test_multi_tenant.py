import json
from pathlib import Path

from src.config import load_config
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


def test_v1_model_metadata_untouched_by_refactor():
    metadata_path = ROOT / "models" / "v1" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["roc_auc"] == 0.8471014492753624
    assert metadata["pr_auc"] == 0.6686762670776025


def test_split_fix_produces_different_but_valid_split(tmp_path):
    config = load_config(CONFIG_PATH)
    telco_config = config["tenants"]["telco"]

    result = train_model(
        ROOT / telco_config["data_path"],
        CONFIG_PATH,
        output_dir=tmp_path / "telco_resplit_check",
        tenant_config=telco_config,
    )

    # Fresh retrains now go through the customerID-based split fix, so they
    # will not reproduce models/v1's frozen 0.847 (that was computed under the
    # old, position-dependent split). This just checks the fixed split still
    # yields a sane model, without comparing to the old number.
    assert 0.75 < result["roc_auc"] < 0.90


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
