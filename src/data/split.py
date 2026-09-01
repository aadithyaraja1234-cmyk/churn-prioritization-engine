from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load import DEFAULT_TENANT_CONFIG


def split_data(
    df: pd.DataFrame,
    config: dict[str, Any],
    tenant_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[int], list[int]]:
    tenant_config = tenant_config or DEFAULT_TENANT_CONFIG
    id_column = tenant_config["id_column"]
    target_column = tenant_config["target_column"]

    split_config = config.get("split", {})
    test_size = float(split_config.get("test_size", 0.2))
    seed = int(split_config.get("seed", 42))

    # Group by id_column sorted order (not physical row order) so that the
    # resulting split depends only on which records exist, never on how
    # the input rows happen to be ordered.
    id_to_pos = dict(zip(df[id_column], df.index))
    ordered = df[[id_column, target_column]].sort_values(id_column)

    groups: dict[str, list] = {}
    for record_id, label in zip(ordered[id_column], ordered[target_column]):
        groups.setdefault(str(label), []).append(record_id)

    rng = random.Random(seed)
    train_ids: list = []
    test_ids: list = []

    for label in sorted(groups):
        label_ids = groups[label]
        shuffled = list(label_ids)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_size))) if len(shuffled) > 1 else 0
        if n_test >= len(shuffled):
            n_test = max(1, len(shuffled) // 2)
        test_ids.extend(shuffled[:n_test])
        train_ids.extend(shuffled[n_test:])

    train_idx = sorted(id_to_pos[rid] for rid in train_ids)
    test_idx = sorted(id_to_pos[rid] for rid in test_ids)

    X_train = df.loc[train_idx].drop(columns=[id_column, target_column]).copy()
    X_test = df.loc[test_idx].drop(columns=[id_column, target_column]).copy()
    y_train = df.loc[train_idx, target_column].copy()
    y_test = df.loc[test_idx, target_column].copy()

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def load_split_indices(model_dir: str | Path) -> tuple[list[int], list[int]]:
    split_info_path = Path(model_dir) / "split_indices.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"Split index file not found: {split_info_path}")

    with split_info_path.open("r", encoding="utf-8") as handle:
        split_info = json.load(handle)
    return split_info["train_idx"], split_info["test_idx"]


def load_tenant_config(model_dir: str | Path) -> dict[str, Any]:
    """Reads back the tenant_config train_model() now saves into
    split_indices.json at training time (additive key alongside the
    existing train_idx/test_idx/feature_names) - the same "read the
    resolved config back from the training-time artifact instead of
    re-passing it at every call site" pattern survival.py/segment.py/
    anomaly.py/clv.py already use for their own pickles (each stores
    tenant_config inside its own .pkl and reads it back via
    saved.get("tenant_config", DEFAULT_TENANT_CONFIG)).

    Falls back to DEFAULT_TENANT_CONFIG (Telco's shape) when the key is
    absent - true for every split_indices.json written before this key
    existed, including Telco/Banking's already-committed, permanently
    guarded models/v1 and models/banking_v1 artifacts, and any
    self-registered tenant trained before this generalization pass. This
    is not a behavior change for any of them: DEFAULT_TENANT_CONFIG is
    exactly the literal Telco column names every caller of this was
    already hardcoded to before this function existed.
    """
    split_info_path = Path(model_dir) / "split_indices.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"Split index file not found: {split_info_path}")

    with split_info_path.open("r", encoding="utf-8") as handle:
        split_info = json.load(handle)
    return split_info.get("tenant_config", DEFAULT_TENANT_CONFIG)
