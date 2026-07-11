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
