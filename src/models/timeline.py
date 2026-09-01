"""Customer 360 timeline — real event history for one customer.

Extracted from api/main.py's customer_timeline() endpoint, which was
previously hardcoded to Telco's literal column names ("customerID",
"tenure", "Contract") the same way prioritize.py/business_impact.py/
explain.py/recommend.py/scenario.py all were before this session's
generalization pass. api/main.py's endpoint now calls into this module
for the raw-data lookup and inferred-snapshot events; the DB-backed real
event log (predictions/recommendations/scenarios) stays in api/main.py
since that part is tenant-scoped DB plumbing, not raw-column logic.

Generalized to accept tenant_config (same pattern as every other module
this session touched - read back from model_dir/split_indices.json via
src/data/split.py's load_tenant_config() when omitted). duration_column
and segment_feature_column are both OPTIONAL for a self-registered tenant
(see business_impact.py's compute_business_impact_bulk() for the same
reasoning) - a tenant missing either just doesn't get that one inferred
event, rather than crashing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.clean import clean_data
from src.data.load import load_raw
from src.data.split import load_tenant_config
from src.models.survival import resolve_segment_feature_column

# A real event history needs at least a few real, checkable customer_ids to
# be worth anything - this sanity gate exists to catch the class of bug
# this session already found twice (a raw column genuinely missing for
# this tenant), not to impose a statistical threshold the way survival's
# c-index or segmentation's silhouette do. See timeline_sanity_check().
SNAPSHOT_SAMPLE_SIZE = 20


class CustomerNotFoundError(ValueError):
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        super().__init__(f"Customer '{customer_id}' not found")


def find_customer_row(df: pd.DataFrame, customer_id: str, tenant_config: dict[str, Any]) -> pd.Series:
    id_column = tenant_config["id_column"]
    matches = df.loc[df[id_column] == customer_id]
    if matches.empty:
        raise CustomerNotFoundError(customer_id)
    return matches.iloc[0]


def build_snapshot_events(
    customer_row: pd.Series, tenant_config: dict[str, Any], now: datetime | None = None
) -> list[dict[str, Any]]:
    """Inferred signup + segment-snapshot events - not real logged events,
    both explicitly labeled data_source="inferred_from_snapshot" (matching
    the rest of this project's honesty discipline - see
    business_impact.py's module docstring for the same (a)/(b) split).

    duration_column drives the inferred signup event; segment_feature_column
    (the "Contract"-equivalent, auto-derived by survival.py - see that
    module's resolve_segment_feature_column()) drives the segment-snapshot
    event. Either or both may be unavailable for a given tenant (no
    duration mapped, or no survival model trained yet) - this returns
    however many of the two events it genuinely can, never crashing on a
    missing one."""
    now = now or datetime.now(timezone.utc)
    duration_column = tenant_config.get("duration_column")
    segment_feature_column = tenant_config.get("segment_feature_column")

    events: list[dict[str, Any]] = []

    if duration_column and duration_column in customer_row.index and pd.notna(customer_row[duration_column]):
        tenure_months = int(customer_row[duration_column])
        signup_date = now - timedelta(days=tenure_months * 30)
        events.append(
            {
                "event_type": "signup",
                "timestamp": signup_date.isoformat(),
                "data_source": "inferred_from_snapshot",
                "estimated_from_tenure": True,
                "details": {"tenure_months": tenure_months},
            }
        )

    if segment_feature_column and segment_feature_column in customer_row.index:
        events.append(
            {
                "event_type": "contract_snapshot",
                "timestamp": now.isoformat(),
                "data_source": "inferred_from_snapshot",
                "estimated_from_tenure": False,
                "details": {
                    "contract": customer_row[segment_feature_column],
                    "note": "Current value on file - not a change-detection event, no historical record exists.",
                },
            }
        )

    return events


def load_customer_snapshot(
    customer_id: str,
    model_dir: str | Path,
    data_path: str | Path,
    tenant_config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """The full raw-data half of a customer's timeline: looks the customer
    up and returns whatever inferred snapshot events are available. Raises
    CustomerNotFoundError if the id doesn't exist in this tenant's data -
    api/main.py's endpoint maps that to a 404, same as it already did
    before this was extracted."""
    model_dir = Path(model_dir)
    data_path = Path(data_path)
    if tenant_config is None:
        tenant_config = load_tenant_config(model_dir)
    # segment_feature_column is auto-derived, not part of the classifier's
    # own tenant_config - resolve it the same way business_impact.py/
    # scenario.py do (never assume tenant_config already has it set).
    resolved_tenant_config = {**tenant_config, "segment_feature_column": resolve_segment_feature_column(model_dir)}

    df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    customer_row = find_customer_row(df, customer_id, tenant_config)
    return build_snapshot_events(customer_row, resolved_tenant_config, now=now)


def timeline_sanity_check(model_dir: str | Path, data_path: str | Path, tenant_config: dict[str, Any]) -> str | None:
    """Returns a human-readable failure reason, or None if this tenant's
    raw data supports building a real customer snapshot. Structural, not
    statistical (see SNAPSHOT_SAMPLE_SIZE's comment) - matches
    docs/ADDING_A_TENANT.md's own description of this module's validation
    step ("verify clean_data/load_raw work cleanly on the new raw file and
    that logged events read back sensibly"): attempts to build a snapshot
    for a real sample of this tenant's own test-set customers and reports
    the first real failure, rather than asserting a numeric threshold this
    module has no equivalent of."""
    from src.data.split import load_split_indices

    model_dir = Path(model_dir)
    try:
        df = clean_data(load_raw(data_path, tenant_config), tenant_config)
    except Exception as exc:  # noqa: BLE001 - report as a gate failure, not a crash
        return f"raw data failed to load/clean: {exc}"

    id_column = tenant_config["id_column"]
    try:
        _, test_idx = load_split_indices(model_dir)
    except FileNotFoundError as exc:
        return f"no split_indices.json for this tenant yet: {exc}"

    sample_idx = test_idx[:SNAPSHOT_SAMPLE_SIZE]
    resolved_tenant_config = {**tenant_config, "segment_feature_column": resolve_segment_feature_column(model_dir)}
    for idx in sample_idx:
        customer_id = df.loc[idx, id_column]
        try:
            customer_row = find_customer_row(df, customer_id, tenant_config)
            build_snapshot_events(customer_row, resolved_tenant_config)
        except Exception as exc:  # noqa: BLE001
            return f"failed to build a snapshot for a real test-set customer ('{customer_id}'): {exc}"
    return None
