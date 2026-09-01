"""Data-sufficiency validation for a newly uploaded, user-mapped CSV.

Structural "checks" (id/target mapped, id unique, target has both classes)
are hard blockers - there is no honest way to eventually train or score
without them. row_count and feature_richness are graded WARNINGS, not hard
blocks: thin data can still be explored, just with an explicit "results may
be less reliable" acknowledgment, matching this project's honesty-forward
pattern used everywhere else (e.g. src/tenant_registry.py's unavailable_
response, never a silent guess).

Thresholds are grounded in this system's own real precedent, not arbitrary
guesses:
  - MIN_RECOMMENDED_ROWS = 5,634 - the exact training-row count
    models/v1/metadata.json's n_train reports for this project's own
    validated Telco model (roc_auc 0.8422 - see models/v1/metadata.json,
    the documented reference baseline as of this project's hyperparameter-
    tuning investigation; n_train itself is unaffected by that investigation
    - it's fixed by the train/test split, not by hyperparameters). Below
    this, we're asking for more data than this system has ever actually
    proven works, not guessing at a round number.
  - MIN_RECOMMENDED_FEATURES = 8 - Telco's model uses 19 raw features
    total (models/v1/split_indices.json), but a curated subset of 8 -
    contract terms, tenure, billing amount, and similar (see
    src/onboarding/schema_fields.py's CORE_SIGNAL_FIELDS) - are the ones
    most responsible for that model's validated accuracy. 8 is the
    practical floor for "enough signal to be worth scoring", not the
    ceiling of what's useful.

A low feature_richness_score has two genuinely different causes, and the
guidance distinguishes them rather than collapsing both into one generic
count:
  - missing_core_fields: a known-signal column isn't present under any
    recognizable name at all - the guidance is "consider adding X".
  - unusable_mapped_columns: the column IS present and mapped, but its
    VALUES don't carry signal (too sparse, or a stuck constant) - the
    guidance is "this specific column needs fixing", not "add more
    columns". Without this distinction, a file with all the right column
    names but broken values (e.g. a half-migrated ETL pipeline) gets the
    same vague "N usable columns" message as a file that's actually
    missing those fields outright, even though the fix is completely
    different in each case.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.onboarding.schema_fields import CORE_SIGNAL_FIELDS, VALID_ROLES, suggest_mapping

MIN_RECOMMENDED_ROWS = 5634  # models/v1/metadata.json: n_train
MIN_RECOMMENDED_FEATURES = 8  # len(CORE_SIGNAL_FIELDS)
MIN_NON_NULL_RATE = 0.8  # a mapped feature column must be at least 80% populated to count as "usable"


def _feature_usability_issue(series: pd.Series) -> str | None:
    """Returns a short, human-readable reason a mapped feature column isn't
    usable (for unusable_mapped_columns below), or None if it's fine. Two
    distinct failure modes, reported distinctly rather than collapsed into
    one boolean - a column that's mostly null and a column that's a stuck
    constant look identical to a bare pass/fail check but need different
    fixes from the uploader."""
    non_null = series.dropna()
    non_null_rate = len(non_null) / max(len(series), 1)
    if non_null_rate < MIN_NON_NULL_RATE:
        return f"{round((1 - non_null_rate) * 100)}% null"
    if non_null.nunique() <= 1:  # constant column - carries no predictive signal
        return "constant value"
    return None


def validate_upload(df: pd.DataFrame, column_mapping: dict[str, str]) -> dict[str, Any]:
    """column_mapping is {csv_column_name: role}, role one of VALID_ROLES -
    the USER-CONFIRMED mapping (never auto-applied suggestions - see
    schema_fields.suggest_mapping's docstring)."""
    unknown_roles = {role for role in column_mapping.values() if role not in VALID_ROLES}
    if unknown_roles:
        raise ValueError(f"Unknown role(s) in column_mapping: {sorted(unknown_roles)}; must be one of {VALID_ROLES}")
    unknown_columns = set(column_mapping) - set(df.columns)
    if unknown_columns:
        raise ValueError(f"column_mapping references column(s) not in the uploaded file: {sorted(unknown_columns)}")

    id_columns = [c for c, r in column_mapping.items() if r == "customer_id"]
    target_columns = [c for c, r in column_mapping.items() if r == "target"]
    feature_columns = [c for c, r in column_mapping.items() if r == "feature"]

    checks: list[dict[str, Any]] = []

    id_mapped = len(id_columns) == 1
    checks.append(
        {
            "name": "id_column_mapped",
            "blocking": True,
            "passed": id_mapped,
            "detail": "Exactly one column must be mapped to customer_id."
            if not id_mapped
            else f"'{id_columns[0]}' mapped as customer_id.",
        }
    )

    id_unique = id_mapped and df[id_columns[0]].notna().all() and df[id_columns[0]].is_unique
    checks.append(
        {
            "name": "id_column_unique_and_non_null",
            "blocking": True,
            "passed": bool(id_unique) if id_mapped else False,
            "detail": "The customer_id column must be non-null and unique for every row."
            if not id_unique
            else "customer_id values are unique and non-null.",
        }
    )

    target_mapped = len(target_columns) == 1
    checks.append(
        {
            "name": "target_column_mapped",
            "blocking": True,
            "passed": target_mapped,
            "detail": "Exactly one column must be mapped to target (the churn/outcome label)."
            if not target_mapped
            else f"'{target_columns[0]}' mapped as target.",
        }
    )

    target_has_two_classes = target_mapped and df[target_columns[0]].nunique(dropna=True) >= 2
    checks.append(
        {
            "name": "target_has_two_classes",
            "blocking": True,
            "passed": bool(target_has_two_classes) if target_mapped else False,
            "detail": "The target column must contain at least two distinct outcome values."
            if not target_has_two_classes
            else "target column has at least two distinct values.",
        }
    )

    usable_feature_columns: list[str] = []
    unusable_mapped_columns: list[str] = []
    for c in feature_columns:
        issue = _feature_usability_issue(df[c])
        if issue is None:
            usable_feature_columns.append(c)
        else:
            unusable_mapped_columns.append(f"{c} ({issue})")
    feature_richness_score = len(usable_feature_columns)

    row_count = len(df)
    row_count_sufficient = row_count >= MIN_RECOMMENDED_ROWS
    feature_richness_sufficient = feature_richness_score >= MIN_RECOMMENDED_FEATURES

    # Which of this system's own core-signal fields does the upload appear
    # to be missing? Re-run name-similarity suggestion over the *mapped
    # feature columns only*, so this reflects what the user actually kept,
    # not just what was uploaded (a column mapped to "ignore" doesn't count
    # as covering that field).
    matched_known_fields = set()
    if feature_columns:
        suggestions = suggest_mapping(feature_columns)
        for suggestion in suggestions.values():
            if suggestion["matched_known_field"]:
                matched_known_fields.add(suggestion["matched_known_field"])
    missing_core_fields = [field for field in CORE_SIGNAL_FIELDS if field not in matched_known_fields]

    sufficiency_warning = not (row_count_sufficient and feature_richness_sufficient)
    guidance_parts: list[str] = []
    if not feature_richness_sufficient:
        guidance_parts.append(
            f"Your data has only {feature_richness_score} usable mapped predictive column"
            f"{'s' if feature_richness_score != 1 else ''}; models trained on this system "
            f"typically use {MIN_RECOMMENDED_FEATURES}+ features (contract type, tenure, "
            "monthly charges, and similar) for reliable accuracy."
        )
        if unusable_mapped_columns:
            # Distinct from missing_core_fields below: these columns ARE
            # present (by name) but their VALUES don't carry signal (a
            # constant, or too sparse) - "fix this column" is different
            # advice from "you don't have this column at all".
            guidance_parts.append(
                "These mapped columns exist but aren't usable as-is: "
                + ", ".join(unusable_mapped_columns)
                + "."
            )
        if missing_core_fields:
            guidance_parts.append("Consider adding: " + ", ".join(missing_core_fields) + ".")
    if not row_count_sufficient:
        guidance_parts.append(
            f"Your upload has {row_count:,} rows. This system's own reference model (Telco) "
            f"was validated on {MIN_RECOMMENDED_ROWS:,} training rows - results from a smaller "
            "dataset may be less statistically reliable."
        )

    data_sufficiency = {
        "row_count": row_count,
        "min_recommended_row_count": MIN_RECOMMENDED_ROWS,
        "row_count_sufficient": row_count_sufficient,
        "feature_richness_score": feature_richness_score,
        "min_recommended_features": MIN_RECOMMENDED_FEATURES,
        "feature_richness_sufficient": feature_richness_sufficient,
        "missing_core_fields": missing_core_fields,
        "unusable_mapped_columns": unusable_mapped_columns,
        "warning": sufficiency_warning,
        "guidance": " ".join(guidance_parts) if guidance_parts else None,
        "requires_acknowledgment": sufficiency_warning,
    }

    can_proceed = all(check["passed"] for check in checks if check["blocking"])

    return {
        "checks": checks,
        "data_sufficiency": data_sufficiency,
        "can_proceed": can_proceed,
    }
