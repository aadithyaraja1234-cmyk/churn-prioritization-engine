"""AI-assisted column-mapping refinement for self-service onboarding, Stage 1b.

src.onboarding.schema_fields.suggest_mapping() is pure name-similarity
matching - fast, deterministic, and free, but it only ever looks at the
column NAME, never its values or the surrounding schema, so it reliably
misses verbose/human-readable names ("Client Reference Number", "Left
Firm") or genuinely ambiguous ones and leaves them at role="ignore" for a
human to fix by hand every time (see data/live_demo_v2/README.md's
Pre-check #1: only 1-3 of 14-15 columns per file auto-suggested correctly
for verbose/camelCase/mixed-casing real-world naming - the exact "most
features going to ignore" experience this module exists to reduce).

This module adds a SECOND, OPTIONAL pass over whatever suggest_mapping()
couldn't resolve (still role="ignore") - reusing this project's existing
Gemini integration (see src/copilot/agent.py) - giving the model the full
column list, a few real sample values, and which roles are already
confidently claimed, and asking it to classify just the unresolved ones.

Same "suggest, never auto-apply" contract as suggest_mapping() - a human
still confirms every field via the mapping-step dropdown before
/api/onboarding/validate ever acts on it (see schema_fields.py's own
docstring); this module only ever changes what gets pre-selected in that
dropdown, never what training actually uses.

DEGRADE POLICY: unlike src/copilot/agent.py (which raises
CopilotUnavailableError and surfaces a 503 to the caller - chat is an
optional feature), this must NEVER break the upload flow, which every
tenant depends on regardless of whether GEMINI_API_KEY is set. Any
failure - missing key, a real Gemini API error, a malformed/unparseable
response - is caught here, and the caller gets back suggest_mapping()'s
original suggestions, unmodified, silently. The one thing this module
always adds, even on total failure, is stamping every suggestion's
`source` field ("name_match" or "none") so the mapping UI can show which
arm of the pipeline actually produced each suggestion - see
api/onboarding.py.

COST: only ever calls Gemini when at least one column is still
role="ignore" after suggest_mapping() - a well-matched file (Telco-shaped
column names) costs nothing extra, same as today.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src.onboarding.schema_fields import VALID_ROLES

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# "duration_leakage_column" is deliberately excluded from what the AI may
# assign: that role is a VALUE-correlation signal (see
# suggest_duration_leakage_column()'s own docstring - it checks correlation
# against duration*revenue directly), not something a column name or a
# handful of sample values can honestly indicate. Offering it here would
# just be a plausible-sounding hallucination risk for zero real benefit.
AI_ASSIGNABLE_ROLES = tuple(role for role in VALID_ROLES if role != "duration_leakage_column")

# Roles a tenant_config can only ever have ONE column claim (see
# src/models/tenant_training.py's _tenant_config_from_mapping(), which
# raises TrainingDataError otherwise). If the name-matcher already
# confidently claimed one of these, the AI is told so in the prompt AND
# barred again here in code - a prompt instruction alone is never treated
# as a guarantee anywhere else in this project (see upload_security.py's
# own "reject-and-explain, don't just trust" policy) and isn't here either.
SINGLETON_ROLES = ("customer_id", "target", "revenue", "duration", "clv")

MAX_SAMPLE_ROWS = 5

_client: "genai.Client | None" = None


def _get_client() -> "genai.Client | None":
    """Returns None (never raises) when GEMINI_API_KEY isn't set - the
    caller's job is to fall back silently, not to surface an error for a
    feature every upload works fine without."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    _client = genai.Client(api_key=api_key)
    return _client


def reset_client_for_testing() -> None:
    """Test-only: clears the cached client so a test can force
    _get_client() to re-check the environment, mirroring
    src/copilot/agent.py's own reset_client_for_testing()."""
    global _client
    _client = None


def _build_prompt(
    unresolved_columns: list[str],
    all_columns: list[str],
    sample_rows: list[dict[str, Any]],
    claimed_roles: dict[str, str],
    offerable_roles: list[str],
) -> str:
    samples = sample_rows[:MAX_SAMPLE_ROWS]
    sample_lines = []
    for column in unresolved_columns:
        values = [str(row.get(column, "")) for row in samples]
        sample_lines.append(f'  - "{column}": example values {values}')

    claimed_lines = (
        "\n".join(f'  - "{col}" is already confidently mapped to "{role}"' for col, role in claimed_roles.items())
        or "  (none yet)"
    )

    return (
        "You are helping map columns of a customer-churn CSV upload to this system's known roles, "
        "for a human to review and confirm - you are NOT making the final decision.\n\n"
        f"All columns in this file: {all_columns}\n\n"
        f"Roles already confidently assigned by name-matching (do NOT reuse these for a different "
        f"column - each may only be claimed once):\n{claimed_lines}\n\n"
        f"The following columns could NOT be confidently matched by name alone and need your best "
        f"guess, based on their name and a few real sample values:\n" + "\n".join(sample_lines) + "\n\n"
        f"Valid roles you may assign: {list(offerable_roles)}\n"
        "Role meanings: customer_id = a unique per-row identifier; target = the churn/cancellation "
        "outcome label (exactly one column, only if not already claimed above); revenue = a "
        "recurring charge/fee/price amount; duration = tenure/time-as-customer; clv = an existing "
        "lifetime-value estimate already in the data; feature = any other column with real "
        "predictive signal (demographics, usage, plan/tier, support history, etc.); ignore = an "
        "identifier-like, free-text, or clearly irrelevant column with no real signal.\n\n"
        "For EACH unresolved column listed above, return your best guess. If genuinely nothing "
        "fits, use \"ignore\" rather than forcing a bad match - a wrong confident-looking guess is "
        "worse than an honest \"ignore\" a human then has to fix anyway."
    )


def _response_schema(unresolved_columns: list[str], offerable_roles: list[str]) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "enum": unresolved_columns},
                "suggested_role": {"type": "string", "enum": offerable_roles},
                "reasoning": {"type": "string"},
            },
            "required": ["column", "suggested_role"],
        },
    }


def refine_mapping_with_ai(
    columns: list[str],
    sample_rows: list[dict[str, Any]],
    suggestions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Takes suggest_mapping()'s (optionally suggest_duration_leakage_
    column()-refined) output and returns a copy with every entry stamped
    `source` ("name_match" or "none"), and - only if GEMINI_API_KEY is set
    AND at least one column is still role="ignore" - an AI-assisted guess
    for those columns (stamped `source`="ai", plus `ai_reasoning`). Falls
    back to the unmodified input (still stamped) on any failure; never
    raises."""
    stamped: dict[str, dict[str, Any]] = {
        column: {**info, "source": "name_match" if info.get("matched_known_field") else "none"}
        for column, info in suggestions.items()
    }

    unresolved = [column for column in columns if stamped.get(column, {}).get("suggested_role") == "ignore"]
    if not unresolved:
        return stamped

    client = _get_client()
    if client is None:
        return stamped

    claimed_roles = {
        column: info["suggested_role"]
        for column, info in stamped.items()
        if info["suggested_role"] in SINGLETON_ROLES
    }
    already_claimed = set(claimed_roles.values())
    offerable_roles = [role for role in AI_ASSIGNABLE_ROLES if role not in already_claimed]

    prompt = _build_prompt(unresolved, columns, sample_rows, claimed_roles, offerable_roles)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_response_schema(unresolved, offerable_roles),
                temperature=0,
            ),
        )
        parsed = json.loads(response.text or "[]")
    except (errors.APIError, TimeoutError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
        return stamped

    if not isinstance(parsed, list):
        return stamped

    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        column = entry.get("column")
        role = entry.get("suggested_role")
        if column not in unresolved or role not in offerable_roles:
            continue
        if role in SINGLETON_ROLES:
            if role in already_claimed:
                continue
            already_claimed.add(role)
        reasoning = entry.get("reasoning")
        stamped[column] = {
            "suggested_role": role,
            "matched_known_field": None,
            "confidence": None,
            "source": "ai",
            "ai_reasoning": reasoning if isinstance(reasoning, str) else None,
        }

    return stamped
