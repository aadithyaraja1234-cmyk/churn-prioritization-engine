"""Known-schema-field dictionary used to SUGGEST a column mapping for a
newly uploaded CSV - never to auto-apply one. A human always confirms (or
overrides) every suggestion via a dropdown before it takes effect (see
api/onboarding.py's /validate endpoint, which only ever acts on the
user-confirmed mapping it receives, not on these suggestions directly).

Grounded in this system's own two real precedents, not a generic guess:
  - Telco (models/v1/split_indices.json's 19 feature_names + config.yaml's
    id/target/revenue columns) - the fully-validated reference tenant.
  - Banking (models/banking_v1/split_indices.json's 12 feature_names) - the
    second real tenant this system has actually onboarded.

CORE_SIGNAL fields are the subset most responsible for Telco's validated
model quality (roc_auc 0.8422 - see models/v1/metadata.json, the
documented reference baseline as of this project's hyperparameter-tuning
investigation) - contract terms, tenure, and billing amount chief among
them. These are what data_sufficiency guidance in
src/onboarding/validation.py points to when a user's upload is thin on
predictive columns.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

import pandas as pd

# Suggestions below this normalized-name-similarity score are not offered at
# all (the column is left as "ignore" for the user to map manually) - chosen
# so near-exact synonyms ("MonthlyCharge" vs "MonthlyCharges") match but
# unrelated short names don't collide by coincidence.
SUGGESTION_THRESHOLD = 0.72

# suggest_duration_leakage_column()'s correlation threshold, matching
# survival.py's own DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD (see that
# module's docstring) - the same "how correlated is too correlated" bar,
# applied at suggestion time instead of training time.
DURATION_LEAKAGE_SUGGESTION_CORRELATION_THRESHOLD = 0.9

VALID_ROLES = (
    "customer_id",
    "target",
    "revenue",
    "feature",
    "duration",
    "clv",
    "duration_leakage_column",
    "ignore",
)

KNOWN_SCHEMA_FIELDS: list[dict[str, Any]] = [
    {
        "field": "customerID",
        "role": "customer_id",
        "core_signal": False,
        "aliases": [
            "customerid", "customer_id", "clientid", "client_id", "userid", "user_id", "custid", "id",
            # Broadened from real onboarded tenants' actual id column names
            # (subscriber_id, account_id, member_id, viewer_id, etc.) - the
            # generic "id" alias alone under-matches these via difflib
            # ratio once a longer owner-noun prefix is added.
            "subscriber_id", "account_id", "member_id", "viewer_id", "acct_id", "policy_id", "resident_id",
        ],
    },
    {
        "field": "Churn",
        "role": "target",
        "core_signal": False,
        "aliases": [
            "churn", "churned", "exited", "is_churn", "is_churned", "target", "label", "attrition",
            # Broadened from real onboarded tenants' actual target column
            # names - "cancelled"/"unsubscribed"/"terminated"/"departed"/
            # "lapsed" all mean the same thing as "churned" but share none
            # of its letters, so difflib similarity alone never catches
            # them; this project has now seen every one of these as a
            # real tenant's real target column name.
            "cancelled", "canceled", "unsubscribed", "terminated", "departed", "lapsed", "left",
            "subscription_lapsed", "left_firm", "left_us",
        ],
    },
    {
        "field": "MonthlyCharges",
        "role": "revenue",
        "core_signal": False,
        "aliases": [
            "monthlycharges", "monthly_charge", "monthly_charges", "revenue", "monthly_revenue", "balance",
            "amount", "billing_amount", "monthly_bill",
        ],
    },
    # --- Optional module roles: unlike customer_id/target/revenue (always
    # required to train at all), these two only matter if a tenant also
    # wants src/models/survival.py (duration) or src/models/clv.py (clv) to
    # produce real results - see src/data/load.py's DEFAULT_TENANT_CONFIG
    # duration_column/clv_column, which is what those modules actually read.
    # A CSV with neither mapped can still train the core classifier and run
    # segment.py/anomaly.py just fine.
    #
    # A tenure-named column now suggests role="duration" directly (see the
    # "tenure" entry below) - reversed from an earlier version of this file
    # that deliberately kept "tenure" suggesting role="feature" only,
    # requiring a human to manually re-pick "duration" from the dropdown
    # every time. That earlier caution turned out to cost real tenants a
    # real module: cascade-retails onboarded with "tenure_months" left as
    # a plain feature, silently never training Survival Analysis at all,
    # purely because nothing ever suggested the more useful role - not
    # because the data couldn't support it (a corrected re-mapping trained
    # Survival cleanly, c-index 0.844). A duration-mapped column is NOT
    # dropped as a classifier feature either way (src/models/
    # tenant_training.py's _filtered_training_frame() keeps duration/clv-
    # mapped columns alongside ordinary "feature" ones, mirroring Telco's
    # own real tenure/TotalCharges dual role), so there is no real
    # downside to suggesting the more complete role by default - a human
    # can still always override it back to plain "feature" from the
    # dropdown if that's genuinely what they want.
    {
        "field": "duration_since_signup",
        "role": "duration",
        "core_signal": False,
        "aliases": [
            "duration", "durationmonths", "duration_months", "subscription_duration",
            "membership_duration", "time_as_customer", "account_duration",
            "customer_duration", "relationship_length",
            # NOTE: "months_subscribed" was removed from here - it scored a
            # dangerously high 0.79 name-similarity against "unsubscribed"
            # (a genuinely plausible TARGET column name: both share the
            # "subscri..." stem), which suggested role="duration" for a
            # customer's actual churn label - found live, on a real
            # tenant's real column set. Not worth the collision risk for
            # one alias when "months_active"/"tenure_months" (on the
            # "tenure" entry below) already cover this same naming pattern.
        ],
    },
    {
        "field": "CLTV",
        "role": "clv",
        "core_signal": False,
        "aliases": [
            "cltv", "clv", "customerlifetimevalue", "customer_lifetime_value", "lifetimevalue",
            "lifetime_value", "ltv", "predicted_clv", "clv_estimate", "lifetime_value_estimate",
            # Broadened after a real miss: "subscriber_value_estimate"
            # (cascade-retails' real CLV-shaped column) scored only 0.68
            # against the aliases above - just under SUGGESTION_THRESHOLD -
            # because "subscriber_"/"account_"/etc. prefixes dilute a
            # difflib ratio even when the meaningful "_value_estimate"
            # suffix matches exactly. These cover the same "<owner>_value_
            # estimate" pattern this project's own live-demo generator
            # scripts already use across multiple companies (Aurora/
            # Palisade/Northgate/Ferrous Vale/Cascade), which is exactly
            # the kind of realistic real-world variation a suggester this
            # small has to be tested against, not just Telco's own naming.
            "subscriber_value_estimate", "member_value_estimate", "membership_value_estimate",
            "account_value_estimate", "portfolio_value_estimate", "customer_value_estimate",
            "projected_lifetime_value", "estimated_value", "value_estimate", "est_lifetime_val",
        ],
    },
    # --- Core signal features: the fields most responsible for Telco's
    # validated model quality; missing these is what data-sufficiency
    # guidance calls out by name. ---
    {
        "field": "tenure",
        "role": "duration",
        "core_signal": True,
        "aliases": ["tenure", "tenuremonths", "tenure_months", "months_active", "account_age", "customer_tenure"],
    },
    {
        "field": "Contract",
        "role": "feature",
        "core_signal": True,
        "aliases": ["contract", "contracttype", "contract_type", "plan_type", "subscription_type", "term"],
    },
    {
        "field": "PaymentMethod",
        "role": "feature",
        "core_signal": True,
        "aliases": ["paymentmethod", "payment_method", "payment_type", "billing_method"],
    },
    {
        "field": "InternetService",
        "role": "feature",
        "core_signal": True,
        "aliases": ["internetservice", "internet_service", "internet_type", "service_type"],
    },
    {
        "field": "OnlineSecurity",
        "role": "feature",
        "core_signal": True,
        "aliases": ["onlinesecurity", "online_security", "security_addon"],
    },
    {
        "field": "TechSupport",
        "role": "feature",
        "core_signal": True,
        "aliases": ["techsupport", "tech_support", "support_addon"],
    },
    {
        "field": "PaperlessBilling",
        "role": "feature",
        "core_signal": True,
        "aliases": ["paperlessbilling", "paperless_billing", "ebilling", "e_billing"],
    },
    {
        "field": "TotalCharges",
        "role": "feature",
        "core_signal": True,
        "aliases": ["totalcharges", "total_charges", "lifetime_billed", "total_spend"],
    },
    # --- Other real Telco/Banking precedent features (not core-signal,
    # but still suggested when a close name match exists). ---
    {"field": "gender", "role": "feature", "core_signal": False, "aliases": ["gender", "sex"]},
    {"field": "SeniorCitizen", "role": "feature", "core_signal": False, "aliases": ["seniorcitizen", "senior_citizen", "is_senior"]},
    {"field": "Partner", "role": "feature", "core_signal": False, "aliases": ["partner", "has_partner", "married"]},
    {"field": "Dependents", "role": "feature", "core_signal": False, "aliases": ["dependents", "has_dependents"]},
    {"field": "PhoneService", "role": "feature", "core_signal": False, "aliases": ["phoneservice", "phone_service"]},
    {"field": "MultipleLines", "role": "feature", "core_signal": False, "aliases": ["multiplelines", "multiple_lines"]},
    {"field": "OnlineBackup", "role": "feature", "core_signal": False, "aliases": ["onlinebackup", "online_backup"]},
    {"field": "DeviceProtection", "role": "feature", "core_signal": False, "aliases": ["deviceprotection", "device_protection"]},
    {"field": "StreamingTV", "role": "feature", "core_signal": False, "aliases": ["streamingtv", "streaming_tv"]},
    {"field": "StreamingMovies", "role": "feature", "core_signal": False, "aliases": ["streamingmovies", "streaming_movies"]},
    {"field": "CreditScore", "role": "feature", "core_signal": False, "aliases": ["creditscore", "credit_score"]},
    {"field": "Geography", "role": "feature", "core_signal": False, "aliases": ["geography", "country", "region"]},
    {"field": "Age", "role": "feature", "core_signal": False, "aliases": ["age", "customer_age"]},
    {"field": "NumOfProducts", "role": "feature", "core_signal": False, "aliases": ["numofproducts", "num_of_products", "product_count"]},
    {"field": "HasCrCard", "role": "feature", "core_signal": False, "aliases": ["hascrcard", "has_cr_card", "has_credit_card"]},
    {"field": "IsActiveMember", "role": "feature", "core_signal": False, "aliases": ["isactivemember", "is_active_member", "is_active"]},
    {"field": "EstimatedSalary", "role": "feature", "core_signal": False, "aliases": ["estimatedsalary", "estimated_salary", "salary", "income"]},
]

CORE_SIGNAL_FIELDS: list[str] = [entry["field"] for entry in KNOWN_SCHEMA_FIELDS if entry["core_signal"]]


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def suggest_mapping(columns: list[str]) -> dict[str, dict[str, Any]]:
    """For each uploaded column name, suggest the closest known schema field
    by normalized name similarity. Returns a suggestion only - the caller
    (the mapping UI) must still let the user confirm or change every one via
    a dropdown before /api/onboarding/validate ever acts on it."""
    suggestions: dict[str, dict[str, Any]] = {}
    for column in columns:
        normalized_column = _normalize(column)
        best_entry: dict[str, Any] | None = None
        best_score = 0.0

        for entry in KNOWN_SCHEMA_FIELDS:
            for candidate in [entry["field"], *entry["aliases"]]:
                normalized_candidate = _normalize(candidate)
                if not normalized_column or not normalized_candidate:
                    continue
                score = (
                    1.0
                    if normalized_column == normalized_candidate
                    else difflib.SequenceMatcher(None, normalized_column, normalized_candidate).ratio()
                )
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry is not None and best_score >= SUGGESTION_THRESHOLD:
            suggestions[column] = {
                "suggested_role": best_entry["role"],
                "matched_known_field": best_entry["field"],
                "confidence": round(best_score, 2),
            }
        else:
            suggestions[column] = {
                "suggested_role": "ignore",
                "matched_known_field": None,
                "confidence": round(best_score, 2),
            }

    return suggestions


def suggest_duration_leakage_column(
    df: pd.DataFrame, suggestions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Value-based refinement pass, run AFTER suggest_mapping()'s pure
    name-similarity pass (which only ever looks at column NAMES, never
    values) - identifies whether any uploaded column is highly correlated
    with duration * revenue, the exact TotalCharges ~= tenure *
    MonthlyCharges relationship that leaked duration information back into
    Telco's survival model before it was caught and excluded (see
    src/models/survival.py's module docstring). Lets a company identify
    their own equivalent column (if one exists) during the guided mapping
    step, rather than only ever catching it automatically at training time
    (see survival.py's DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD fallback for
    when nothing is mapped here).

    Needs a preliminary duration/revenue column to compute the product
    against - "preliminary" because at upload time nothing is confirmed
    yet, so this reads matched_known_field (tenure/duration_since_signup
    for duration, any revenue-role match for revenue) rather than
    suggested_role: a column literally named "tenure" already, correctly,
    keeps suggesting role="feature" (see KNOWN_SCHEMA_FIELDS' comment on
    why), but it's still real evidence of which column plays the duration
    role for THIS check. Only ever ADDS a "duration_leakage_column"
    suggestion on top of suggest_mapping()'s output for the one column (if
    any) whose correlation clears the threshold - never removes or
    overrides any other column's suggested role.
    """
    duration_candidates = [
        column
        for column, info in suggestions.items()
        if info["matched_known_field"] in ("tenure", "duration_since_signup")
    ]
    revenue_candidates = [column for column, info in suggestions.items() if info["suggested_role"] == "revenue"]
    if not duration_candidates or not revenue_candidates:
        return suggestions

    duration_column = duration_candidates[0]
    revenue_column = revenue_candidates[0]
    if duration_column not in df.columns or revenue_column not in df.columns:
        return suggestions

    duration_values = pd.to_numeric(df[duration_column], errors="coerce")
    revenue_values = pd.to_numeric(df[revenue_column], errors="coerce")
    product = duration_values * revenue_values
    if product.notna().sum() < 2:
        return suggestions

    best_column: str | None = None
    best_correlation = 0.0
    for column in df.columns:
        if column in (duration_column, revenue_column):
            continue
        candidate_values = pd.to_numeric(df[column], errors="coerce")
        if candidate_values.notna().sum() < 2:
            continue
        correlation = product.corr(candidate_values)
        if correlation is not None and abs(correlation) > best_correlation:
            best_correlation = abs(correlation)
            best_column = column

    if best_column is None or best_correlation < DURATION_LEAKAGE_SUGGESTION_CORRELATION_THRESHOLD:
        return suggestions

    refined = dict(suggestions)
    refined[best_column] = {
        "suggested_role": "duration_leakage_column",
        "matched_known_field": None,
        "confidence": round(best_correlation, 2),
    }
    return refined
