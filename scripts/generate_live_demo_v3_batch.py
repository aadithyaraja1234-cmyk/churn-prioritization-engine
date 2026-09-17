"""Generates 3 distinct, clean, live-demo-ready company CSVs - v3 batch.

See data/live_demo_v3/README.md for each company's backstory, row count,
and column-to-role mapping. Distinct from data/live_demo/ (the first
6-company batch, all below the row-count floor by design) and
data/live_demo_v2/ (the naming-convention stress batch, also below the
row-count floor by design): this batch exists for the opposite reason -
every company here clears BOTH of src/onboarding/validation.py's real
sufficiency floors (MIN_RECOMMENDED_ROWS = 5,634, MIN_RECOMMENDED_FEATURES
= 8) with real margin, so a live demo can show the validation step with NO
"results may be less reliable" warning at all, for once. Column naming is
ordinary snake_case throughout (like data/live_demo/, not data/live_demo_v2/'s
deliberate naming-convention stress test) - the row count is the point of
this batch, not the column names.

  1. Vertex Cloud Software  - B2B SaaS subscription (project-management
     tool), 10,500 rows
  2. Meridian Mobile         - postpaid wireless carrier, 14,000 rows
  3. Harvest Table Meal Kit  - meal-kit delivery subscription, 11,500 rows

Every company gets the full mappable-role set (id/target/revenue/duration/
segment-hazard-driver-categorical/clv) plus feature columns with real,
non-trivial variance and genuine churn structure, same quality bar and same
THE-SEGMENT-COLUMN-DOMINANCE-TRICK alphabetical-order convention as
scripts/generate_live_demo_batch.py (see that script's own module docstring
for the full explanation: survival.py auto-derives segment_feature_column
from whichever categorical's hazard ratio deviates furthest from 1.0, and
categorical levels are encoded in sorted-name order, so a tier/plan-style
column's real risk signal only survives cleanly when its alphabetical order
already matches its true risk order).

This script does NOT register, upload, map, or train any of these -
generate-and-save only, exactly like scripts/generate_live_demo_batch.py
(see this project's own live-demo workflow: register/upload/map/validate/
train is done live, by a person, choosing which company to use).

Deterministic: every random draw goes through a single seeded
numpy.random.Generator (SEED), so re-running this script reproduces
byte-identical output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "live_demo_v3"


def _ids(n: int, prefix: str, width: int = 6) -> list[str]:
    return [f"{prefix}-{i:0{width}d}" for i in range(1, n + 1)]


def _tiered_numeric(rng: np.random.Generator, tier: np.ndarray, tier_order: list[str], params: dict) -> np.ndarray:
    """params: {tier_name: (mean, sd, lo, hi)} - independent noisy draw per
    tier band, not a bare lookup (every company's revenue column has real
    within-tier variance). Same helper as scripts/generate_live_demo_batch.py."""
    out = np.empty(len(tier))
    for t in tier_order:
        mean, sd, lo, hi = params[t]
        mask = tier == t
        out[mask] = np.round(rng.normal(mean, sd, size=int(mask.sum())).clip(lo, hi), 2)
    return out


def _make_clv(
    rng: np.random.Generator,
    duration: np.ndarray,
    revenue: np.ndarray,
    bonus: np.ndarray,
    mult_sigma: float = 0.45,
    additive_noise_sd: float = 60.0,
) -> np.ndarray:
    """Genuine, noisy CLV-like column - real signal from duration*revenue
    PLUS substantial multiplicative + additive noise, deliberately NOT a
    near-deterministic function of duration/revenue (the exact leak shape
    src/models/survival.py's DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD exists
    to catch). Every caller's correlation is checked in __main__ below."""
    noise_factor = rng.lognormal(mean=0.0, sigma=mult_sigma, size=len(duration))
    value = duration * revenue * noise_factor + bonus + rng.normal(0, additive_noise_sd, size=len(duration))
    return np.round(value.clip(0, None), 2)


def _bernoulli(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-logit))
    return rng.random(len(logit)) < p


# ---------------------------------------------------------------------------
# 1. Vertex Cloud Software - B2B SaaS project-management subscription.
# ---------------------------------------------------------------------------

VERTEX_TIERS = ["Free", "Growth", "Scale"]  # alphabetical == decreasing risk
VERTEX_TIER_WEIGHTS = [0.34, 0.42, 0.24]
VERTEX_FEE_PARAMS = {
    "Free": (0.0, 0.0, 0.0, 0.0),
    "Growth": (49.0, 9.0, 25.0, 85.0),
    "Scale": (199.0, 35.0, 120.0, 350.0),
}


def make_vertex_cloud(rng: np.random.Generator, n: int = 10_500) -> pd.DataFrame:
    plan_tier = rng.choice(VERTEX_TIERS, size=n, p=VERTEX_TIER_WEIGHTS)
    tenure_months = rng.integers(0, 61, size=n)
    monthly_subscription_fee = _tiered_numeric(rng, plan_tier, VERTEX_TIERS, VERTEX_FEE_PARAMS)
    num_seats = rng.poisson(6.0, size=n).clip(1, 250)
    integrations_connected = rng.poisson(2.1, size=n).clip(0, 20)
    api_calls_per_month = rng.gamma(shape=2.0, scale=1500.0, size=n).clip(0, None).round().astype(int)
    support_tickets_ytd = rng.poisson(1.3, size=n).clip(0, 25)
    onboarding_completed = rng.random(n) < 0.71
    admin_logins_per_week = np.round(rng.gamma(shape=2.2, scale=1.8, size=n).clip(0, 40), 1)
    has_sso = rng.random(n) < (0.10 + 0.55 * (plan_tier == "Scale") + 0.20 * (plan_tier == "Growth"))
    # 0-100, not the standard -100..100 NPS range: a negative cell value
    # would start with "-" and trip src/onboarding/upload_security.py's
    # FORMULA_INJECTION_PREFIXES check on real upload (any cell starting
    # with "=", "+", "-", "@" is rejected as a spreadsheet-formula-
    # injection risk) - verified directly against that same check in
    # __main__ below, not just assumed safe.
    nps_score = rng.integers(0, 101, size=n)
    days_since_last_login = rng.exponential(scale=9.0, size=n).clip(0, 365).round().astype(int)
    billing_cycle = rng.choice(["monthly", "annual"], size=n, p=[0.68, 0.32])

    logit = (
        -1.35
        + 1.75 * (plan_tier == "Free")
        - 1.10 * (plan_tier == "Scale")
        + 0.55 * (tenure_months < 3)
        - 0.30 * (tenure_months >= 24)
        - 0.60 * onboarding_completed
        - 0.35 * has_sso
        + 0.16 * support_tickets_ytd
        - 0.010 * nps_score
        + 0.030 * days_since_last_login
        - 0.05 * num_seats.clip(0, 30) / 10
        - 0.20 * (billing_cycle == "annual")
        + rng.normal(0, 0.55, size=n)
    )
    churned = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, tenure_months, monthly_subscription_fee,
        bonus=1.5 * api_calls_per_month.clip(0, 8000) / 100 + np.where(has_sso, 220.0, 0.0),
        additive_noise_sd=90.0,
    )

    return pd.DataFrame(
        {
            "customer_id": _ids(n, "VTX"),
            "churned": churned,
            "monthly_subscription_fee": monthly_subscription_fee,
            "tenure_months": tenure_months,
            "estimated_account_value": ltv,
            "plan_tier": plan_tier,
            "num_seats": num_seats,
            "integrations_connected": integrations_connected,
            "api_calls_per_month": api_calls_per_month,
            "support_tickets_ytd": support_tickets_ytd,
            "onboarding_completed": np.where(onboarding_completed, "Yes", "No"),
            "admin_logins_per_week": admin_logins_per_week,
            "has_sso": np.where(has_sso, "Yes", "No"),
            "nps_score": nps_score,
            "days_since_last_login": days_since_last_login,
            "billing_cycle": billing_cycle,
        }
    )


# ---------------------------------------------------------------------------
# 2. Meridian Mobile - postpaid wireless carrier.
# ---------------------------------------------------------------------------

MERIDIAN_TIERS = ["Basic", "Plus", "Premium"]  # alphabetical == decreasing risk
MERIDIAN_TIER_WEIGHTS = [0.36, 0.40, 0.24]
MERIDIAN_FEE_PARAMS = {
    "Basic": (35.0, 5.0, 20.0, 55.0),
    "Plus": (60.0, 7.0, 42.0, 85.0),
    "Premium": (95.0, 12.0, 65.0, 140.0),
}


def make_meridian_mobile(rng: np.random.Generator, n: int = 14_000) -> pd.DataFrame:
    plan_type = rng.choice(MERIDIAN_TIERS, size=n, p=MERIDIAN_TIER_WEIGHTS)
    tenure_months = rng.integers(0, 97, size=n)
    monthly_bill = _tiered_numeric(rng, plan_type, MERIDIAN_TIERS, MERIDIAN_FEE_PARAMS)
    num_lines = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.38, 0.28, 0.16, 0.12, 0.06])
    has_international_plan = rng.random(n) < 0.18
    data_overage_gb = rng.gamma(shape=1.4, scale=2.5, size=n).clip(0, 60)
    autopay_enabled = rng.random(n) < 0.64
    device_financing_active = rng.random(n) < 0.41
    network_type = rng.choice(["4G", "5G"], size=n, p=[0.34, 0.66])
    support_calls_ytd = rng.poisson(1.5, size=n).clip(0, 20)
    days_since_last_upgrade = rng.integers(0, 1500, size=n)
    family_plan_member = rng.random(n) < (num_lines >= 3)
    satisfaction_score = rng.integers(1, 11, size=n)

    logit = (
        -1.15
        + 1.55 * (plan_type == "Basic")
        - 0.95 * (plan_type == "Premium")
        + 0.50 * (tenure_months < 6)
        - 0.30 * (tenure_months >= 48)
        - 0.45 * autopay_enabled
        - 0.30 * device_financing_active
        + 0.15 * support_calls_ytd
        + 0.05 * data_overage_gb / 5
        - 0.10 * satisfaction_score
        - 0.12 * family_plan_member
        + rng.normal(0, 0.55, size=n)
    )
    churned = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, tenure_months, monthly_bill,
        bonus=18.0 * num_lines + np.where(device_financing_active, 140.0, 0.0),
        additive_noise_sd=140.0,
    )

    return pd.DataFrame(
        {
            "customer_id": _ids(n, "MRD"),
            "churned": churned,
            "monthly_bill": monthly_bill,
            "tenure_months": tenure_months,
            "estimated_customer_value": ltv,
            "plan_type": plan_type,
            "num_lines": num_lines,
            "has_international_plan": np.where(has_international_plan, "Yes", "No"),
            "data_overage_gb": np.round(data_overage_gb, 2),
            "autopay_enabled": np.where(autopay_enabled, "Yes", "No"),
            "device_financing_active": np.where(device_financing_active, "Yes", "No"),
            "network_type": network_type,
            "support_calls_ytd": support_calls_ytd,
            "days_since_last_upgrade": days_since_last_upgrade,
            "family_plan_member": np.where(family_plan_member, "Yes", "No"),
            "satisfaction_score": satisfaction_score,
        }
    )


# ---------------------------------------------------------------------------
# 3. Harvest Table Meal Kit - meal-kit delivery subscription.
# ---------------------------------------------------------------------------

HARVEST_TIERS = ["Basic", "Family", "Gourmet"]  # alphabetical == decreasing risk
HARVEST_TIER_WEIGHTS = [0.38, 0.36, 0.26]
HARVEST_FEE_PARAMS = {
    "Basic": (39.99, 6.0, 25.0, 60.0),
    "Family": (69.99, 9.0, 50.0, 95.0),
    "Gourmet": (99.99, 13.0, 75.0, 145.0),
}


def make_harvest_table(rng: np.random.Generator, n: int = 11_500) -> pd.DataFrame:
    meal_plan = rng.choice(HARVEST_TIERS, size=n, p=HARVEST_TIER_WEIGHTS)
    tenure_weeks = rng.integers(0, 105, size=n)
    weekly_box_price = _tiered_numeric(rng, meal_plan, HARVEST_TIERS, HARVEST_FEE_PARAMS)
    household_size = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.18, 0.34, 0.20, 0.16, 0.08, 0.04])
    dietary_preference = rng.choice(
        ["Omnivore", "Vegetarian", "Vegan", "Pescatarian"], size=n, p=[0.56, 0.22, 0.12, 0.10]
    )
    meals_skipped_ytd = rng.poisson(3.2, size=n).clip(0, 40)
    referral_count = rng.poisson(0.4, size=n).clip(0, 10)
    recipe_rating_avg = np.round(rng.normal(4.1, 0.6, size=n).clip(1.0, 5.0), 1)
    delivery_issues_ytd = rng.poisson(0.7, size=n).clip(0, 15)
    app_engagement_score = rng.integers(0, 101, size=n)
    payment_method = rng.choice(["card", "paypal", "bnpl"], size=n, p=[0.62, 0.27, 0.11])
    auto_renew = rng.random(n) < 0.58
    boxes_per_week = rng.choice([1, 2, 3], size=n, p=[0.50, 0.35, 0.15])

    logit = (
        -1.25
        + 1.60 * (meal_plan == "Basic")
        - 1.00 * (meal_plan == "Gourmet")
        + 0.55 * (tenure_weeks < 8)
        - 0.30 * (tenure_weeks >= 52)
        + 0.14 * meals_skipped_ytd
        - 0.22 * referral_count
        - 0.16 * (recipe_rating_avg - 3.0)
        + 0.18 * delivery_issues_ytd
        - 0.012 * app_engagement_score
        - 0.45 * auto_renew
        + rng.normal(0, 0.55, size=n)
    )
    cancelled = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, tenure_weeks, weekly_box_price,
        bonus=12.0 * referral_count + 6.0 * boxes_per_week,
        additive_noise_sd=55.0,
    )

    return pd.DataFrame(
        {
            "customer_id": _ids(n, "HTM"),
            "cancelled": cancelled,
            "weekly_box_price": weekly_box_price,
            "tenure_weeks": tenure_weeks,
            "estimated_lifetime_value": ltv,
            "meal_plan": meal_plan,
            "household_size": household_size,
            "dietary_preference": dietary_preference,
            "meals_skipped_ytd": meals_skipped_ytd,
            "referral_count": referral_count,
            "recipe_rating_avg": recipe_rating_avg,
            "delivery_issues_ytd": delivery_issues_ytd,
            "app_engagement_score": app_engagement_score,
            "payment_method": payment_method,
            "auto_renew": np.where(auto_renew, "Yes", "No"),
            "boxes_per_week": boxes_per_week,
        }
    )


COMPANIES = [
    ("vertex-cloud-software.csv", make_vertex_cloud),
    ("meridian-mobile.csv", make_meridian_mobile),
    ("harvest-table-meal-kit.csv", make_harvest_table),
]

# (target_column, positive_value) - used by __main__'s sanity printout below.
TARGET_INFO = {
    "vertex-cloud-software.csv": ("churned", "Yes"),
    "meridian-mobile.csv": ("churned", "Yes"),
    "harvest-table-meal-kit.csv": ("cancelled", "Yes"),
}

# (duration_column, revenue_column, clv_column) - used by __main__'s
# leakage-correlation check below: _make_clv() is designed to NOT be a
# near-deterministic function of duration*revenue (unlike Telco's own real
# TotalCharges, correlation 0.9996 with tenure*MonthlyCharges - the exact
# leak shape src/models/survival.py's DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD
# (0.9) exists to catch), but that's a property of the noise parameters
# chosen per company, not something to just assert - so it's actually
# checked here against the same real threshold, for every company, not
# left as an unverified claim.
LEAKAGE_CHECK_COLUMNS = {
    "vertex-cloud-software.csv": ("tenure_months", "monthly_subscription_fee", "estimated_account_value"),
    "meridian-mobile.csv": ("tenure_months", "monthly_bill", "estimated_customer_value"),
    "harvest-table-meal-kit.csv": ("tenure_weeks", "weekly_box_price", "estimated_lifetime_value"),
}
DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD = 0.9  # mirrors src/models/survival.py's own real threshold

# Mirrors src/onboarding/upload_security.py's FORMULA_INJECTION_PREFIXES
# exactly: real upload (not just this offline generator) rejects ANY cell
# whose stripped value starts with one of these - including a plain
# negative number ("-42") - as a spreadsheet-formula-injection risk. A file
# that only ever gets pre-checked via prepare_and_train() directly (which
# skips the upload endpoint's own validation) can silently ship a column
# that fails the instant someone actually uploads it through the UI - see
# nps_score below, which is why this is checked here for every cell in
# every generated file, not assumed safe from column design alone.
FORMULA_INJECTION_PREFIXES = ("=", "+", "-", "@")


def _assert_no_formula_injection_risk(filename: str, csv_text: str) -> None:
    import csv
    import io

    rows = list(csv.reader(io.StringIO(csv_text)))
    header = rows[0]
    for row_number, row in enumerate(rows[1:], start=1):
        for column_index, cell in enumerate(row):
            stripped = cell.strip()
            if stripped and stripped[0] in FORMULA_INJECTION_PREFIXES:
                column_name = header[column_index] if column_index < len(header) else f"column {column_index + 1}"
                raise AssertionError(
                    f"{filename}: row {row_number}, column '{column_name}' has value {cell!r} starting with "
                    f"{stripped[0]!r} - real upload_security.py would reject this as formula-injection-like. "
                    "Fix the generator's value range for this column before this file is demo-safe."
                )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for filename, make_fn in COMPANIES:
        df = make_fn(rng)
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        target_col, positive_value = TARGET_INFO[filename]
        positive_rate = (df[target_col] == positive_value).mean()

        duration_col, revenue_col, clv_col = LEAKAGE_CHECK_COLUMNS[filename]
        product = df[duration_col] * df[revenue_col]
        clv_correlation = product.corr(df[clv_col])
        assert clv_correlation < DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD, (
            f"{filename}: {clv_col} correlates at {clv_correlation:.4f} with {duration_col}*{revenue_col} - "
            f"at or above the {DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD} leakage threshold; noise parameters "
            "need adjusting before this file is demo-safe."
        )

        _assert_no_formula_injection_risk(filename, path.read_text())

        print(
            f"{filename}: {len(df):,} rows, {len(df.columns)} columns, "
            f"{target_col}={positive_value!r} rate {positive_rate:.1%}, "
            f"clv-vs-duration*revenue correlation {clv_correlation:.4f} -> {path}"
        )


if __name__ == "__main__":
    main()
