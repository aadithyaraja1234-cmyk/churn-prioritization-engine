"""Generates 6 distinct, clean, live-demo-ready company CSVs - see
data/live_demo/README.md for each company's backstory, row count, and
column-to-role mapping.

Distinct from data/test_onboarding_samples/ (deliberately shows the
onboarding pipeline's FULL range, including thin data, confusing column
names, quiet data-quality rot, and a malicious upload) and data/raw/
(Telco/Banking's committed reference data). Every file here is
deliberately clean - no injected data-quality issues - and designed to
plausibly clear every module's own sanity gate
(src/models/tenant_training.py's _run_optional_modules()/_sanity_check())
if registered/uploaded/trained through the real onboarding UI, which this
script does NOT do itself (see this project's own live-demo workflow -
register/upload/map/validate/train is done live, by a person, choosing
which company to use).

Every company gets the full mappable-role set, not just the classifier
minimum: a customer_id, a target, a revenue column (recurring charge), a
duration column (tenure-equivalent), a CLV column (genuine noise, not a
deterministic function of other columns - see _make_clv()'s docstring),
and 6+ additional feature columns.

THE SEGMENT-COLUMN DOMINANCE TRICK (read before adding a 7th company):
survival.py has no onboarding role for "segment_feature_column" - it's
auto-derived from whichever categorical covariate the fitted Cox model's
own hazard ratios deviate furthest from 1.0 (see that module's docstring).
Categorical covariates are encoded as plain integer codes in SORTED-NAME
order (src/features/encode.py's fit_categorical_encoders: alphabetical),
then fit as an ordinary linear covariate - so a 3-level category whose
alphabetical order does NOT match its true risk order gets its real
signal diluted (verified empirically: a symmetric Basic/highest-risk,
Standard/lowest-risk, Premium/middle-risk design, encoded alphabetically
as Basic=0/Premium=1/Standard=2, fits a hazard ratio within 2% of 1.0
despite a large designed effect - the encoding order scrambles it against
a collinear continuous revenue column). The fix, applied to every company
below: pick tier/plan/contract-equivalent LABELS whose ALPHABETICAL order
already matches the intended risk order (highest-risk name sorts first),
the same coincidence Telco's own real "Contract" column already has
("Month-to-month" < "One year" < "Two year", both alphabetically and by
risk) - not a hack, just deliberately replicating a real, already-working
pattern instead of leaving it to chance. Verified directly against
lifelines.CoxPHFitter (this project's own real survival engine) before
committing to this approach.

Deterministic: every random draw goes through a single seeded
numpy.random.Generator (SEED), so re-running this script reproduces
byte-identical output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "live_demo"


def _ids(n: int, prefix: str, width: int = 5) -> list[str]:
    return [f"{prefix}-{i:0{width}d}" for i in range(1, n + 1)]


def _tiered_numeric(rng: np.random.Generator, tier: np.ndarray, tier_order: list[str], params: dict) -> np.ndarray:
    """params: {tier_name: (mean, sd, lo, hi)} - independent noisy draw per
    tier band, not a bare lookup (every company's revenue column has real
    within-tier variance)."""
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
    """Genuine, noisy CLV-like column: real signal from duration*revenue (the
    same real-world "spend so far" relationship Telco's TotalCharges has)
    PLUS a substantial multiplicative noise factor (median 1.0, real
    spread) PLUS an independent additive bonus/noise term - deliberately
    NOT a near-deterministic function of duration/revenue, unlike
    TotalCharges (tenure*MonthlyCharges, correlation 0.9996 with the
    product) - the exact leak shape
    src/models/survival.py's DURATION_LEAKAGE_AUTO_DETECT_THRESHOLD (0.9)
    exists to catch, and this project has now caught twice for real. Every
    caller's correlation is checked in __main__ below, not just assumed."""
    noise_factor = rng.lognormal(mean=0.0, sigma=mult_sigma, size=len(duration))
    value = duration * revenue * noise_factor + bonus + rng.normal(0, additive_noise_sd, size=len(duration))
    return np.round(value.clip(0, None), 2)


def _bernoulli(rng: np.random.Generator, logit: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-logit))
    return rng.random(len(logit)) < p


# ---------------------------------------------------------------------------
# 1. Aurora Streaming Co. - consumer video/audio streaming subscription
# ---------------------------------------------------------------------------

AURORA_TIERS = ["Basic", "Plus", "Premium"]  # alphabetical == decreasing risk
AURORA_TIER_WEIGHTS = [0.38, 0.40, 0.22]
AURORA_FEE_PARAMS = {"Basic": (8.99, 1.0, 6.0, 12.0), "Plus": (13.99, 1.2, 11.0, 17.5), "Premium": (17.99, 1.6, 15.0, 24.0)}


def make_aurora_streaming(rng: np.random.Generator, n: int = 4000) -> pd.DataFrame:
    tier = rng.choice(AURORA_TIERS, size=n, p=AURORA_TIER_WEIGHTS)
    tenure = rng.integers(0, 73, size=n)
    monthly_fee = _tiered_numeric(rng, tier, AURORA_TIERS, AURORA_FEE_PARAMS)
    payment_method = rng.choice(["Credit Card", "Debit Card", "PayPal", "Gift Card"], size=n, p=[0.45, 0.25, 0.25, 0.05])
    paperless_billing = np.where(rng.random(n) < 0.74, "Yes", "No")
    device_count = (rng.poisson(1.8, size=n).clip(0, 5) + 1).astype(int)
    content_hours_per_month = np.round(rng.gamma(shape=3.2, scale=15.0, size=n).clip(0, 260), 1)
    family_plan_p = (0.12 + 0.28 * (tier == "Premium") + 0.10 * (tier == "Plus") + 0.12 * (device_count >= 4)).clip(0.03, 0.85)
    has_family_plan = rng.random(n) < family_plan_p
    support_tickets_ytd = rng.poisson(1.1, size=n).clip(0, 12)
    profile_count = (rng.poisson(1.6, size=n).clip(0, 5) + 1).astype(int)
    days_since_last_login = rng.gamma(shape=1.6, scale=6.0, size=n).clip(0, 90).round().astype(int)
    email_verified = rng.random(n) < 0.88

    logit = (
        -1.15
        + 1.75 * (tier == "Basic")
        - 1.15 * (tier == "Premium")
        + 0.55 * (tenure < 12)
        - 0.35 * (tenure >= 48)
        - 0.30 * has_family_plan
        - 0.010 * content_hours_per_month
        + 0.10 * support_tickets_ytd
        + 0.012 * days_since_last_login
        - 0.15 * email_verified
        + rng.normal(0, 0.55, size=n)
    )
    churned = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(rng, tenure, monthly_fee, bonus=0.55 * content_hours_per_month + np.where(has_family_plan, 45.0, 0.0))

    return pd.DataFrame(
        {
            "customer_id": _ids(n, "AUR"),
            "tenure": tenure,
            "subscription_tier": tier,
            "monthly_fee": monthly_fee,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "device_count": device_count,
            "content_hours_per_month": content_hours_per_month,
            "has_family_plan": np.where(has_family_plan, "Yes", "No"),
            "support_tickets_ytd": support_tickets_ytd,
            "profile_count": profile_count,
            "days_since_last_login": days_since_last_login,
            "email_verified": np.where(email_verified, "Yes", "No"),
            "lifetime_value_estimate": ltv,
            "churned": churned,
        }
    )


# ---------------------------------------------------------------------------
# 2. Palisade Fitness - subscription gym chain
# ---------------------------------------------------------------------------

PALISADE_TIERS = ["Basic", "Core", "Elite"]  # alphabetical == decreasing risk
PALISADE_TIER_WEIGHTS = [0.42, 0.38, 0.20]
PALISADE_DUES_PARAMS = {"Basic": (24.99, 3.0, 15.0, 35.0), "Core": (44.99, 4.0, 35.0, 55.0), "Elite": (79.99, 8.0, 60.0, 110.0)}


def make_palisade_fitness(rng: np.random.Generator, n: int = 3000) -> pd.DataFrame:
    tier = rng.choice(PALISADE_TIERS, size=n, p=PALISADE_TIER_WEIGHTS)
    tenure_months = rng.integers(0, 61, size=n)
    monthly_dues = _tiered_numeric(rng, tier, PALISADE_TIERS, PALISADE_DUES_PARAMS)
    visit_frequency_per_month = np.round(rng.gamma(shape=2.6, scale=2.8, size=n).clip(0, 30), 1)
    personal_training_addon = rng.random(n) < (0.10 + 0.30 * (tier == "Elite") + 0.12 * (tier == "Core"))
    group_classes_per_month = rng.poisson(2.2, size=n).clip(0, 20)
    has_locker_rental = rng.random(n) < 0.22
    referred_by_member = rng.random(n) < 0.18
    payment_method = rng.choice(["Credit Card", "Debit Card", "Bank Draft"], size=n, p=[0.55, 0.30, 0.15])
    avg_session_minutes = np.round(rng.normal(52, 14, size=n).clip(15, 120), 1)
    app_engagement_score = rng.integers(0, 101, size=n)
    has_nutrition_plan = rng.random(n) < 0.16

    logit = (
        -0.95
        + 1.55 * (tier == "Basic")
        - 1.05 * (tier == "Elite")
        + 0.50 * (tenure_months < 6)
        - 0.30 * (tenure_months >= 24)
        - 0.045 * visit_frequency_per_month
        - 0.35 * personal_training_addon
        - 0.20 * referred_by_member
        - 0.006 * app_engagement_score
        - 0.20 * has_nutrition_plan
        + rng.normal(0, 0.55, size=n)
    )
    cancelled = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, tenure_months, monthly_dues,
        bonus=1.2 * group_classes_per_month + np.where(personal_training_addon, 60.0, 0.0),
    )

    return pd.DataFrame(
        {
            "member_id": _ids(n, "PAL"),
            "tenure_months": tenure_months,
            "membership_tier": tier,
            "monthly_dues": monthly_dues,
            "visit_frequency_per_month": visit_frequency_per_month,
            "personal_training_addon": np.where(personal_training_addon, "Yes", "No"),
            "group_classes_per_month": group_classes_per_month,
            "has_locker_rental": np.where(has_locker_rental, "Yes", "No"),
            "referred_by_member": np.where(referred_by_member, "Yes", "No"),
            "payment_method": payment_method,
            "avg_session_minutes": avg_session_minutes,
            "app_engagement_score": app_engagement_score,
            "has_nutrition_plan": np.where(has_nutrition_plan, "Yes", "No"),
            "membership_value_estimate": ltv,
            "cancelled": cancelled,
        }
    )


# ---------------------------------------------------------------------------
# 3. Northgate Broadband - B2B internet service provider (fewer, larger accounts)
# ---------------------------------------------------------------------------

NORTHGATE_TIERS = ["Flex", "Standard", "Term"]  # alphabetical == decreasing risk (month-to-month -> multi-year)
NORTHGATE_TIER_WEIGHTS = [0.30, 0.42, 0.28]
NORTHGATE_BILL_PARAMS = {"Flex": (420.0, 60.0, 250.0, 650.0), "Standard": (610.0, 80.0, 420.0, 900.0), "Term": (890.0, 140.0, 600.0, 1400.0)}


def make_northgate_broadband(rng: np.random.Generator, n: int = 800) -> pd.DataFrame:
    contract_length = rng.choice(NORTHGATE_TIERS, size=n, p=NORTHGATE_TIER_WEIGHTS)
    tenure_months = rng.integers(0, 97, size=n)
    monthly_bill = _tiered_numeric(rng, contract_length, NORTHGATE_TIERS, NORTHGATE_BILL_PARAMS)
    num_locations = rng.poisson(2.3, size=n).clip(0, 25) + 1
    support_tickets_ytd = rng.poisson(3.0, size=n).clip(0, 40)
    bandwidth_tier = rng.choice(["Standard", "High", "Enterprise"], size=n, p=[0.45, 0.35, 0.20])
    dedicated_support_rep = rng.random(n) < (0.15 + 0.35 * (contract_length == "Term") + 0.10 * (num_locations >= 5))
    employee_count = rng.integers(3, 500, size=n)
    autopay_enabled = rng.random(n) < 0.68
    sla_uptime_pct = np.round(rng.normal(99.5, 0.4, size=n).clip(97.0, 99.99), 2)
    years_in_business = rng.integers(1, 40, size=n)
    has_backup_line = rng.random(n) < (0.18 + 0.22 * (contract_length == "Term"))

    logit = (
        -2.05
        + 1.90 * (contract_length == "Flex")
        - 1.30 * (contract_length == "Term")
        + 0.70 * (tenure_months < 6)
        - 0.12 * np.log1p(num_locations)
        + 0.11 * support_tickets_ytd
        - 0.55 * dedicated_support_rep
        - 0.35 * has_backup_line
        + rng.normal(0, 0.50, size=n)
    )
    terminated = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, tenure_months, monthly_bill,
        bonus=25.0 * np.log1p(num_locations) + np.where(dedicated_support_rep, 400.0, 0.0),
        additive_noise_sd=900.0,
    )

    return pd.DataFrame(
        {
            "account_id": _ids(n, "NGB", width=5),
            "tenure_months": tenure_months,
            "contract_length": contract_length,
            "monthly_bill": monthly_bill,
            "num_locations": num_locations,
            "support_tickets_ytd": support_tickets_ytd,
            "bandwidth_tier": bandwidth_tier,
            "sla_uptime_pct": sla_uptime_pct,
            "years_in_business": years_in_business,
            "has_backup_line": np.where(has_backup_line, "Yes", "No"),
            "dedicated_support_rep": np.where(dedicated_support_rep, "Yes", "No"),
            "employee_count": employee_count,
            "autopay_enabled": np.where(autopay_enabled, "Yes", "No"),
            "account_value_estimate": ltv,
            "terminated": terminated,
        }
    )


# ---------------------------------------------------------------------------
# 4. Solstice Media - consumer streaming/media, short tenure, high baseline churn
# ---------------------------------------------------------------------------

SOLSTICE_TIERS = ["Ad-Lite", "Standard", "Ultra"]  # alphabetical == decreasing risk
SOLSTICE_TIER_WEIGHTS = [0.46, 0.36, 0.18]
SOLSTICE_RATE_PARAMS = {"Ad-Lite": (5.99, 0.8, 4.0, 8.0), "Standard": (11.99, 1.1, 9.0, 15.0), "Ultra": (16.99, 1.4, 13.0, 22.0)}


def make_solstice_media(rng: np.random.Generator, n: int = 5000) -> pd.DataFrame:
    plan_level = rng.choice(SOLSTICE_TIERS, size=n, p=SOLSTICE_TIER_WEIGHTS)
    # Short average tenure, deliberately skewed low (a beta-shaped draw over
    # 0-48 months, not a flat uniform like Aurora/Palisade) so this
    # company's overall population reads as structurally high-churn, not
    # just a different random draw of the same shape.
    months_subscribed = np.round(rng.beta(1.4, 3.2, size=n) * 48).astype(int)
    monthly_rate = _tiered_numeric(rng, plan_level, SOLSTICE_TIERS, SOLSTICE_RATE_PARAMS)
    devices_registered = (rng.poisson(1.4, size=n).clip(0, 4) + 1).astype(int)
    weekly_watch_hours = np.round(rng.gamma(shape=2.4, scale=3.2, size=n).clip(0, 60), 1)
    autoplay_next_episode = rng.random(n) < 0.63
    payment_method = rng.choice(["Credit Card", "PayPal", "Mobile Carrier Billing", "Gift Card"], size=n, p=[0.40, 0.28, 0.24, 0.08])
    binge_sessions_per_month = rng.poisson(3.5, size=n).clip(0, 25)
    signup_channel = rng.choice(["App Store", "Web", "Partner Bundle"], size=n, p=[0.40, 0.38, 0.22])
    num_profiles = (rng.poisson(1.2, size=n).clip(0, 4) + 1).astype(int)
    days_since_last_watch = rng.gamma(shape=1.7, scale=5.5, size=n).clip(0, 90).round().astype(int)
    parental_controls_enabled = rng.random(n) < 0.21

    logit = (
        -0.55
        + 1.35 * (plan_level == "Ad-Lite")
        - 0.75 * (plan_level == "Ultra")
        + 0.60 * (months_subscribed < 6)
        - 0.30 * (months_subscribed >= 24)
        - 0.045 * weekly_watch_hours
        - 0.18 * autoplay_next_episode
        + 0.011 * days_since_last_watch
        - 0.10 * parental_controls_enabled
        + rng.normal(0, 0.55, size=n)
    )
    subscription_lapsed = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, months_subscribed, monthly_rate,
        bonus=0.8 * weekly_watch_hours + 3.0 * binge_sessions_per_month,
    )

    return pd.DataFrame(
        {
            "viewer_id": _ids(n, "SOL"),
            "months_subscribed": months_subscribed,
            "plan_level": plan_level,
            "monthly_rate": monthly_rate,
            "devices_registered": devices_registered,
            "weekly_watch_hours": weekly_watch_hours,
            "autoplay_next_episode": np.where(autoplay_next_episode, "Yes", "No"),
            "payment_method": payment_method,
            "binge_sessions_per_month": binge_sessions_per_month,
            "signup_channel": signup_channel,
            "num_profiles": num_profiles,
            "days_since_last_watch": days_since_last_watch,
            "parental_controls_enabled": np.where(parental_controls_enabled, "Yes", "No"),
            "projected_lifetime_value": ltv,
            "subscription_lapsed": subscription_lapsed,
        }
    )


# ---------------------------------------------------------------------------
# 5. Ferrous Vale Wealth - wealth management / financial advisory
# ---------------------------------------------------------------------------

FERROUS_TIERS = ["Core", "Premier", "Private"]  # alphabetical == decreasing risk
FERROUS_TIER_WEIGHTS = [0.48, 0.34, 0.18]
FERROUS_FEE_PARAMS = {"Core": (2200.0, 400.0, 1200.0, 3200.0), "Premier": (5800.0, 900.0, 3500.0, 8500.0), "Private": (14500.0, 3200.0, 9000.0, 24000.0)}


def make_ferrous_vale_wealth(rng: np.random.Generator, n: int = 1200) -> pd.DataFrame:
    portfolio_tier = rng.choice(FERROUS_TIERS, size=n, p=FERROUS_TIER_WEIGHTS)
    account_tenure = rng.integers(0, 26, size=n)  # years - wealth-management relationships run long
    annual_fee_revenue = _tiered_numeric(rng, portfolio_tier, FERROUS_TIERS, FERROUS_FEE_PARAMS)
    advisor_assigned = rng.random(n) < (0.35 + 0.30 * (portfolio_tier == "Private") + 0.15 * (portfolio_tier == "Premier"))
    contact_frequency_per_year = rng.poisson(4.0, size=n).clip(0, 30)
    num_accounts_held = rng.poisson(1.6, size=n).clip(0, 8) + 1
    digital_logins_per_month = rng.poisson(3.2, size=n).clip(0, 40)
    referred_by_client = rng.random(n) < 0.24
    risk_tolerance_profile = rng.choice(["Conservative", "Moderate", "Aggressive"], size=n, p=[0.38, 0.42, 0.20])
    estate_planning_client = rng.random(n) < (0.10 + 0.25 * (portfolio_tier == "Private") + 0.10 * (portfolio_tier == "Premier"))
    avg_response_time_hours = np.round(rng.gamma(shape=2.0, scale=6.0, size=n).clip(0.5, 72), 1)
    has_joint_account = rng.random(n) < 0.34

    # Wealth management is genuinely low-churn/high-retention - a much
    # smaller baseline hazard than the consumer-subscription companies
    # above, same real-world shape as this industry.
    logit = (
        -2.60
        + 1.55 * (portfolio_tier == "Core")
        - 0.95 * (portfolio_tier == "Private")
        + 0.55 * (account_tenure < 2)
        - 0.35 * (account_tenure >= 10)
        - 0.55 * advisor_assigned
        - 0.045 * contact_frequency_per_year
        + 0.012 * avg_response_time_hours
        - 0.25 * estate_planning_client
        + rng.normal(0, 0.55, size=n)
    )
    left_firm = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, account_tenure, annual_fee_revenue,
        bonus=150.0 * num_accounts_held + np.where(advisor_assigned, 1200.0, 0.0),
        additive_noise_sd=3500.0,
    )

    return pd.DataFrame(
        {
            "client_id": _ids(n, "FVW"),
            "account_tenure": account_tenure,
            "portfolio_tier": portfolio_tier,
            "annual_fee_revenue": annual_fee_revenue,
            "advisor_assigned": np.where(advisor_assigned, "Yes", "No"),
            "contact_frequency_per_year": contact_frequency_per_year,
            "num_accounts_held": num_accounts_held,
            "digital_logins_per_month": digital_logins_per_month,
            "referred_by_client": np.where(referred_by_client, "Yes", "No"),
            "risk_tolerance_profile": risk_tolerance_profile,
            "estate_planning_client": np.where(estate_planning_client, "Yes", "No"),
            "avg_response_time_hours": avg_response_time_hours,
            "has_joint_account": np.where(has_joint_account, "Yes", "No"),
            "portfolio_value_estimate": ltv,
            "left_firm": left_firm,
        }
    )


# ---------------------------------------------------------------------------
# 6. Cascade Retail - e-commerce / retail subscription box service
# ---------------------------------------------------------------------------

CASCADE_TIERS = ["Basic", "Curated", "Deluxe"]  # alphabetical == decreasing risk
CASCADE_TIER_WEIGHTS = [0.40, 0.38, 0.22]
CASCADE_PRICE_PARAMS = {"Basic": (19.99, 2.0, 14.0, 26.0), "Curated": (34.99, 3.0, 26.0, 45.0), "Deluxe": (54.99, 5.0, 42.0, 72.0)}


def make_cascade_retail(rng: np.random.Generator, n: int = 2500) -> pd.DataFrame:
    plan_type = rng.choice(CASCADE_TIERS, size=n, p=CASCADE_TIER_WEIGHTS)
    tenure_months = rng.integers(0, 49, size=n)
    monthly_price = _tiered_numeric(rng, plan_type, CASCADE_TIERS, CASCADE_PRICE_PARAMS)
    items_per_box = rng.poisson(5.5, size=n).clip(2, 16)
    referral_count = rng.poisson(0.6, size=n).clip(0, 10)
    boxes_skipped_ytd = rng.poisson(1.3, size=n).clip(0, 12)
    gift_subscription = rng.random(n) < 0.09
    payment_method = rng.choice(["Credit Card", "Debit Card", "PayPal", "Buy Now Pay Later"], size=n, p=[0.48, 0.22, 0.24, 0.06])
    loyalty_points_balance = rng.gamma(shape=2.0, scale=180.0, size=n).clip(0, None).round().astype(int)
    app_installed = rng.random(n) < 0.57
    avg_review_rating = np.round(rng.normal(4.1, 0.7, size=n).clip(1.0, 5.0), 1)
    social_media_follower = rng.random(n) < 0.19

    logit = (
        -0.85
        + 1.45 * (plan_type == "Basic")
        - 0.85 * (plan_type == "Deluxe")
        + 0.50 * (tenure_months < 4)
        - 0.06 * items_per_box
        - 0.18 * referral_count
        + 0.16 * boxes_skipped_ytd
        - 0.35 * gift_subscription
        - 0.16 * app_installed
        - 0.14 * social_media_follower
        + rng.normal(0, 0.55, size=n)
    )
    unsubscribed = np.where(_bernoulli(rng, logit), "Yes", "No")

    ltv = _make_clv(
        rng, tenure_months, monthly_price,
        bonus=6.0 * referral_count + 0.4 * loyalty_points_balance,
    )

    return pd.DataFrame(
        {
            "subscriber_id": _ids(n, "CAS"),
            "tenure_months": tenure_months,
            "plan_type": plan_type,
            "monthly_price": monthly_price,
            "items_per_box": items_per_box,
            "referral_count": referral_count,
            "boxes_skipped_ytd": boxes_skipped_ytd,
            "gift_subscription": np.where(gift_subscription, "Yes", "No"),
            "payment_method": payment_method,
            "loyalty_points_balance": loyalty_points_balance,
            "app_installed": np.where(app_installed, "Yes", "No"),
            "avg_review_rating": avg_review_rating,
            "social_media_follower": np.where(social_media_follower, "Yes", "No"),
            "subscriber_value_estimate": ltv,
            "unsubscribed": unsubscribed,
        }
    )


COMPANIES = [
    ("aurora-streaming.csv", make_aurora_streaming),
    ("palisade-fitness.csv", make_palisade_fitness),
    ("northgate-broadband.csv", make_northgate_broadband),
    ("solstice-media.csv", make_solstice_media),
    ("ferrous-vale-wealth.csv", make_ferrous_vale_wealth),
    ("cascade-retail.csv", make_cascade_retail),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for filename, make_fn in COMPANIES:
        df = make_fn(rng)
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        target_col = df.columns[-1]
        positive_rate = (df[target_col] == "Yes").mean()
        print(f"{filename}: {len(df):,} rows, {len(df.columns)} columns, {target_col}=Yes rate {positive_rate:.1%} -> {path}")


if __name__ == "__main__":
    main()
