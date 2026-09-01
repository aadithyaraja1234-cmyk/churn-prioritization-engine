"""Generates 5 synthetic CSV datasets used as onboarding-pipeline demo
material - see data/test_onboarding_samples/README.md for each fictional
company's backstory and what each file is meant to demonstrate.

Deterministic: every random draw goes through a single seeded
numpy.random.Generator, so re-running this script reproduces byte-identical
output.

Row counts: the task brief for Meridian Wireless suggested "~2,000 rows",
but this system's own real row-count threshold (MIN_RECOMMENDED_ROWS in
src/onboarding/validation.py) is 5,634 - the exact training-row count of
the real Telco model. 2,000 rows would trip the row-count warning and
never show a clean "Ready to go" verdict, contradicting the brief's own
goal for that file. Meridian, Fernwood, and Cobalt all use 6,000 rows
instead, comfortably above that threshold, so their row counts don't
confound the specific thing each file is meant to demonstrate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "test_onboarding_samples"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
CONTRACT_WEIGHTS = [0.55, 0.24, 0.21]
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
YES_NO = ["Yes", "No"]


def _customer_ids(rng: np.random.Generator, n: int, prefix: str) -> list[str]:
    numbers = rng.choice(90000, size=n, replace=False) + 10000
    suffixes = rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), size=(n, 4))
    return [f"{prefix}-{num}-{''.join(suf)}" for num, suf in zip(numbers, suffixes)]


def _yesno_dependent_on_internet(rng: np.random.Generator, internet_service: np.ndarray) -> np.ndarray:
    """OnlineSecurity/TechSupport/etc-style column: "No internet service" is
    forced whenever InternetService is "No" (mirrors the real Telco data's
    own dependent-column pattern), Yes/No drawn otherwise."""
    out = np.where(internet_service == "No", "No internet service", "")
    has_internet = internet_service != "No"
    draws = rng.choice(YES_NO, size=has_internet.sum(), p=[0.35, 0.65])
    out[has_internet] = draws
    return out


def make_telecom_frame(rng: np.random.Generator, n_rows: int, id_prefix: str) -> pd.DataFrame:
    """Base Telco-shaped synthetic frame shared by Meridian, Fernwood
    (renamed), Cobalt (degraded), and Redline (formula-injected)."""
    tenure = rng.integers(0, 73, size=n_rows)
    contract = rng.choice(CONTRACTS, size=n_rows, p=CONTRACT_WEIGHTS)
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows, p=[0.34, 0.19, 0.24, 0.23])
    internet_service = rng.choice(INTERNET_SERVICES, size=n_rows, p=[0.35, 0.44, 0.21])
    phone_service = rng.choice(YES_NO, size=n_rows, p=[0.90, 0.10])
    monthly_charges = np.round(rng.normal(64, 30, size=n_rows).clip(18, 120), 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 15, size=n_rows), 2).clip(0, None)

    # Churn probability correlated with realistic risk factors (month-to-
    # month contracts, short tenure, electronic check) so the label isn't
    # just random noise - mirrors the real Telco churn distribution.
    churn_score = (
        0.10
        + 0.30 * (contract == "Month-to-month")
        + 0.15 * (payment_method == "Electronic check")
        + 0.20 * (tenure < 12)
        - 0.15 * (contract == "Two year")
    )
    churn = rng.random(n_rows) < churn_score.clip(0.02, 0.85)

    df = pd.DataFrame(
        {
            "customerID": _customer_ids(rng, n_rows, id_prefix),
            "gender": rng.choice(["Male", "Female"], size=n_rows),
            "SeniorCitizen": rng.choice([0, 1], size=n_rows, p=[0.84, 0.16]),
            "Partner": rng.choice(YES_NO, size=n_rows, p=[0.48, 0.52]),
            "Dependents": rng.choice(YES_NO, size=n_rows, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": np.where(
                phone_service == "No", "No phone service", rng.choice(YES_NO, size=n_rows)
            ),
            "InternetService": internet_service,
            "OnlineSecurity": _yesno_dependent_on_internet(rng, internet_service),
            "OnlineBackup": _yesno_dependent_on_internet(rng, internet_service),
            "DeviceProtection": _yesno_dependent_on_internet(rng, internet_service),
            "TechSupport": _yesno_dependent_on_internet(rng, internet_service),
            "StreamingTV": _yesno_dependent_on_internet(rng, internet_service),
            "StreamingMovies": _yesno_dependent_on_internet(rng, internet_service),
            "Contract": contract,
            "PaperlessBilling": rng.choice(YES_NO, size=n_rows, p=[0.59, 0.41]),
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    return df


def make_meridian_wireless(rng: np.random.Generator) -> pd.DataFrame:
    """Clean-data telecom startup - should pass validation cleanly with a
    high feature-richness score and a green 'Ready to go' verdict."""
    return make_telecom_frame(rng, n_rows=6000, id_prefix="MW")


def make_harborline_insurance(rng: np.random.Generator) -> pd.DataFrame:
    """Thin data: ~200 rows, only 3 columns (id, a revenue-like field, a
    binary label) - should trigger both the row-count warning and the
    feature-richness warning."""
    n_rows = 200
    ids = _customer_ids(rng, n_rows, "HB")
    annual_premium = np.round(rng.normal(1450, 520, size=n_rows).clip(300, 6000), 2)
    lapsed_score = 0.18 + 0.25 * (annual_premium > 2200)
    lapsed = rng.random(n_rows) < lapsed_score
    return pd.DataFrame(
        {
            "PolicyholderID": ids,
            "AnnualPremiumUSD": annual_premium,
            "Lapsed": np.where(lapsed, "Yes", "No"),
        }
    )


def make_fernwood_retail(rng: np.random.Generator) -> pd.DataFrame:
    """Same shape/richness as Meridian but with unconventional column
    names - most are recognizable synonyms (still auto-suggest correctly),
    a few are genuinely ambiguous and must be mapped by hand."""
    df = make_telecom_frame(rng, n_rows=6000, id_prefix="FR")
    return df.rename(
        columns={
            "customerID": "cust_ref",  # ambiguous: no auto-suggestion offered
            "gender": "sex",
            "SeniorCitizen": "is_senior",
            "Partner": "has_partner",
            "Dependents": "has_dependents",
            "tenure": "tenure_months",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "Contract": "agreement_type",  # ambiguous: auto-suggests, but WRONG (matches PaymentMethod at low confidence)
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "mo_bill_usd",  # ambiguous: no auto-suggestion offered
            "TotalCharges": "total_charges",
            "Churn": "left_us",  # ambiguous: no auto-suggestion offered
        }
    )


def _null_out(rng: np.random.Generator, series: pd.Series, frac: float) -> pd.Series:
    series = series.astype(object).copy()
    mask = rng.random(len(series)) < frac
    series[mask] = np.nan
    return series


def make_cobalt_health(rng: np.random.Generator) -> pd.DataFrame:
    """Same structure as Meridian (21 columns, 6,000 rows - row count is
    intentionally sufficient) but with realistic null patterns and a couple
    of columns that are constant throughout despite looking populated.
    Meant to surface a 'not enough USABLE columns' finding even though the
    raw column count looks fine - a data-quality problem, not a
    data-quantity one."""
    df = make_telecom_frame(rng, n_rows=6000, id_prefix="CH")

    # Mild, realistic null noise on columns that should REMAIN usable
    # (15% null -> 85% non-null, still clears the 80% usability floor).
    for col in ["tenure", "Contract", "InternetService", "TechSupport", "TotalCharges", "MonthlyCharges"]:
        df[col] = _null_out(rng, df[col], 0.15)

    # Heavier null damage on columns meant to become UNUSABLE (>20% null
    # drops non-null rate below the 80% floor).
    for col in ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies", "PaymentMethod"]:
        df[col] = _null_out(rng, df[col], 0.28)

    # Constant-value columns that still look fully populated (no nulls) -
    # a stuck-default-value ETL bug, not a missing-data problem.
    df["PaperlessBilling"] = "Yes"
    df["OnlineSecurity"] = "No"

    return df


REDLINE_PAYLOADS = ["=cmd|'/c calc'!A1", "+1+1", "@SUM(1,1)", "-2+3"]


def make_redline_motors(rng: np.random.Generator) -> pd.DataFrame:
    """Same shape as Meridian, but with formula-injection payloads planted
    in a few otherwise-normal-looking cells. Every plausible-looking value
    (MonthlyCharges/TotalCharges kept non-negative on purpose) is
    untouched elsewhere, so the only reason this upload gets rejected is
    the planted payloads - demonstrating the security layer, not a broken
    fixture. The upload endpoint rejects on the FIRST offending cell it
    scans (fail-fast), so only one of these four ever surfaces in the
    actual API response even though all four are planted - the first is
    placed in row 1 so it's the one a live demo will always see."""
    df = make_telecom_frame(rng, n_rows=2000, id_prefix="RM")
    df.loc[0, "PaymentMethod"] = REDLINE_PAYLOADS[0]
    df.loc[120, "gender"] = REDLINE_PAYLOADS[1]
    df.loc[300, "Contract"] = REDLINE_PAYLOADS[2]
    df.loc[450, "MultipleLines"] = REDLINE_PAYLOADS[3]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    datasets = [
        ("company_meridian_wireless.csv", make_meridian_wireless(rng)),
        ("company_harborline_insurance.csv", make_harborline_insurance(rng)),
        ("company_fernwood_retail.csv", make_fernwood_retail(rng)),
        ("company_cobalt_health.csv", make_cobalt_health(rng)),
        ("company_redline_motors.csv", make_redline_motors(rng)),
    ]

    for filename, df in datasets:
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        print(f"{filename}: {len(df):,} rows, {len(df.columns)} columns -> {path}")


if __name__ == "__main__":
    main()
