"""Intervention Benchmark - a real, cited, third-party benchmark for
intervention effectiveness, since this system has no first-party
intervention-outcome data of its own yet (no A/B test or historical
retention-offer results exist for this system's own customers).

Source: Kevin Hillstrom's MineThatData E-Mail Analytics And Data Mining
Challenge (2008) - a randomized controlled trial of 64,000 retail
customers who had purchased within the prior twelve months. One third were
randomly assigned a men's-merchandise email campaign, one third a
women's-merchandise campaign, and one third no email (control). Real,
measured outcomes - visit, conversion (purchase), spend - were tracked
over the two weeks following the campaign.

Citation: Hillstrom, Kevin. "MineThatData E-Mail Analytics And Data Mining
Challenge." MineThatData blog, March 2008.
Downloaded directly from the original author's own site:
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
(also widely mirrored via public uplift-modeling packages, e.g.
scikit-uplift, causeinfer). Verified on download: 64,000 rows, three
groups of ~21,300-21,400 each, matching the documented 1/3 RCT split.
File placed at data/raw/hillstrom.csv.

IMPORTANT - what this benchmark is, and is not:

Measured from the Hillstrom (2008) email marketing RCT - a different
domain (retail email) from this system's own customers' churn. Used as
supporting evidence for the intervention-effectiveness assumption in
business_impact.py, NOT as a measurement of this system's own customers.

It also measures a DIFFERENT QUANTITY than business_impact.py's
INTERVENTION_SUCCESS_RATE. Hillstrom measures the increase in an
unconditional, rare purchase probability (0.57% baseline) across a general
customer population, most of whom were never going to buy regardless.
INTERVENTION_SUCCESS_RATE models the probability that a retention offer,
already targeted at a customer flagged as high-churn-risk, succeeds in
preventing that specific churn event - a conditional retention-success
rate, not a population-wide purchase-uplift. This module reports
Hillstrom's real numbers as-is; it does not convert or rescale them into
an equivalent for INTERVENTION_SUCCESS_RATE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

HILLSTROM_CITATION = (
    'Hillstrom, Kevin. "MineThatData E-Mail Analytics And Data Mining Challenge." '
    "MineThatData blog, March 2008. Randomized controlled trial, 64,000 retail customers."
)
HILLSTROM_SOURCE_URL = (
    "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

_CACHE: dict[str, dict[str, Any]] = {}


def _segment_stats(df: pd.DataFrame, segment_mask: pd.Series) -> dict[str, float]:
    group = df.loc[segment_mask]
    return {
        "n": int(len(group)),
        "conversion_rate": float(group["conversion"].mean()),
        "visit_rate": float(group["visit"].mean()),
        "avg_spend": float(group["spend"].mean()),
    }


def compute_hillstrom_benchmark(data_path: str | Path = "data/raw/hillstrom.csv") -> dict[str, Any]:
    """The REAL, measured intervention effect from the Hillstrom RCT - not
    an assumption. conversion_absolute_uplift = conversion_rate_treatment -
    conversion_rate_control, computed directly from the real trial data.

    Treatment is the two email campaigns COMBINED (Mens E-Mail + Womens
    E-Mail) vs the No E-Mail control: the reusable benchmark question here
    is "does a targeted outreach email move the needle at all," not which
    specific campaign creative performs best. The per-campaign breakdown
    (treatment_mens_email / treatment_womens_email) is reported separately
    for transparency, since the two differ meaningfully in this data (mens
    email shows roughly double womens email's relative lift).

    Cached in-memory per data_path - this is a fixed historical dataset
    that never changes at runtime, so there is no reason to re-read and
    re-aggregate the 64,000-row CSV on every call.
    """
    data_path = Path(data_path)
    cache_key = str(data_path)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    df = pd.read_csv(data_path)

    control = _segment_stats(df, df["segment"] == "No E-Mail")
    treatment = _segment_stats(df, df["segment"] != "No E-Mail")
    mens = _segment_stats(df, df["segment"] == "Mens E-Mail")
    womens = _segment_stats(df, df["segment"] == "Womens E-Mail")

    absolute_uplift = treatment["conversion_rate"] - control["conversion_rate"]
    relative_uplift = absolute_uplift / control["conversion_rate"] if control["conversion_rate"] else 0.0

    result = {
        "source": HILLSTROM_CITATION,
        "source_url": HILLSTROM_SOURCE_URL,
        "note": (
            "Measured from the Hillstrom (2008) email marketing RCT - a different domain (retail email) "
            "from this system's own customers' churn. Used as supporting evidence for the "
            "intervention-effectiveness assumption below, NOT as a measurement of this system's own customers."
        ),
        "control": control,
        "treatment_combined": treatment,
        "treatment_mens_email": mens,
        "treatment_womens_email": womens,
        "conversion_absolute_uplift": absolute_uplift,
        "conversion_relative_uplift": relative_uplift,
    }
    _CACHE[cache_key] = result
    return result
