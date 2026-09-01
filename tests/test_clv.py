from pathlib import Path

from src.models.clv import (
    clv_estimate_sanity_check,
    estimate_clv_bulk,
    load_clv_estimate_summary,
    train_clv_model,
)

ROOT = Path(__file__).resolve().parents[1]


def test_r2_meaningfully_above_zero():
    result = train_clv_model(
        model_dir=ROOT / "models" / "v1_enriched",
        data_path=ROOT / "data" / "raw" / "telco_enriched.csv",
    )
    assert result["r2"] > 0.1


def test_r2_below_leakage_sanity_ceiling():
    # Same pattern as the survival model's <0.90 check: CLTV's correlation
    # with MonthlyCharges/tenure/TotalCharges is well below 0.9, so nothing
    # was excluded on leakage grounds, but this still guards against a
    # future change accidentally reintroducing a near-deterministic feature.
    result = train_clv_model(
        model_dir=ROOT / "models" / "v1_enriched",
        data_path=ROOT / "data" / "raw" / "telco_enriched.csv",
    )
    assert result["r2"] < 0.95


# --- estimate_clv_bulk()/clv_estimate_sanity_check(): the formula-based CLV
# PROXY for a tenant with no real clv_column at all (see
# src/models/tenant_training.py's _run_optional_modules() - clv_estimated is
# only ever attempted in exactly that situation). Banking (models/banking_v1)
# has a real, mapped duration_column (Tenure) and a real, passing survival
# model, but genuinely no CLV-equivalent column in bank_churn.csv - the
# concrete, real tenant this feature unlocks (see config/config.yaml's
# banking entry).

BANKING_MODEL_DIR = ROOT / "models" / "banking_v1"
BANKING_DATA_PATH = ROOT / "data" / "raw" / "bank_churn.csv"


def test_estimate_clv_bulk_for_banking_produces_differentiated_nonnegative_estimates():
    result = estimate_clv_bulk(model_dir=BANKING_MODEL_DIR, data_path=BANKING_DATA_PATH)
    assert len(result) == 10000  # bank_churn.csv's real row count
    assert (result["estimated_clv"] >= 0).all()
    # Real differentiation, not every customer getting the same number -
    # Balance (Banking's revenue_column) varies a lot across customers, so
    # the estimate should too.
    assert result["estimated_clv"].nunique() > 1000


def test_clv_estimate_sanity_check_passes_for_banking():
    reason, metrics = clv_estimate_sanity_check(model_dir=BANKING_MODEL_DIR, data_path=BANKING_DATA_PATH)
    assert reason is None
    assert metrics["n_non_finite"] == 0
    assert metrics["n_negative"] == 0
    assert metrics["coefficient_of_variation"] > 0.05


def test_load_clv_estimate_summary_for_banking_discloses_methodology_and_has_sane_shape():
    summary = load_clv_estimate_summary(model_dir=BANKING_MODEL_DIR, data_path=BANKING_DATA_PATH, top_n=5)
    assert summary["estimated"] is True
    assert "not a trained regression" in summary["methodology_note"]
    assert summary["count"] == 10000
    assert summary["min"] <= summary["median"] <= summary["max"]
    assert len(summary["top_customers"]) == 5
    # Sorted descending by estimated_clv.
    values = [c["estimated_clv"] for c in summary["top_customers"]]
    assert values == sorted(values, reverse=True)
    assert values[0] == summary["max"]


def test_clv_estimate_sanity_check_catches_a_degenerate_flat_estimate(tmp_path):
    """Every customer sharing the exact same revenue_column value (and no
    duration/survival to differentiate the forward-looking term either)
    collapses estimated_clv to the same number for everyone - a real,
    structural failure this gate exists to catch, not a hypothetical one."""
    import pandas as pd

    n = 200
    df = pd.DataFrame(
        {
            "cust_id": [f"C{i:04d}" for i in range(n)],
            "monthly_fee": [50.0] * n,  # identical for every customer, on purpose
            "left": ["Yes" if i % 5 == 0 else "No" for i in range(n)],
        }
    )
    data_path = tmp_path / "flat.csv"
    df.to_csv(data_path, index=False)

    tenant_config = {
        "id_column": "cust_id",
        "target_column": "left",
        "target_positive_value": "Yes",
        "revenue_column": "monthly_fee",
        "tuning_enabled": False,
    }
    # No survival_model.pkl / segment_model.pkl at all in this empty
    # tmp_path model_dir - estimate_clv_bulk() must degrade gracefully
    # (flat FALLBACK_SURVIVAL_MONTHS_FOR_INESTIMABLE_CONTRACT for everyone)
    # rather than crash on missing artifacts.
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    reason, metrics = clv_estimate_sanity_check(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    assert reason is not None
    assert "does not meaningfully differentiate" in reason
    assert metrics["coefficient_of_variation"] == 0.0


def test_estimate_clv_bulk_for_a_freshly_trained_tenant_with_non_telco_column_names(tmp_path):
    """Self-contained variant (trains its own tiny synthetic tenant, WITH a
    real duration column and a fitted survival model, in tmp_path) so the
    survival-informed branch - Banking's real data always exercises it, but
    a from-scratch guard should too - runs anywhere, CI included."""
    import numpy as np
    import pandas as pd

    from src.models.survival import train_survival_model
    from src.models.train import train_model

    CONFIG_PATH = ROOT / "config" / "config.yaml"

    rng = np.random.default_rng(7)
    n = 400
    tenure_like = rng.integers(0, 60, size=n)
    fee = np.round(rng.normal(30, 8, size=n).clip(5, 60), 2)
    plan = rng.choice(["Alpha", "Beta"], size=n, p=[0.5, 0.5])
    logit = -0.3 - 0.02 * tenure_like + 0.9 * (plan == "Alpha") + rng.normal(0, 0.6, size=n)
    left = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame(
        {
            "member_id": [f"M{i:04d}" for i in range(n)],
            "months_active": tenure_like,
            "dues": fee,
            "plan_name": plan,
            "left": left,
        }
    )
    data_path = tmp_path / "synthetic.csv"
    df.to_csv(data_path, index=False)

    tenant_config = {
        "id_column": "member_id",
        "target_column": "left",
        "target_positive_value": "Yes",
        "revenue_column": "dues",
        "duration_column": "months_active",
        "tuning_enabled": False,
    }
    model_dir = tmp_path / "model"
    train_model(data_path, CONFIG_PATH, output_dir=model_dir, tenant_config=tenant_config)
    train_survival_model(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)

    reason, metrics = clv_estimate_sanity_check(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    assert reason is None
    assert metrics["coefficient_of_variation"] > 0.05

    result = estimate_clv_bulk(model_dir=model_dir, data_path=data_path, tenant_config=tenant_config)
    assert len(result) == n
    assert (result["estimated_clv"] >= 0).all()
