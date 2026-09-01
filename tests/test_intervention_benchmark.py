import pytest

from src.models.intervention_benchmark import compute_hillstrom_benchmark


@pytest.fixture(scope="module")
def benchmark():
    return compute_hillstrom_benchmark()


def test_group_sizes_match_known_dataset_structure(benchmark):
    # Known structure: ~64,000 rows split roughly evenly into three groups
    # (Mens E-Mail, Womens E-Mail, No E-Mail), each ~21,300-21,400.
    total = benchmark["control"]["n"] + benchmark["treatment_combined"]["n"]
    assert 63000 <= total <= 65000

    assert 20000 <= benchmark["control"]["n"] <= 23000
    assert 20000 <= benchmark["treatment_mens_email"]["n"] <= 23000
    assert 20000 <= benchmark["treatment_womens_email"]["n"] <= 23000

    # treatment_combined must be exactly mens + womens, not double-counted
    assert benchmark["treatment_combined"]["n"] == (
        benchmark["treatment_mens_email"]["n"] + benchmark["treatment_womens_email"]["n"]
    )


def test_conversion_uplift_is_a_real_sane_number(benchmark):
    uplift = benchmark["conversion_absolute_uplift"]
    assert uplift == pytest.approx(
        benchmark["treatment_combined"]["conversion_rate"] - benchmark["control"]["conversion_rate"]
    )
    # A real measured effect: not NaN/inf, not exactly zero, and small in
    # absolute terms (conversion is a rare event in this dataset).
    assert uplift != 0.0
    assert 0.0 < uplift < 0.10

    relative_uplift = benchmark["conversion_relative_uplift"]
    assert relative_uplift != 0.0
    assert 0.0 < relative_uplift < 5.0


def test_conversion_rates_are_valid_probabilities(benchmark):
    for group_key in ("control", "treatment_combined", "treatment_mens_email", "treatment_womens_email"):
        group = benchmark[group_key]
        assert 0.0 <= group["conversion_rate"] <= 1.0
        assert 0.0 <= group["visit_rate"] <= 1.0
        assert group["avg_spend"] >= 0.0


def test_treatment_conversion_rate_exceeds_control(benchmark):
    # The real, measured direction of the effect in this dataset: email
    # campaigns increased conversion relative to no email.
    assert benchmark["treatment_combined"]["conversion_rate"] > benchmark["control"]["conversion_rate"]


def test_metadata_includes_source_citation_and_disclaimer(benchmark):
    assert "minethatdata.com" in benchmark["source_url"]
    assert "Hillstrom" in benchmark["source"]
    assert "different domain" in benchmark["note"]
    assert "NOT as a measurement of this system's own customers" in benchmark["note"]


def test_result_is_cached_across_calls():
    first = compute_hillstrom_benchmark()
    second = compute_hillstrom_benchmark()
    assert first is second
