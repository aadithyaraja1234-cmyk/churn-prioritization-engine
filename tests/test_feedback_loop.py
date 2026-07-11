"""Tests for Phase 5: Feedback Loop Simulation.

All tests verify the MECHANISM of the feedback loop, not real model performance.
Synthetic data is appropriately labeled and segregated from real data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.feedback_loop import retrain_with_feedback, simulate_feedback_batch


def test_feedback_batch_has_synthetic_label():
    """Verify feedback batch is labeled as synthetic."""
    batch = simulate_feedback_batch(n_customers=50, retention_rate=0.3, seed=42)

    # Check data_source column exists and all values are correct
    assert "data_source" in batch.columns, "data_source column missing"
    assert (batch["data_source"] == "synthetic_feedback_not_real").all(), (
        "Not all rows labeled as synthetic feedback"
    )
    assert len(batch) > 0, "Feedback batch is empty"


def test_feedback_batch_retains_synthetic_customers():
    """Verify simulated retention actually marks some customers as retained."""
    batch = simulate_feedback_batch(n_customers=100, retention_rate=0.5, seed=42)

    # At 50% retention, we expect ~50 customers with Churn='No' (newly retained)
    # Note: exact count depends on simulation, but should be non-zero
    retained_count = (batch["Churn"] == "No").sum()
    assert retained_count > 0, "No customers marked as retained in feedback batch"
    assert retained_count <= len(batch), "More retained than total customers"


def test_v2_metadata_includes_synthetic_disclosure():
    """Verify v2 model metadata includes synthetic data disclosure."""
    # Create test output directory
    output_dir = Path("models/test_v2_feedback")

    # Run retraining (will create metadata)
    batch = simulate_feedback_batch(n_customers=50, retention_rate=0.3, seed=42)
    metadata = retrain_with_feedback(
        feedback_batch=batch,
        model_dir="models/v1",
        output_dir=output_dir,
        data_path="data/raw/telco.csv",
        config_path="config/config.yaml",
        retention_rate=0.3,
    )

    # Check disclosure fields exist
    assert "data_source" in metadata, "data_source field missing"
    assert "synthetic" in metadata["data_source"].lower(), "Synthetic label missing from data_source"

    assert "synthetic_data_disclosure" in metadata, "synthetic_data_disclosure field missing"
    assert "simulated" in metadata["synthetic_data_disclosure"].lower(), (
        "Explicit 'simulated' language missing from disclosure"
    )

    assert "feedback_batch_info" in metadata, "feedback_batch_info missing"
    assert "retention_rate_applied" in metadata["feedback_batch_info"], "retention_rate not recorded"

    # Clean up
    import shutil

    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_v1_model_untouched_after_v2_creation():
    """Verify v1 model and data are never modified by v2 creation."""
    v1_dir = Path("models/v1")
    v1_model_path = v1_dir / "model.pkl"
    v1_metadata_path = v1_dir / "metadata.json"

    # Record v1 metadata before v2 retraining
    with open(v1_metadata_path, "r", encoding="utf-8") as f:
        v1_metadata_before = json.load(f)

    v1_model_mtime_before = v1_model_path.stat().st_mtime

    # Create v2
    output_dir = Path("models/test_v2_unchanged")
    batch = simulate_feedback_batch(n_customers=50, retention_rate=0.3, seed=42)
    retrain_with_feedback(
        feedback_batch=batch,
        model_dir="models/v1",
        output_dir=output_dir,
        data_path="data/raw/telco.csv",
        config_path="config/config.yaml",
        retention_rate=0.3,
    )

    # Verify v1 unchanged
    with open(v1_metadata_path, "r", encoding="utf-8") as f:
        v1_metadata_after = json.load(f)

    v1_model_mtime_after = v1_model_path.stat().st_mtime

    assert v1_metadata_before == v1_metadata_after, "v1 metadata was modified"
    assert v1_model_mtime_before == v1_model_mtime_after, "v1 model.pkl was modified"

    # Clean up
    import shutil

    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_v2_model_files_created_in_separate_directory():
    """Verify v2 model is saved to separate directory from v1."""
    output_dir = Path("models/test_v2_separate")
    v1_dir = Path("models/v1")

    # Ensure output_dir != v1_dir
    assert output_dir != v1_dir, "Test is using same directory as v1"

    batch = simulate_feedback_batch(n_customers=50, retention_rate=0.3, seed=42)
    retrain_with_feedback(
        feedback_batch=batch,
        model_dir="models/v1",
        output_dir=output_dir,
        data_path="data/raw/telco.csv",
        config_path="config/config.yaml",
        retention_rate=0.3,
    )

    # Verify v2 files exist in output_dir
    assert (output_dir / "model.pkl").exists(), "v2 model.pkl not created"
    assert (output_dir / "encoders.pkl").exists(), "v2 encoders.pkl not created"
    assert (output_dir / "metadata.json").exists(), "v2 metadata.json not created"
    assert (output_dir / "split_indices.json").exists(), "v2 split_indices.json not created"

    # Verify v1 files are untouched
    assert (v1_dir / "model.pkl").exists(), "v1 model.pkl missing"
    assert (v1_dir / "encoders.pkl").exists(), "v1 encoders.pkl missing"

    # Clean up
    import shutil

    if output_dir.exists():
        shutil.rmtree(output_dir)
