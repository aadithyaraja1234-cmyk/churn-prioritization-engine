"""run_pipeline.py's --output-dir has no default on purpose - the
underlying train_model() defaults to models/v1 when output_dir is
omitted, which would silently overwrite the live, validated Telco
reference model in place. Running the script without --output-dir must
fail clearly (a real, non-zero exit) rather than silently succeeding into
models/v1."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_running_without_output_dir_fails_clearly_instead_of_defaulting_to_models_v1():
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--output-dir" in result.stderr
    assert "required" in result.stderr.lower()


def test_running_with_explicit_output_dir_trains_and_writes_there_not_models_v1(tmp_path):
    output_dir = tmp_path / "pipeline_test_output"
    models_v1_metadata = ROOT / "models" / "v1" / "metadata.json"
    models_v1_mtime_before = models_v1_metadata.stat().st_mtime

    result = subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py"), "--output-dir", str(output_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "model.pkl").exists()
    assert (output_dir / "metadata.json").exists()
    # The real, live reference model must be completely untouched.
    assert models_v1_metadata.stat().st_mtime == models_v1_mtime_before
