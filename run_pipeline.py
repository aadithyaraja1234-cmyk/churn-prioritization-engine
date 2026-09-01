from __future__ import annotations

import argparse
from pathlib import Path

from src.models.train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain the Telco reference model from data/raw/telco.csv. "
        "--output-dir has no default on purpose - src.models.train.train_model() itself "
        "defaults to models/v1 when output_dir is omitted, which would silently overwrite "
        "the live, validated Telco model in place. Pass --output-dir models/v1 explicitly "
        "if that's really what you want."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write model.pkl/encoders.pkl/metadata.json/split_indices.json.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    metadata = train_model(
        root / "data" / "raw" / "telco.csv",
        root / "config" / "config.yaml",
        output_dir=args.output_dir,
    )
    print(f"Trained model written to {args.output_dir} (roc_auc={metadata['roc_auc']:.3f}, pr_auc={metadata['pr_auc']:.3f})")
