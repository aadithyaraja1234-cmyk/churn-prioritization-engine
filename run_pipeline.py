from __future__ import annotations

from pathlib import Path

from src.models.train import train_model


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    train_model(root / "data" / "raw" / "telco.csv", root / "config" / "config.yaml")
