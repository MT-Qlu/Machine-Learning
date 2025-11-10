"""CLI stub for running anomaly detection experiments."""

from __future__ import annotations

from pathlib import Path

from src.config import AnomalyDetectionConfig
from src.train import train


def main() -> None:
    config = AnomalyDetectionConfig()
    data_path = Path("data/dataset.csv")
    print("Training placeholder anomaly detector...")
    model_path = train(config, data_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    raise SystemExit(main())
