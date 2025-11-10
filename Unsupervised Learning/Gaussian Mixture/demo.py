"""CLI stub for Gaussian Mixture experiments."""

from __future__ import annotations

from pathlib import Path

from src.config import GaussianMixtureConfig
from src.train import train


def main() -> None:
    config = GaussianMixtureConfig()
    data_path = Path("data/dataset.csv")
    print("Training Gaussian Mixture placeholder...")
    model_path = train(config, data_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    raise SystemExit(main())
