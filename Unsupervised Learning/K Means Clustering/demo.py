"""CLI stub for K-Means clustering."""

from __future__ import annotations

from pathlib import Path

from src.config import KMeansConfig
from src.train import train


def main() -> None:
    config = KMeansConfig()
    data_path = Path("data/dataset.csv")
    print("Training K-Means placeholder...")
    model_path = train(config, data_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    raise SystemExit(main())
