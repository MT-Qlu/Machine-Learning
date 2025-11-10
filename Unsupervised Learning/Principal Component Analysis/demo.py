"""CLI stub for PCA."""

from __future__ import annotations

from pathlib import Path

from src.config import PCAConfig
from src.train import train


def main() -> None:
    config = PCAConfig()
    data_path = Path("data/dataset.csv")
    print("Training PCA placeholder...")
    model_path = train(config, data_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    raise SystemExit(main())
