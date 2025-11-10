"""CLI stub for DBSCAN clustering."""

from __future__ import annotations

from pathlib import Path

from src.config import DBSCANConfig
from src.train import train


def main() -> None:
    config = DBSCANConfig()
    data_path = Path("data/dataset.csv")
    print("Fitting DBSCAN placeholder...")
    model_path = train(config, data_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    raise SystemExit(main())
