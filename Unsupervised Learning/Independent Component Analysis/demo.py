"""CLI stub for Independent Component Analysis."""

from __future__ import annotations

from pathlib import Path

from src.config import ICAConfig
from src.train import train


def main() -> None:
    config = ICAConfig()
    data_path = Path("data/dataset.csv")
    print("Running FastICA placeholder...")
    model_path = train(config, data_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    raise SystemExit(main())
