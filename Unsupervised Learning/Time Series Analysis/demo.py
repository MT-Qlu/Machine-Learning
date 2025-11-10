"""CLI stub for unsupervised time-series analysis."""

from __future__ import annotations

from pathlib import Path

from src.config import TimeSeriesAnalysisConfig
from src.train import analyse


def main() -> None:
    config = TimeSeriesAnalysisConfig()
    data_path = Path("data/series.csv")
    print("Running seasonal decomposition placeholder...")
    artefact_path = analyse(config, data_path)
    print(f"Decomposition stored at {artefact_path}")


if __name__ == "__main__":
    raise SystemExit(main())
