"""Analysis scaffolding for unsupervised time-series diagnostics."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import TimeSeriesAnalysisConfig
from .data import load_series
from .pipeline import seasonal_decomposition


def analyse(config: TimeSeriesAnalysisConfig, data_path: Path, *, artefact_dir: Path | None = None) -> Path:
    """Run decomposition and persist results for later inspection."""

    series = load_series(data_path)
    decomposition = seasonal_decomposition(series, config)

    artefact_dir = artefact_dir or Path("artifacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)
    artefact_path = artefact_dir / "decomposition.joblib"
    joblib.dump(decomposition, artefact_path)
    return artefact_path
