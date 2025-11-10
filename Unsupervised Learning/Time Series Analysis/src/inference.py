"""Inference helpers for unsupervised time-series analysis."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import TimeSeriesAnalysisConfig
from .pipeline import DecompositionResult, seasonal_decomposition


def load_decomposition(artefact_path: Path) -> DecompositionResult:
    """Load a persisted decomposition artefact."""

    return joblib.load(artefact_path)


def run_decomposition(series: pd.Series, config: TimeSeriesAnalysisConfig) -> DecompositionResult:
    """Convenience helper mirroring the training pipeline."""

    return seasonal_decomposition(series, config)
