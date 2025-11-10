"""Pipeline utilities for unsupervised time-series diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from .config import TimeSeriesAnalysisConfig


@dataclass(slots=True)
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series


def seasonal_decomposition(series: pd.Series, config: TimeSeriesAnalysisConfig) -> DecompositionResult:
    """Perform additive seasonal decomposition and return the components."""

    result = seasonal_decompose(series, period=config.seasonal_periods, model="additive")
    return DecompositionResult(
        trend=result.trend.dropna(),
        seasonal=result.seasonal.dropna(),
        residual=result.resid.dropna(),
    )
