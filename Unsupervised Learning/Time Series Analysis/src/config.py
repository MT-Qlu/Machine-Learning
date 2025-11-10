"""Configuration schema for unsupervised time-series analysis."""

from dataclasses import dataclass


@dataclass(slots=True)
class TimeSeriesAnalysisConfig:
    """Default configuration for decomposition and autocorrelation studies."""

    seasonal_periods: int = 12
    rolling_window: int = 24
