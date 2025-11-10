"""Dataset utilities for unsupervised time-series analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_series(csv_path: Path, *, timestamp_column: str = "timestamp", value_column: str = "value") -> pd.Series:
    """Load a univariate time series from a CSV file."""

    frame = pd.read_csv(csv_path, parse_dates=[timestamp_column])
    frame = frame.set_index(timestamp_column)
    return frame[value_column].sort_index()
