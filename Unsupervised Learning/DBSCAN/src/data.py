"""Dataset utilities for DBSCAN clustering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_features(csv_path: Path) -> pd.DataFrame:
    """Load features from a CSV file for DBSCAN clustering."""

    return pd.read_csv(csv_path)
