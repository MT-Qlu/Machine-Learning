"""Dataset utilities for K-Means clustering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_features(csv_path: Path) -> pd.DataFrame:
    """Load features for K-Means clustering."""

    return pd.read_csv(csv_path)
