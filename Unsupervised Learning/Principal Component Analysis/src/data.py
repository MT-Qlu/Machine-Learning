"""Dataset utilities for PCA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_features(csv_path: Path) -> pd.DataFrame:
    """Load features for PCA."""

    return pd.read_csv(csv_path)
