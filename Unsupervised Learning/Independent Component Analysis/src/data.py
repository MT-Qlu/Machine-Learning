"""Dataset utilities for ICA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_features(csv_path: Path) -> pd.DataFrame:
    """Load features for Independent Component Analysis."""

    return pd.read_csv(csv_path)
