"""Data loading utilities for anomaly detection datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Load the dataset from disk, returning features and optional labels.

    For pure unsupervised workflows the label series will be ``None``. When
    anomaly labels exist (for benchmarking purposes) they can be surfaced for
    evaluation.
    """

    frame = pd.read_csv(csv_path)
    label_columns = [col for col in frame.columns if col.lower() in {"label", "anomaly"}]
    labels = frame[label_columns[0]] if label_columns else None
    features = frame.drop(columns=label_columns) if label_columns else frame
    return features, labels
