"""Data loading utilities for the Prophet module."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import CONFIG, ProphetConfig


def load_dataframe(config: ProphetConfig = CONFIG) -> pd.DataFrame:
    df = pd.read_csv(config.data_path)
    df[config.date_column] = pd.to_datetime(df[config.date_column])
    df = df.sort_values(config.date_column)
    return df


def train_test_split(config: ProphetConfig = CONFIG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataframe(config)
    split_index = int(len(df) * config.train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df
