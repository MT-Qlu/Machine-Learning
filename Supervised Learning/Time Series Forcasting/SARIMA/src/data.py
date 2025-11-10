"""Data loading utilities for the AirPassengers SARIMA module."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import CONFIG, SARIMAConfig


def load_dataframe(config: SARIMAConfig = CONFIG) -> pd.DataFrame:
    df = pd.read_csv(config.data_path)
    df[config.index_column] = pd.to_datetime(df[config.index_column])
    df = df.sort_values(config.index_column)
    df = df.set_index(config.index_column)
    df.index = df.index.to_period(config.frequency).to_timestamp()
    df[config.target_column] = df[config.target_column].astype(float)
    return df


def load_series(config: SARIMAConfig = CONFIG) -> pd.Series:
    df = load_dataframe(config)
    return df[config.target_column]


def train_test_split(config: SARIMAConfig = CONFIG) -> Tuple[pd.Series, pd.Series]:
    series = load_series(config)
    split_index = int(len(series) * config.train_ratio)
    train = series.iloc[:split_index]
    test = series.iloc[split_index:]
    return train, test
