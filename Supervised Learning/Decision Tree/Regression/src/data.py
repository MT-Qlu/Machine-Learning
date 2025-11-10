"""Data loading utilities for the California housing decision tree regressor."""
from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import CONFIG, DecisionTreeRegressionConfig


_RENAME_MAP = {
    "MedInc": "median_income",
    "HouseAge": "house_age",
    "AveRooms": "average_rooms",
    "AveBedrms": "average_bedrooms",
    "Population": "population",
    "AveOccup": "average_occupancy",
    "Latitude": "latitude",
    "Longitude": "longitude",
}


def _load_from_source(config: DecisionTreeRegressionConfig) -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df = df.rename(columns=_RENAME_MAP | {"MedHouseVal": config.target_column})
    config.data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.data_path, index=False)
    return df


def load_dataset(
    config: DecisionTreeRegressionConfig = CONFIG,
) -> pd.DataFrame:
    """Load the dataset from disk, fetching from scikit-learn if needed."""
    if config.data_path.exists():
        df = pd.read_csv(config.data_path)
    else:
        df = _load_from_source(config)
    return df


def build_features(
    df: pd.DataFrame, config: DecisionTreeRegressionConfig = CONFIG
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[list(config.feature_columns)]
    y = df[config.target_column].astype(float)
    return X, y


def train_validation_split(
    config: DecisionTreeRegressionConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_dataset(config)
    X, y = build_features(df, config)
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
