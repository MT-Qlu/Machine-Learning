"""Data utilities for voting ensemble."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from .config import VotingConfig


def load_iris_data(config: VotingConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Load Iris dataset."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df.columns = [col.replace(" (cm)", "").replace(" ", "_") for col in df.columns]
    df["species"] = iris.target
    
    X = df[config.feature_columns]
    y = df[config.target_column]
    
    return X, y


def train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: VotingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
