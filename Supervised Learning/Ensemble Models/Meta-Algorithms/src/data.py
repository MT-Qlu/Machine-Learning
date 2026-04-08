"""Data loading utilities for meta-algorithms demonstration."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler as ScalerType


def load_iris_data(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load and preprocess Iris dataset for meta-algorithm training.

    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_iris_class_names() -> dict[int, str]:
    """Return mapping of Iris class indices to names."""
    return {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
