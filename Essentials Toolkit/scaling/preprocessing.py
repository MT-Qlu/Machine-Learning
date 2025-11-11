"""Feature scaling utilities implemented with NumPy."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

Array2D = NDArray[np.float64]
Params = Dict[str, np.ndarray | float]


def _validate_matrix(matrix: Array2D) -> Array2D:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array shaped (n_samples, n_features).")
    return array


def standardise(matrix: Array2D, *, axis: int = 0) -> Tuple[Array2D, Params]:
    data = _validate_matrix(matrix)
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, ddof=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    scaled = (data - mean) / std
    return scaled, {"mean": mean, "std": std}


def inverse_standardise(transformed: Array2D, params: Params) -> Array2D:
    mean = params["mean"]
    std = params["std"]
    return transformed * std + mean


def min_max_scale(matrix: Array2D, *, feature_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[Array2D, Params]:
    data = _validate_matrix(matrix)
    data_min = np.min(data, axis=0, keepdims=True)
    data_max = np.max(data, axis=0, keepdims=True)
    scale = np.where(data_max - data_min == 0, 1.0, data_max - data_min)
    min_val, max_val = feature_range
    scaled = (data - data_min) / scale
    scaled = scaled * (max_val - min_val) + min_val
    return scaled, {"min": data_min, "max": data_max, "feature_range": feature_range}


def inverse_min_max_scale(transformed: Array2D, params: Params) -> Array2D:
    data_min = params["min"]
    data_max = params["max"]
    min_val, max_val = params["feature_range"]
    scale = np.where(data_max - data_min == 0, 1.0, data_max - data_min)
    data = (transformed - min_val) / (max_val - min_val)
    return data * scale + data_min


def robust_scale(matrix: Array2D) -> Tuple[Array2D, Params]:
    data = _validate_matrix(matrix)
    median = np.median(data, axis=0, keepdims=True)
    q1 = np.percentile(data, 25, axis=0, keepdims=True)
    q3 = np.percentile(data, 75, axis=0, keepdims=True)
    iqr = np.where(q3 - q1 == 0, 1.0, q3 - q1)
    scaled = (data - median) / iqr
    return scaled, {"median": median, "iqr": iqr}


def inverse_robust_scale(transformed: Array2D, params: Params) -> Array2D:
    return transformed * params["iqr"] + params["median"]


def max_abs_scale(matrix: Array2D) -> Tuple[Array2D, Params]:
    data = _validate_matrix(matrix)
    max_abs = np.max(np.abs(data), axis=0, keepdims=True)
    max_abs = np.where(max_abs == 0, 1.0, max_abs)
    scaled = data / max_abs
    return scaled, {"max_abs": max_abs}


def inverse_max_abs_scale(transformed: Array2D, params: Params) -> Array2D:
    return transformed * params["max_abs"]


def normalise_l2(matrix: Array2D, *, axis: int = 1) -> Tuple[Array2D, Params]:
    data = _validate_matrix(matrix)
    norms = np.linalg.norm(data, ord=2, axis=axis, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return data / norms, {"norms": norms, "axis": axis}


def inverse_normalise_l2(transformed: Array2D, params: Params) -> Array2D:
    norms = params["norms"]
    return transformed * norms


def log_scale(matrix: Array2D, *, epsilon: float = 1e-9) -> Tuple[Array2D, Params]:
    data = _validate_matrix(matrix)
    if np.any(data < 0):
        raise ValueError("Log scaling requires non-negative inputs.")
    transformed = np.log1p(data + epsilon)
    return transformed, {"epsilon": epsilon}


def inverse_log_scale(transformed: Array2D, params: Params) -> Array2D:
    epsilon = params["epsilon"]
    return np.expm1(transformed) - epsilon


__all__ = [
    "standardise",
    "inverse_standardise",
    "min_max_scale",
    "inverse_min_max_scale",
    "robust_scale",
    "inverse_robust_scale",
    "max_abs_scale",
    "inverse_max_abs_scale",
    "normalise_l2",
    "inverse_normalise_l2",
    "log_scale",
    "inverse_log_scale",
]
