"""Dataset loading utilities."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pandas as pd
from sklearn import datasets as sk_datasets

from .config import DatasetConfig

BUILTIN_DATASETS: Dict[str, Callable[..., Any]] = {
    "iris": sk_datasets.load_iris,
    "wine": sk_datasets.load_wine,
    "breast_cancer": sk_datasets.load_breast_cancer,
    "digits": sk_datasets.load_digits,
    "california_housing": sk_datasets.fetch_california_housing,
    "diabetes": sk_datasets.load_diabetes,
}


def _dispatch_loader(alias: str) -> Callable[..., Any]:
    if alias in BUILTIN_DATASETS:
        return BUILTIN_DATASETS[alias]

    if ":" in alias:
        module_name, attr = alias.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr)

    if "." in alias:
        module_name, attr = alias.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr)

    raise ValueError(f"Unsupported dataset loader alias: {alias}")


def _load_from_callable(loader: Callable[..., Any], config: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    resource = loader(**config.options)

    if hasattr(resource, "frame") and isinstance(resource.frame, pd.DataFrame):
        frame = resource.frame.copy()
        if config.target:
            target_name = config.target
        else:
            target_series = getattr(resource, "target", None)
            if isinstance(target_series, pd.Series) and target_series.name in frame:
                target_name = target_series.name
            elif "target" in frame:
                target_name = "target"
            else:
                raise ValueError("Specify dataset.target when using frame-based loaders.")
        y = frame[target_name]
        X = frame.drop(columns=[target_name])
        return X, y

    data = getattr(resource, "data", None)
    target = getattr(resource, "target", None)

    if data is None or target is None:
        raise ValueError("Dataset loader must expose 'data' and 'target'.")

    feature_names = getattr(resource, "feature_names", None)
    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(target, name=config.target or "target")
    return X, y


def _load_from_file(path: Path, config: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path, **config.options)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, **config.options)
    else:
        raise ValueError("Unsupported file extension for dataset loading.")

    if config.drop_na:
        frame = frame.dropna()

    if not config.target or config.target not in frame:
        raise ValueError("Target column must be provided and exist in the dataset file.")

    target = frame.pop(config.target)

    if config.features:
        missing = set(config.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Requested features are missing in the dataset: {missing}")
        frame = frame[config.features]

    return frame, target


def load_dataset(config: DatasetConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset according to the dataset configuration."""

    loader_key = config.loader
    path = Path(loader_key)
    if path.exists():
        return _load_from_file(path, config)

    loader = _dispatch_loader(loader_key)
    return _load_from_callable(loader, config)
