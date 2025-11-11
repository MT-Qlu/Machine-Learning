"""Configuration objects and helpers for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator


class DatasetConfig(BaseModel):
    """Configuration describing how to load the evaluation dataset."""

    loader: str = Field(
        ..., description="Loader alias, dotted path to a loader function, or a file path."
    )
    target: Optional[str] = Field(
        None,
        description="Target column name (for tabular data sources)."
    )
    features: Optional[List[str]] = Field(
        None,
        description="Subset of feature columns to retain when working with tabular files."
    )
    options: Dict[str, Any] = Field(default_factory=dict)
    drop_na: bool = True


class ModelConfig(BaseModel):
    """Definition of a model that will participate in the benchmark."""

    name: str
    estimator: str = Field(
        ..., description="Alias or dotted path pointing to an estimator class."
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MetricConfig(BaseModel):
    """Metrics to compute for each fitted estimator."""

    name: str
    alias: Optional[str] = None
    prediction: Literal["labels", "probabilities", "decision"] = "labels"
    average: Optional[str] = None
    greater_is_better: Optional[bool] = None

    @validator("prediction")
    def guard_prediction_support(cls, value: str) -> str:  # noqa: D417
        if value not in {"labels", "probabilities", "decision"}:
            raise ValueError("Unsupported prediction type.")
        return value


class SplitConfig(BaseModel):
    """Train/validation split strategy."""

    strategy: Literal["train_test", "kfold", "stratified_kfold"] = "train_test"
    test_size: float = 0.2
    random_state: Optional[int] = 42
    shuffle: bool = True
    folds: int = 5

    @validator("test_size")
    def validate_test_size(cls, value: float) -> float:  # noqa: D417
        if not 0.0 < value < 1.0:
            raise ValueError("test_size must lie in the open interval (0, 1).")
        return value

    @validator("folds")
    def validate_folds(cls, value: int) -> int:  # noqa: D417
        if value < 2:
            raise ValueError("Cross-validation requires at least two folds.")
        return value


class OutputConfig(BaseModel):
    """Persistence options for benchmark artefacts."""

    directory: Path = Field(
        Path("Essentials Toolkit/Benchmark Tools/output"),
        description="Directory where benchmark results will be written.",
    )
    save_csv: bool = True
    save_json: bool = True
    timestamped: bool = True


class BenchmarkConfig(BaseModel):
    """Top-level configuration for running a benchmark suite."""

    name: str
    task_type: Literal["classification", "regression"]
    description: Optional[str] = None
    dataset: DatasetConfig
    models: List[ModelConfig]
    metrics: List[MetricConfig]
    split: SplitConfig = Field(default_factory=SplitConfig)
    output: Optional[OutputConfig] = Field(default_factory=OutputConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("models")
    def ensure_models(cls, value: Iterable[ModelConfig]) -> List[ModelConfig]:  # noqa: D417
        items = list(value)
        if not items:
            raise ValueError("At least one model must be specified for benchmarking.")
        return items

    @validator("metrics")
    def ensure_metrics(cls, value: Iterable[MetricConfig]) -> List[MetricConfig]:  # noqa: D417
        items = list(value)
        if not items:
            raise ValueError("At least one metric must be specified for benchmarking.")
        return items


def load_benchmark_config(source: Path | str | Dict[str, Any]) -> BenchmarkConfig:
    """Load benchmark configuration from a YAML/JSON file or dictionary."""

    if isinstance(source, BenchmarkConfig):
        return source

    if isinstance(source, dict):
        return BenchmarkConfig(**source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            raise ValueError("Configuration files must be YAML or JSON.")

    if not isinstance(payload, dict):
        raise TypeError("Configuration payload must be a dictionary.")

    return BenchmarkConfig(**payload)
