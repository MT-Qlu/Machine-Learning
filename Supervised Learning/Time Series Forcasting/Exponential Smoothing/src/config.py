"""Configuration for the AirPassengers Exponential Smoothing forecaster."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExponentialSmoothingConfig:
    """Immutable configuration shared across the Exponential Smoothing pipeline."""

    index_column: str = "month"
    target_column: str = "passengers"
    frequency: str = "MS"
    trend: str = "add"
    seasonal: str = "mul"
    seasonal_periods: int = 12
    train_ratio: float = 0.85
    default_horizon: int = 12
    data_path: Path = ALGORITHM_ROOT / "data" / "air_passengers.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "expsmooth_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = ExponentialSmoothingConfig()
CONFIG.ensure_directories()
