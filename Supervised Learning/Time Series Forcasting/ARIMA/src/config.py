"""Configuration for the AirPassengers ARIMA forecaster."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ARIMAConfig:
    """Immutable configuration shared across the ARIMA pipeline."""

    index_column: str = "month"
    target_column: str = "passengers"
    frequency: str = "MS"
    order: Tuple[int, int, int] = (1, 1, 1)
    train_ratio: float = 0.85
    default_horizon: int = 12
    data_path: Path = ALGORITHM_ROOT / "data" / "air_passengers.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "arima_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = ARIMAConfig()
CONFIG.ensure_directories()
