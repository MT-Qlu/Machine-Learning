"""Configuration for the AirPassengers Prophet forecaster."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProphetConfig:
    """Immutable configuration shared across the Prophet pipeline."""

    date_column: str = "ds"
    target_column: str = "y"
    frequency: str = "MS"
    train_ratio: float = 0.85
    default_horizon: int = 12
    seasonality_mode: str = "multiplicative"
    changepoint_prior_scale: float = 0.05
    data_path: Path = ALGORITHM_ROOT / "data" / "air_passengers.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "prophet_model.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = ProphetConfig()
CONFIG.ensure_directories()
