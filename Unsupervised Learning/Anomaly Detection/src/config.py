"""Configuration schema for anomaly detection experiments."""

from dataclasses import dataclass


@dataclass(slots=True)
class AnomalyDetectionConfig:
    """Default hyperparameters for the anomaly-detection pipeline."""

    model_name: str = "isolation_forest"
    contamination: float = 0.05
    random_state: int = 42
    n_estimators: int = 200
