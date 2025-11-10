"""Configuration schema for DBSCAN experiments."""

from dataclasses import dataclass


@dataclass(slots=True)
class DBSCANConfig:
    """Default hyperparameters for DBSCAN clustering."""

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    n_jobs: int | None = None
