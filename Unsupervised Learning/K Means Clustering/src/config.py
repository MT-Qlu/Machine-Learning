"""Configuration schema for K-Means clustering."""

from dataclasses import dataclass


@dataclass(slots=True)
class KMeansConfig:
    """Default hyperparameters for scikit-learn KMeans."""

    n_clusters: int = 8
    init: str = "k-means++"
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42
    n_init: str | int = "auto"
