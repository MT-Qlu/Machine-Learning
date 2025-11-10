"""Configuration schema for PCA."""

from dataclasses import dataclass


@dataclass(slots=True)
class PCAConfig:
    """Default hyperparameters for PCA decompositions."""

    n_components: int | float | None = None
    whiten: bool = False
    svd_solver: str = "auto"
    random_state: int | None = 42
