"""Configuration schema for ICA decompositions."""

from dataclasses import dataclass


@dataclass(slots=True)
class ICAConfig:
    """Default hyperparameters for FastICA."""

    n_components: int | None = None
    max_iter: int = 400
    tol: float = 1e-4
    random_state: int = 42
