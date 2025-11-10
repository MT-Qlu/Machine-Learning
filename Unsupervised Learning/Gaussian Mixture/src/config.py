"""Configuration schema for Gaussian Mixture Models."""

from dataclasses import dataclass


@dataclass(slots=True)
class GaussianMixtureConfig:
    """Default hyperparameters for expectation-maximisation."""

    n_components: int = 3
    covariance_type: str = "full"
    max_iter: int = 200
    tol: float = 1e-3
    random_state: int = 42
