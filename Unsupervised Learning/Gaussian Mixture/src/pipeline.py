"""Pipeline constructors for Gaussian Mixture Models."""

from __future__ import annotations

from sklearn.mixture import GaussianMixture

from .config import GaussianMixtureConfig


def build_pipeline(config: GaussianMixtureConfig) -> GaussianMixture:
    """Instantiate a GaussianMixture estimator with repo defaults."""

    return GaussianMixture(
        n_components=config.n_components,
        covariance_type=config.covariance_type,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state,
    )
