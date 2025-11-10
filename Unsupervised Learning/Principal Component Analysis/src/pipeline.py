"""Pipeline constructors for PCA."""

from __future__ import annotations

from sklearn.decomposition import PCA

from .config import PCAConfig


def build_pipeline(config: PCAConfig) -> PCA:
    """Instantiate a PCA transformer with repository defaults."""

    return PCA(
        n_components=config.n_components,
        whiten=config.whiten,
        svd_solver=config.svd_solver,
        random_state=config.random_state,
    )
