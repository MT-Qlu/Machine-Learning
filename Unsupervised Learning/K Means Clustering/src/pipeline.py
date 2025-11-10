"""Pipeline constructors for K-Means clustering."""

from __future__ import annotations

from sklearn.cluster import KMeans

from .config import KMeansConfig


def build_pipeline(config: KMeansConfig) -> KMeans:
    """Instantiate a scikit-learn KMeans estimator with repo defaults."""

    return KMeans(
        n_clusters=config.n_clusters,
        init=config.init,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state,
        n_init=config.n_init,
    )
