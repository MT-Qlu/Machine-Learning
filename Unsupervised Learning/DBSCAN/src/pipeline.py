"""Pipeline constructors for DBSCAN clustering."""

from __future__ import annotations

from sklearn.cluster import DBSCAN

from .config import DBSCANConfig


def build_pipeline(config: DBSCANConfig) -> DBSCAN:
    """Instantiate a DBSCAN clustering model with config defaults."""

    return DBSCAN(
        eps=config.eps,
        min_samples=config.min_samples,
        metric=config.metric,
        n_jobs=config.n_jobs,
    )
