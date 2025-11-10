"""Pipeline constructors for Independent Component Analysis."""

from __future__ import annotations

from sklearn.decomposition import FastICA

from .config import ICAConfig


def build_pipeline(config: ICAConfig) -> FastICA:
    """Instantiate a FastICA transformer."""

    return FastICA(
        n_components=config.n_components,
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=config.random_state,
    )
