"""Configuration for meta-algorithms models."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetaAlgorithmsConfig:
    """Configuration for blending ensemble and other meta-algorithms."""

    # Blending configuration
    blending_holdout_ratio: float = 0.25  # Holdout set ratio for blending
    blending_random_state: int = 42

    # Base learners configuration
    n_estimators_rf: int = 100
    n_estimators_gb: int = 100
    max_depth_dt: int = 5
    n_neighbors_knn: int = 5

    # Meta-learner configuration
    meta_learner_type: str = "logistic"  # logistic, svm, or rf
    meta_learner_C: float = 1.0
    meta_learner_max_iter: int = 1000

    # Paths
    model_path: str = "artifacts/meta_algorithms_model.pkl"
    scaler_path: str = "artifacts/scaler.pkl"
    metrics_path: str = "artifacts/metrics.json"

    # Data
    random_state: int = 42
    test_size: float = 0.2


# Global config instance
CONFIG = MetaAlgorithmsConfig()
