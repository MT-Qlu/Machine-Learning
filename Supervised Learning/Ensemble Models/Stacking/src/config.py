"""Configuration for Stacking Ensemble training."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StackingConfig:
    """Hyperparameters and paths for stacking ensemble."""

    # Data paths
    data_path: str = "data/iris.csv"
    model_path: str = "artifacts/stacking_model.pkl"

    # Feature and target columns
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    )
    target_column: str = "species"

    # Train/test split
    test_size: float = 0.2
    random_state: int = 42

    # Stacking parameters
    cv_folds: int = 5
    
    # Base learners
    base_learners: list[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "decision_tree",
            "knn",
        ]
    )

    # Meta-learner
    meta_learner: str = "logistic_regression"


CONFIG = StackingConfig()
