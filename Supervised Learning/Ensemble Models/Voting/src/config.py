"""Configuration for Voting Ensemble."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VotingConfig:
    """Configuration for voting ensemble."""

    data_path: str = "data/iris.csv"
    model_path: str = "artifacts/voting_model.pkl"

    feature_columns: list[str] = field(
        default_factory=lambda: [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    )
    target_column: str = "species"

    test_size: float = 0.2
    random_state: int = 42
    
    voting_type: str = "soft"  # "hard" or "soft"
    
    # Optional weights for each base learner
    weights: list[float] | None = None  # None = equal weights


CONFIG = VotingConfig()
