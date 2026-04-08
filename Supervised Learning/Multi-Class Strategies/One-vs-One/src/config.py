"""Configuration for One-vs-One."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OneVsOneConfig:
    """Configuration."""

    data_path: str = "data/iris.csv"
    model_path: str = "artifacts/ovo_model.pkl"

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


CONFIG = OneVsOneConfig()
