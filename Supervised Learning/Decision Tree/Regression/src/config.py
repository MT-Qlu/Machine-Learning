"""Configuration for the California housing decision tree regressor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DecisionTreeRegressionConfig:
    """Immutable configuration shared across the decision tree regressor."""

    target_column: str = "median_house_value"
    feature_columns: tuple[str, ...] = (
        "median_income",
        "house_age",
        "average_rooms",
        "average_bedrooms",
        "population",
        "average_occupancy",
        "latitude",
        "longitude",
    )
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int | None = 12
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    ccp_alpha: float = 0.0
    data_path: Path = ALGORITHM_ROOT / "data" / "california_housing.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "decision_tree_regressor.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = DecisionTreeRegressionConfig()
CONFIG.ensure_directories()
