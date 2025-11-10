"""Configuration for the Iris decision tree classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ALGORITHM_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DecisionTreeClassificationConfig:
    """Immutable configuration shared across the decision tree classifier."""

    target_column: str = "species"
    class_names: tuple[str, ...] = ("setosa", "versicolor", "virginica")
    feature_columns: tuple[str, ...] = (
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
    )
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    ccp_alpha: float = 0.0
    data_path: Path = ALGORITHM_ROOT / "data" / "iris.csv"
    artifact_dir: Path = ALGORITHM_ROOT / "artifacts"
    model_path: Path = artifact_dir / "decision_tree_classifier.joblib"
    metrics_path: Path = artifact_dir / "metrics.json"

    def ensure_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)


CONFIG = DecisionTreeClassificationConfig()
CONFIG.ensure_directories()
