"""Training pipeline utilities for the California housing decision tree regressor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import CONFIG, DecisionTreeRegressionConfig
from .data import train_validation_split


class CaliforniaDecisionTreePipeline:
    """Compose decision tree regression training, evaluation, and persistence."""

    def __init__(self, config: DecisionTreeRegressionConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "regressor",
                    DecisionTreeRegressor(
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state,
                        ccp_alpha=self.config.ccp_alpha,
                    ),
                )
            ]
        )

    def train(self) -> dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        preds = np.asarray(self.pipeline.predict(X_val))
        r2 = float(r2_score(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))

        regressor: DecisionTreeRegressor = self.pipeline.named_steps["regressor"]
        depth = float(regressor.get_depth())
        leaves = float(regressor.get_n_leaves())

        metrics: dict[str, float] = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "tree_depth": depth,
            "num_leaves": leaves,
        }
        return metrics

    def save(self) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not trained; call train() before save().")
        self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.config.model_path)
        return self.config.model_path

    @staticmethod
    def load(path: Path | None = None) -> Pipeline:
        pipeline_path = path or CONFIG.model_path
        return joblib.load(pipeline_path)

    def write_metrics(self, metrics: dict[str, float]) -> Path:
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        return self.config.metrics_path


def train_and_persist(
    config: DecisionTreeRegressionConfig | None = None,
) -> dict[str, float]:
    pipeline = CaliforniaDecisionTreePipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
