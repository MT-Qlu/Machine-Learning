"""Training pipeline utilities for the Iris decision tree classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import CONFIG, DecisionTreeClassificationConfig
from .data import train_validation_split


class IrisDecisionTreePipeline:
    """Compose decision tree training, evaluation, and persistence."""

    def __init__(self, config: DecisionTreeClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        return Pipeline(
            steps=
            [
                (
                    "classifier",
                    DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=self.config.max_depth,
                        min_samples_split=self.config.min_samples_split,
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state,
                        ccp_alpha=self.config.ccp_alpha,
                    ),
                ),
            ]
        )

    def train(self) -> dict[str, float]:
        X_train, X_val, y_train, y_val = train_validation_split(self.config)
        self.pipeline = self.build()
        self.pipeline.fit(X_train, y_train)

        preds = np.asarray(self.pipeline.predict(X_val))
        accuracy = float(accuracy_score(y_val, preds))
        macro_precision = float(precision_score(y_val, preds, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_val, preds, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

        classifier: DecisionTreeClassifier = self.pipeline.named_steps["classifier"]
        depth = float(classifier.get_depth())
        leaves = float(classifier.get_n_leaves())

        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
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
    config: DecisionTreeClassificationConfig | None = None,
) -> dict[str, float]:
    pipeline = IrisDecisionTreePipeline(config)
    metrics = pipeline.train()
    pipeline.save()
    pipeline.write_metrics(metrics)
    return metrics
