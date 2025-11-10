"""Inference utilities for the Iris decision tree classifier."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, DecisionTreeClassificationConfig
from .pipeline import IrisDecisionTreePipeline, train_and_persist


class IrisDecisionTreeRequest(BaseModel):
    """Input schema aligned with the Iris feature space."""

    sepal_length_cm: float = Field(..., gt=0)
    sepal_width_cm: float = Field(..., gt=0)
    petal_length_cm: float = Field(..., gt=0)
    petal_width_cm: float = Field(..., gt=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length_cm": 6.1,
                "sepal_width_cm": 3.0,
                "petal_length_cm": 4.6,
                "petal_width_cm": 1.4,
            }
        }
    )


class IrisDecisionTreeResponse(BaseModel):
    """Prediction payload served by the FastAPI endpoint."""

    predicted_label: str
    class_probabilities: dict[str, float]
    feature_importances: dict[str, float]
    model_version: str
    metrics: dict[str, float]


class IrisDecisionTreeService:
    """High-level service object for Iris classification."""

    def __init__(self, config: DecisionTreeClassificationConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self):
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return IrisDecisionTreePipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: IrisDecisionTreeRequest) -> IrisDecisionTreeResponse:
        record = {column: getattr(payload, column) for column in self.config.feature_columns}
        data = pd.DataFrame([record])
        prediction = self.pipeline.predict(data)[0]
        probabilities = self.pipeline.predict_proba(data)[0]
        classifier = self.pipeline.named_steps["classifier"]
        class_probabilities = {
            str(label): float(prob)
            for label, prob in zip(classifier.classes_, probabilities)
        }
        feature_importances = {
            feature: float(importance)
            for feature, importance in zip(self.config.feature_columns, classifier.feature_importances_)
        }
        model_version = self._artifact_version(self.config.model_path)
        return IrisDecisionTreeResponse(
            predicted_label=str(prediction),
            class_probabilities=class_probabilities,
            feature_importances=feature_importances,
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> IrisDecisionTreeService:
    return IrisDecisionTreeService()


RequestModel = IrisDecisionTreeRequest
ResponseModel = IrisDecisionTreeResponse
