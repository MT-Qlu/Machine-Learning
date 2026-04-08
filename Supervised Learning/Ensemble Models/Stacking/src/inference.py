"""Inference utilities for serving stacking ensemble predictions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, StackingConfig
from .pipeline import StackingEnsemblePipeline


class StackingEnsembleRequest(BaseModel):
    """Request schema for iris classification via stacking ensemble."""

    sepal_length: float = Field(..., gt=0.0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0.0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0.0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0.0, description="Petal width in cm")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    )


class StackingEnsembleResponse(BaseModel):
    """Response schema returned by inference calls."""

    predicted_class: int
    predicted_class_name: str
    confidence_scores: list[float]
    base_learners_used: list[str]
    model_version: str

    model_config = ConfigDict(use_enum_values=True)


class StackingEnsembleService:
    """High-level service object that wraps the trained stacking ensemble pipeline."""

    IRIS_CLASSES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    def __init__(self, config: StackingConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()

    def _load_or_train(self) -> Any:
        if not self.config.model_path.exists():
            pipeline = StackingEnsemblePipeline(self.config)
            pipeline.train()
        else:
            pipeline = StackingEnsemblePipeline(self.config)
            # Load trained model
            models = joblib.load(self.config.model_path)
            pipeline.base_learners = models["base_learners"]
            pipeline.meta_learner = models["meta_learner"]
        return pipeline

    def predict(self, payload: StackingEnsembleRequest) -> StackingEnsembleResponse:
        """Generate prediction using stacking ensemble."""
        features = np.array([[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]], dtype=float)

        # Generate predictions from base learners (meta-features)
        X_meta = []
        for learner in self.pipeline.base_learners.values():
            X_meta.append(learner.predict_proba(features))
        X_meta = np.hstack(X_meta)

        # Meta-learner prediction
        y_pred = self.pipeline.meta_learner.predict(X_meta)[0]
        confidence = self.pipeline.meta_learner.predict_proba(X_meta)[0]

        return StackingEnsembleResponse(
            predicted_class=int(y_pred),
            predicted_class_name=self.IRIS_CLASSES[int(y_pred)],
            confidence_scores=[float(c) for c in confidence],
            base_learners_used=list(self.pipeline.base_learners.keys()),
            model_version=self._artifact_version(self.config.model_path),
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> StackingEnsembleService:
    """Factory returning a cached service instance for FastAPI integration."""
    return StackingEnsembleService()


RequestModel = StackingEnsembleRequest
ResponseModel = StackingEnsembleResponse
