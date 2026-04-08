"""Inference utilities for serving meta-algorithm predictions."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, MetaAlgorithmsConfig
from .data import get_iris_class_names
from .pipeline import BlendingPipeline


class MetaAlgorithmRequest(BaseModel):
    """Request schema for iris classification via meta-algorithms."""

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


class MetaAlgorithmResponse(BaseModel):
    """Response schema returned by meta-algorithm inference calls."""

    predicted_class: int
    predicted_class_name: str
    confidence_scores: list[float]
    ensemble_type: str
    model_version: str

    model_config = ConfigDict(use_enum_values=True)


class MetaAlgorithmService:
    """High-level service object for blending ensemble inference."""

    def __init__(self, config: MetaAlgorithmsConfig | None = None):
        """Initialize service with optional custom config."""
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.class_names = get_iris_class_names()

    def _load_or_train(self) -> BlendingPipeline:
        """Load existing model or train new one."""
        pipeline = BlendingPipeline(self.config)
        artifacts_dir = Path(self.config.model_path).parent

        if (artifacts_dir / "meta_learner.pkl").exists():
            pipeline.load_artifacts()
            print("Loaded existing meta-algorithm artifacts")
        else:
            print("Training new meta-algorithm model...")
            pipeline.train()

        return pipeline

    def predict(self, payload: MetaAlgorithmRequest) -> MetaAlgorithmResponse:
        """Generate prediction for iris flower classification.

        Args:
            payload: Request containing iris features

        Returns:
            Response with prediction, class name, and confidence scores
        """
        X = np.array(
            [
                [
                    payload.sepal_length,
                    payload.sepal_width,
                    payload.petal_length,
                    payload.petal_width,
                ]
            ]
        )

        predictions, probabilities = self.pipeline.predict(X)
        predicted_class = int(predictions[0])
        confidence_scores = probabilities[0].tolist()
        class_name = self.class_names.get(predicted_class, "Unknown")

        return MetaAlgorithmResponse(
            predicted_class=predicted_class,
            predicted_class_name=class_name,
            confidence_scores=confidence_scores,
            ensemble_type="blending",
            model_version="1.0.0",
        )


@lru_cache(maxsize=1)
def get_service() -> MetaAlgorithmService:
    """Factory function to get or create cached service instance."""
    return MetaAlgorithmService()


# Schema exports for FastAPI
RequestModel = MetaAlgorithmRequest
ResponseModel = MetaAlgorithmResponse
