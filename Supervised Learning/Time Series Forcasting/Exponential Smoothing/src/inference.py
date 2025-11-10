"""Inference utilities for the AirPassengers Exponential Smoothing forecaster."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, ExponentialSmoothingConfig
from .pipeline import AirPassengersExponentialSmoothingPipeline, train_and_persist


class ExponentialSmoothingForecastRequest(BaseModel):
    """Request payload for Exponential Smoothing forecasts."""

    horizon: int = Field(12, ge=1, le=60)


class ExponentialSmoothingForecastResponse(BaseModel):
    """Response payload returned by the FastAPI endpoint."""

    forecast: List[float]
    index: List[str]
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class ExponentialSmoothingForecastService:
    """High-level service that wraps Holt-Winters inference."""

    def __init__(self, config: ExponentialSmoothingConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> AirPassengersExponentialSmoothingPipeline:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return AirPassengersExponentialSmoothingPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: ExponentialSmoothingForecastRequest) -> ExponentialSmoothingForecastResponse:
        forecast_series = self.pipeline.forecast(payload.horizon)
        model_version = self._artifact_version(self.config.model_path)
        return ExponentialSmoothingForecastResponse(
            forecast=[float(v) for v in forecast_series.values],
            index=[ts.strftime("%Y-%m-%d") for ts in forecast_series.index],
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> ExponentialSmoothingForecastService:
    return ExponentialSmoothingForecastService()


RequestModel = ExponentialSmoothingForecastRequest
ResponseModel = ExponentialSmoothingForecastResponse
