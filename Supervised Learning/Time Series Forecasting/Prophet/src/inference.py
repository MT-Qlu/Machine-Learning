"""Inference utilities for the AirPassengers Prophet forecaster."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .config import CONFIG, ProphetConfig
from .pipeline import AirPassengersProphetPipeline, train_and_persist


class ProphetForecastRequest(BaseModel):
    """Request payload for Prophet forecasts."""

    horizon: int = Field(12, ge=1, le=60)


class ProphetForecastResponse(BaseModel):
    """Response payload returned by the FastAPI endpoint."""

    forecast: List[float]
    lower: List[float]
    upper: List[float]
    index: List[str]
    model_version: str
    metrics: dict[str, float]

    model_config = ConfigDict(use_enum_values=True)


class ProphetForecastService:
    """High-level service that wraps Prophet inference."""

    def __init__(self, config: ProphetConfig | None = None) -> None:
        self.config = config or CONFIG
        self.pipeline = self._load_or_train()
        self.metrics = self._load_metrics()

    def _load_or_train(self) -> AirPassengersProphetPipeline:
        if not self.config.model_path.exists():
            train_and_persist(self.config)
        return AirPassengersProphetPipeline.load(self.config.model_path)

    def _load_metrics(self) -> dict[str, float]:
        if self.config.metrics_path.exists():
            with self.config.metrics_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        return {}

    def predict(self, payload: ProphetForecastRequest) -> ProphetForecastResponse:
        forecast_df = self.pipeline.forecast(payload.horizon)
        model_version = self._artifact_version(self.config.model_path)
        return ProphetForecastResponse(
            forecast=[float(v) for v in forecast_df["yhat"].values],
            lower=[float(v) for v in forecast_df["yhat_lower"].values],
            upper=[float(v) for v in forecast_df["yhat_upper"].values],
            index=[ts.strftime("%Y-%m-%d") for ts in forecast_df["ds"]],
            model_version=model_version,
            metrics=self.metrics,
        )

    @staticmethod
    def _artifact_version(path: Path) -> str:
        stat = path.stat()
        return f"{int(stat.st_mtime)}"


@lru_cache(maxsize=1)
def get_service() -> ProphetForecastService:
    return ProphetForecastService()


RequestModel = ProphetForecastRequest
ResponseModel = ProphetForecastResponse
