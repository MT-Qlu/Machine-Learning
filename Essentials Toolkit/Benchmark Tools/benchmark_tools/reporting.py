"""Utilities for presenting and persisting benchmark results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import OutputConfig


@dataclass
class MetricScore:
    """Score for a model/metric pair across folds."""

    metric: str
    fold_scores: List[float]

    @property
    def mean(self) -> float:
        return float(sum(self.fold_scores) / len(self.fold_scores))

    @property
    def std(self) -> float:
        if len(self.fold_scores) <= 1:
            return 0.0
        mean = self.mean
        variance = sum((score - mean) ** 2 for score in self.fold_scores) / (len(self.fold_scores) - 1)
        return float(variance ** 0.5)


@dataclass
class ModelReport:
    """Aggregated performance for a single model."""

    name: str
    scores: List[MetricScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            **{score.metric: score.mean for score in self.scores},
            **{f"{score.metric}_std": score.std for score in self.scores},
        }

    def detailed(self) -> Dict[str, Any]:
        return {
            "model": self.name,
            "metrics": [
                {
                    "name": score.metric,
                    "fold_scores": score.fold_scores,
                    "mean": score.mean,
                    "std": score.std,
                }
                for score in self.scores
            ],
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark outputs ready for persistence."""

    results: List[ModelReport]
    metadata: Dict[str, Any]

    def as_dataframe(self) -> pd.DataFrame:
        rows = [report.to_dict() for report in self.results]
        return pd.DataFrame(rows)

    def as_json(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "results": [report.detailed() for report in self.results],
        }

    def persist(self, output: OutputConfig) -> None:
        directory = Path(output.directory)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = ""
        if output.timestamped:
            timestamp = datetime.utcnow().strftime("_%Y%m%dT%H%M%SZ")

        if output.save_csv:
            frame = self.as_dataframe()
            frame.to_csv(directory / f"benchmark_summary{timestamp}.csv", index=False)

        if output.save_json:
            payload = self.as_json()
            with (directory / f"benchmark_details{timestamp}.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
