"""Metric registry bridging Essentials utilities and scikit-learn."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
from sklearn import metrics as sk_metrics

from .config import MetricConfig


def _load_essentials_metrics() -> Dict[str, Callable[..., float]]:
    root = Path(__file__).resolve().parents[2]
    metrics_path = root / "metrics" / "metrics.py"
    if not metrics_path.exists():
        # Fallback for legacy location with spaces in the path.
        metrics_path = root.parent / "metrics" / "metrics.py"
    if not metrics_path.exists():
        metrics_path = Path(__file__).resolve().parents[2] / "Essentials Toolkit" / "metrics" / "metrics.py"

    if not metrics_path.exists():
        raise FileNotFoundError("Unable to locate Essentials metrics module.")

    module_name = "_essentials_metrics"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, metrics_path)
        if spec is None or spec.loader is None:
            raise ImportError("Failed to import Essentials metrics module.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

    registry = getattr(module, "METRIC_REGISTRY", None)
    if registry is None:
        raise AttributeError("Essentials metrics module must expose METRIC_REGISTRY.")
    return dict(registry)


_ESSENTIALS_REGISTRY = _load_essentials_metrics()

_SKLEARN_REGISTRY: Dict[str, Callable[..., float]] = {
    "accuracy": sk_metrics.accuracy_score,
    "balanced_accuracy": sk_metrics.balanced_accuracy_score,
    "precision": sk_metrics.precision_score,
    "recall": sk_metrics.recall_score,
    "f1": sk_metrics.f1_score,
    "roc_auc": sk_metrics.roc_auc_score,
    "average_precision": sk_metrics.average_precision_score,
    "log_loss": sk_metrics.log_loss,
    "explained_variance": sk_metrics.explained_variance_score,
    "max_error": sk_metrics.max_error,
    "mean_absolute_error": sk_metrics.mean_absolute_error,
    "mean_squared_error": sk_metrics.mean_squared_error,
    "root_mean_squared_error": lambda y_true, y_pred: float(
        np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))
    ),
    "r2": sk_metrics.r2_score,
    "median_absolute_error": sk_metrics.median_absolute_error,
}


def _resolve_callable(metric: MetricConfig) -> tuple[Callable[..., float], Dict[str, Any]]:
    name = metric.name
    kwargs: Dict[str, Any] = {}

    if name in _ESSENTIALS_REGISTRY:
        func = _ESSENTIALS_REGISTRY[name]
    elif name in _SKLEARN_REGISTRY:
        func = _SKLEARN_REGISTRY[name]
    else:
        raise KeyError(f"Metric '{name}' is not registered.")

    if metric.average:
        kwargs["average"] = metric.average

    if metric.prediction == "labels" and name == "log_loss":
        kwargs.setdefault("labels", None)

    return func, kwargs


class ResolvedMetric:
    """Runtime representation of a metric ready for evaluation."""

    def __init__(self, config: MetricConfig) -> None:
        self.config = config
        self.func, self.kwargs = _resolve_callable(config)
        self.identifier = config.alias or config.name
        self.prediction = config.prediction
        if config.greater_is_better is not None:
            self.greater_is_better = config.greater_is_better
        else:
            self.greater_is_better = self._infer_direction()

    def _infer_direction(self) -> bool:
        loss_like = {
            "mean_squared_error",
            "mean_absolute_error",
            "root_mean_squared_error",
            "median_absolute_error",
            "mean_absolute_percentage_error",
            "symmetric_mean_absolute_percentage_error",
            "mean_absolute_scaled_error",
            "huber",
            "log_cosh",
            "quantile",
            "log_loss",
            "binary_cross_entropy",
            "categorical_cross_entropy",
        }
        return self.config.name not in loss_like

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_prob: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
    ) -> float:
        if self.prediction == "labels":
            if y_pred is None:
                raise ValueError("Predicted labels are required for this metric.")
            return float(self.func(y_true, y_pred, **self.kwargs))

        if self.prediction == "probabilities":
            if y_prob is None:
                raise ValueError("Predicted probabilities are required for this metric.")
            return float(self.func(y_true, y_prob, **self.kwargs))

        if self.prediction == "decision":
            if y_score is None:
                raise ValueError("Decision function outputs are required for this metric.")
            return float(self.func(y_true, y_score, **self.kwargs))

        raise ValueError("Unsupported prediction modality for metric evaluation.")


def resolve_metrics(configs: Iterable[MetricConfig]) -> Iterable[ResolvedMetric]:
    """Resolve metric configurations into evaluators."""

    return [ResolvedMetric(cfg) for cfg in configs]
