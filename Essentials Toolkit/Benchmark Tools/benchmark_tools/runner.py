"""Core execution engine for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .config import BenchmarkConfig
from .datasets import load_dataset
from .metrics import ResolvedMetric, resolve_metrics
from .models import instantiate_model
from .reporting import BenchmarkReport, MetricScore, ModelReport


@dataclass
class FoldOutcome:
    """Predictions and truth values captured for a single fold."""

    estimator: BaseEstimator
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    y_score: Optional[np.ndarray]


@dataclass
class BenchmarkSummary:
    """High-level summary of a benchmark run."""

    config: BenchmarkConfig
    report: BenchmarkReport

    def dataframe(self):
        return self.report.as_dataframe()


class BenchmarkRunner:
    """Run benchmark suites based on configuration."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.metrics: Sequence[ResolvedMetric] = list(resolve_metrics(config.metrics))
        features, target = load_dataset(config.dataset)
        self.feature_frame = features
        self.target_series = target
        self.X = np.asarray(features)
        self.y = np.asarray(target)
        self.task_type = config.task_type

    def run(self) -> BenchmarkSummary:
        model_reports: List[ModelReport] = []

        for model_cfg in self.config.models:
            estimator = instantiate_model(model_cfg)
            fold_scores = self._evaluate_estimator(estimator)
            metric_scores = [
                MetricScore(metric=metric_id, fold_scores=scores)
                for metric_id, scores in fold_scores.items()
            ]
            model_reports.append(ModelReport(name=model_cfg.name, scores=metric_scores))

        metadata = {
            "name": self.config.name,
            "task_type": self.config.task_type,
            "description": self.config.description,
            "metadata": self.config.metadata,
        }

        report = BenchmarkReport(results=model_reports, metadata=metadata)

        if self.config.output:
            report.persist(self.config.output)

        return BenchmarkSummary(config=self.config, report=report)

    def _evaluate_estimator(self, estimator: BaseEstimator) -> Dict[str, List[float]]:
        split_cfg = self.config.split
        fold_scores: Dict[str, List[float]] = {metric.identifier: [] for metric in self.metrics}

        if split_cfg.strategy == "train_test":
            X_train, X_test, y_train, y_test = train_test_split(
                self.X,
                self.y,
                test_size=split_cfg.test_size,
                random_state=split_cfg.random_state,
                shuffle=split_cfg.shuffle,
                stratify=self.y if self.task_type == "classification" else None,
            )
            estimator.fit(X_train, y_train)
            folds = [self._predict_fold(estimator, X_test, y_test)]
        else:
            folds = self._cross_validate(estimator)

        for fold in folds:
            for metric in self.metrics:
                score = metric.evaluate(
                    fold.y_true,
                    y_pred=fold.y_pred,
                    y_prob=fold.y_prob,
                    y_score=fold.y_score,
                )
                fold_scores[metric.identifier].append(score)

        return fold_scores

    def _cross_validate(self, estimator: BaseEstimator) -> List[FoldOutcome]:
        split_cfg = self.config.split
        folds: List[FoldOutcome] = []
        if self.task_type == "classification" and split_cfg.strategy == "kfold":
            cv = StratifiedKFold(
                n_splits=split_cfg.folds,
                shuffle=split_cfg.shuffle,
                random_state=split_cfg.random_state,
            )
        elif self.task_type == "classification" and split_cfg.strategy == "stratified_kfold":
            cv = StratifiedKFold(
                n_splits=split_cfg.folds,
                shuffle=split_cfg.shuffle,
                random_state=split_cfg.random_state,
            )
        else:
            cv = KFold(
                n_splits=split_cfg.folds,
                shuffle=split_cfg.shuffle,
                random_state=split_cfg.random_state,
            )

        for train_idx, test_idx in cv.split(self.X, self.y):
            estimator_clone = self._clone_estimator(estimator)
            estimator_clone.fit(self.X[train_idx], self.y[train_idx])
            folds.append(
                self._predict_fold(
                    estimator_clone,
                    self.X[test_idx],
                    self.y[test_idx],
                )
            )
        return folds

    def _predict_fold(
        self,
        estimator: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> FoldOutcome:
        y_pred = estimator.predict(X_test)
        y_prob: Optional[np.ndarray] = None
        y_score: Optional[np.ndarray] = None

        if any(metric.prediction == "probabilities" for metric in self.metrics):
            if hasattr(estimator, "predict_proba"):
                y_prob = estimator.predict_proba(X_test)
            elif hasattr(estimator, "decision_function"):
                y_prob = estimator.decision_function(X_test)
            else:
                raise AttributeError("Estimator does not provide probability estimates.")

        if any(metric.prediction == "decision" for metric in self.metrics):
            if hasattr(estimator, "decision_function"):
                y_score = estimator.decision_function(X_test)
            elif hasattr(estimator, "predict_proba"):
                y_score = estimator.predict_proba(X_test)
            else:
                raise AttributeError("Estimator does not expose decision scores.")

        return FoldOutcome(
            estimator=estimator,
            y_true=np.asarray(y_test),
            y_pred=np.asarray(y_pred),
            y_prob=None if y_prob is None else np.asarray(y_prob),
            y_score=None if y_score is None else np.asarray(y_score),
        )

    @staticmethod
    def _clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
        from sklearn.base import clone

        return clone(estimator)
