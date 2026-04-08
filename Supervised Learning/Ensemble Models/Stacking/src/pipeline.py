"""Stacking Ensemble pipeline implementation."""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from .config import CONFIG, StackingConfig
from .data import load_iris_data, train_validation_split


class StackingEnsemblePipeline:
    """Stacking ensemble implementation combining multiple base learners."""

    def __init__(self, config: StackingConfig | None = None) -> None:
        self.config = config or CONFIG
        self.base_learners: dict[str, Any] = {}
        self.meta_learner: Any = None
        self.scaler: StandardScaler = StandardScaler()
        self.pipeline: Pipeline | None = None

    def _create_base_learners(self) -> dict[str, Any]:
        """Create base learner instances."""
        learners = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=200, random_state=self.config.random_state)),
            ]),
            "decision_tree": DecisionTreeClassifier(
                max_depth=5,
                random_state=self.config.random_state,
            ),
            "knn": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]),
        }
        return {name: learners[name] for name in self.config.base_learners}

    def _generate_base_predictions(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> pd.DataFrame:
        """Generate out-of-fold predictions from base learners using cross-validation."""
        meta_features = []
        
        for name, learner in self.base_learners.items():
            # Get cross-validated predictions (probabilities for each class)
            oof_predictions = cross_val_predict(
                learner,
                X_train,
                y_train,
                cv=self.config.cv_folds,
                method="predict_proba",
            )
            meta_features.append(oof_predictions)
        
        # Concatenate all base learner predictions
        return np.hstack(meta_features)

    def train(self) -> dict[str, float]:
        """Train stacking ensemble."""
        # Load data
        X, y = load_iris_data(self.config)
        X_train, X_test, y_train, y_test = train_validation_split(X, y, self.config)

        # Create base learners
        self.base_learners = self._create_base_learners()

        # Generate meta-features from base learners
        X_meta_train = self._generate_base_predictions(X_train, y_train)

        # Train meta-learner on meta-features
        self.meta_learner = LogisticRegression(
            max_iter=200,
            random_state=self.config.random_state,
        )
        self.meta_learner.fit(X_meta_train, y_train)

        # Retrain base learners on full training data for prediction
        for learner in self.base_learners.values():
            learner.fit(X_train, y_train)

        # Generate predictions on test set
        X_meta_test = []
        for learner in self.base_learners.values():
            X_meta_test.append(learner.predict_proba(X_test))
        X_meta_test = np.hstack(X_meta_test)

        y_pred = self.meta_learner.predict(X_meta_test)

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        # Save model
        joblib.dump({
            "base_learners": self.base_learners,
            "meta_learner": self.meta_learner,
        }, self.config.model_path)

        return metrics


# Instance for demo
_instance = None


def get_pipeline(config: StackingConfig | None = None) -> StackingEnsemblePipeline:
    """Get or create pipeline instance."""
    global _instance
    if _instance is None:
        _instance = StackingEnsemblePipeline(config)
    return _instance
