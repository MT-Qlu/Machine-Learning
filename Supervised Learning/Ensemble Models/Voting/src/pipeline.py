"""Voting Ensemble implementation."""
from __future__ import annotations

import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from .config import CONFIG, VotingConfig
from .data import load_iris_data, train_validation_split


class VotingEnsemblePipeline:
    """Voting ensemble combining multiple base learners."""

    def __init__(self, config: VotingConfig | None = None) -> None:
        self.config = config or CONFIG
        self.hard_voting = None
        self.soft_voting = None

    def _create_base_learners(self):
        """Create base learner instances."""
        return [
            ("lr", Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=200, random_state=self.config.random_state)),
            ])),
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=self.config.random_state)),
            ("knn", Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ])),
        ]

    def train(self) -> dict[str, dict[str, float]]:
        """Train voting ensemble with both hard and soft voting."""
        # Load data
        X, y = load_iris_data(self.config)
        X_train, X_test, y_train, y_test = train_validation_split(X, y, self.config)

        base_learners = self._create_base_learners()

        # Hard Voting
        self.hard_voting = VotingClassifier(
            estimators=base_learners,
            voting="hard",
            weights=self.config.weights,
        )
        self.hard_voting.fit(X_train, y_train)
        y_pred_hard = self.hard_voting.predict(X_test)

        metrics_hard = {
            "accuracy": float(accuracy_score(y_test, y_pred_hard)),
            "precision_weighted": float(precision_score(y_test, y_pred_hard, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred_hard, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_hard, average="weighted", zero_division=0)),
        }

        # Soft Voting
        base_learners = self._create_base_learners()  # Fresh instances
        self.soft_voting = VotingClassifier(
            estimators=base_learners,
            voting="soft",
            weights=self.config.weights,
        )
        self.soft_voting.fit(X_train, y_train)
        y_pred_soft = self.soft_voting.predict(X_test)

        metrics_soft = {
            "accuracy": float(accuracy_score(y_test, y_pred_soft)),
            "precision_weighted": float(precision_score(y_test, y_pred_soft, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred_soft, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_soft, average="weighted", zero_division=0)),
        }

        # Save models
        joblib.dump({
            "hard_voting": self.hard_voting,
            "soft_voting": self.soft_voting,
        }, self.config.model_path)

        return {
            "hard_voting": metrics_hard,
            "soft_voting": metrics_soft,
        }
