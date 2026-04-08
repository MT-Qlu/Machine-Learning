"""One-vs-One implementation."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, OneVsOneConfig
from .data import load_iris_data, train_validation_split


class OneVsOnePipeline:
    """One-vs-One multi-class strategy."""

    def __init__(self, config: OneVsOneConfig | None = None) -> None:
        self.config = config or CONFIG
        self.ovo_classifier = None

    def train(self) -> dict[str, dict[str, float]]:
        """Train OvO classifier and compare strategies."""
        X, y = load_iris_data(self.config)
        X_train, X_test, y_train, y_test = train_validation_split(X, y, self.config)

        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # One-vs-One with SVM
        self.ovo_classifier = OneVsOneClassifier(
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        )
        self.ovo_classifier.fit(X_train_scaled, y_train)
        y_pred_ovo = self.ovo_classifier.predict(X_test_scaled)

        n_classes = len(np.unique(y_train))
        n_estimators = n_classes * (n_classes - 1) // 2

        metrics_ovo = {
            "accuracy": float(accuracy_score(y_test, y_pred_ovo)),
            "precision_weighted": float(precision_score(y_test, y_pred_ovo, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred_ovo, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_ovo, average="weighted", zero_division=0)),
            "n_estimators": n_estimators,
            "n_classes": n_classes,
        }

        # Save model
        joblib.dump({
            "ovo_classifier": self.ovo_classifier,
            "scaler": scaler,
        }, self.config.model_path)

        return {
            "one_vs_one": metrics_ovo,
        }
