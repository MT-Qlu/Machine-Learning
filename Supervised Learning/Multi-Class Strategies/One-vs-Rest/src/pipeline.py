"""One-vs-Rest implementation."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import CONFIG, OneVsRestConfig
from .data import load_iris_data, train_validation_split


class OneVsRestPipeline:
    """One-vs-Rest multi-class strategy."""

    def __init__(self, config: OneVsRestConfig | None = None) -> None:
        self.config = config or CONFIG
        self.ovr_classifier = None
        self.native_classifier = None

    def train(self) -> dict[str, dict[str, float]]:
        """Train OvR classifier and compare with native multi-class."""
        X, y = load_iris_data(self.config)
        X_train, X_test, y_train, y_test = train_validation_split(X, y, self.config)

        # Scaler for preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # One-vs-Rest with SVM
        self.ovr_classifier = OneVsRestClassifier(
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        )
        self.ovr_classifier.fit(X_train_scaled, y_train)
        y_pred_ovr = self.ovr_classifier.predict(X_test_scaled)

        metrics_ovr = {
            "accuracy": float(accuracy_score(y_test, y_pred_ovr)),
            "precision_weighted": float(precision_score(y_test, y_pred_ovr, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred_ovr, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_ovr, average="weighted", zero_division=0)),
            "n_estimators": self.ovr_classifier.n_classes_,
        }

        # Native multi-class SVM for comparison
        self.native_classifier = SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr")
        self.native_classifier.fit(X_train_scaled, y_train)
        y_pred_native = self.native_classifier.predict(X_test_scaled)

        metrics_native = {
            "accuracy": float(accuracy_score(y_test, y_pred_native)),
            "precision_weighted": float(precision_score(y_test, y_pred_native, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred_native, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_native, average="weighted", zero_division=0)),
        }

        # Save model
        joblib.dump({
            "ovr_classifier": self.ovr_classifier,
            "scaler": scaler,
        }, self.config.model_path)

        return {
            "one_vs_rest": metrics_ovr,
            "native_multiclass_svm": metrics_native,
        }
