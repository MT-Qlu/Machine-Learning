"""Core implementation of Blending meta-algorithm."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CONFIG, MetaAlgorithmsConfig
from .data import load_iris_data


class BlendingPipeline:
    """Blending ensemble: trains base learners and meta-learner on holdout set.

    Blending differs from Stacking:
    - Uses fixed holdout set (not k-fold cross-validation)
    - Simpler and faster
    - Less data for meta-learner training
    - Lower risk of overfitting on meta-features
    """

    def __init__(self, config: MetaAlgorithmsConfig | None = None):
        """Initialize blending pipeline with configuration."""
        self.config = config or CONFIG
        self.base_learners: dict[str, Any] = {}
        self.meta_learner: Any = None
        self.scaler: StandardScaler | None = None
        self.metrics: dict[str, float] = {}

    def train(self) -> dict[str, float]:
        """Train blending ensemble:
        1. Split data into train/validation/test
        2. Train base learners on train set
        3. Generate meta-features on validation set
        4. Train meta-learner on meta-features
        5. Evaluate on test set
        """
        # Load data
        X, X_test, y, y_test, scaler = load_iris_data(
            test_size=0.2, random_state=self.config.random_state
        )
        self.scaler = scaler

        # Split train into train/validation for blending
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.config.blending_holdout_ratio,
            random_state=self.config.blending_random_state,
            stratify=y,
        )

        # Step 1: Train base learners on training set
        print("Training base learners...")
        base_learners = {
            "rf": RandomForestClassifier(
                n_estimators=self.config.n_estimators_rf,
                random_state=self.config.random_state,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=self.config.n_estimators_gb,
                random_state=self.config.random_state,
            ),
            "dt": DecisionTreeClassifier(
                max_depth=self.config.max_depth_dt,
                random_state=self.config.random_state,
            ),
        }

        for name, learner in base_learners.items():
            learner.fit(X_train, y_train)
            self.base_learners[name] = learner

        # Step 2: Generate meta-features on validation set
        print("Generating meta-features...")
        meta_features_val = np.column_stack(
            [learner.predict_proba(X_val) for learner in base_learners.values()]
        )
        meta_features_test = np.column_stack(
            [learner.predict_proba(X_test) for learner in base_learners.values()]
        )

        # Step 3: Train meta-learner
        print("Training meta-learner...")
        if self.config.meta_learner_type == "logistic":
            self.meta_learner = LogisticRegression(
                max_iter=self.config.meta_learner_max_iter,
                random_state=self.config.random_state,
            )
        else:
            self.meta_learner = LogisticRegression(
                max_iter=self.config.meta_learner_max_iter,
                random_state=self.config.random_state,
            )

        self.meta_learner.fit(meta_features_val, y_val)

        # Step 4: Evaluate on test set
        print("Evaluating ensemble...")
        y_pred = self.meta_learner.predict(meta_features_test)
        accuracy = np.mean(y_pred == y_test)

        # Individual base learner accuracies
        base_accuracies = {}
        for name, learner in self.base_learners.items():
            base_pred = learner.predict(X_test)
            base_accuracies[f"base_{name}"] = float(np.mean(base_pred == y_test))

        self.metrics = {
            "blending_accuracy": float(accuracy),
            **base_accuracies,
        }

        # Save artifacts
        self._save_artifacts()

        print(f"Blending Ensemble Accuracy: {accuracy:.4f}")
        print(f"Base learner accuracies: {base_accuracies}")

        return self.metrics

    def _save_artifacts(self) -> None:
        """Save trained models and metadata to artifacts."""
        artifacts_dir = Path(self.config.model_path).parent
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save base learners
        joblib.dump(self.base_learners, artifacts_dir / "base_learners.pkl")

        # Save meta-learner
        joblib.dump(self.meta_learner, artifacts_dir / "meta_learner.pkl")

        # Save scaler
        joblib.dump(self.scaler, artifacts_dir / "scaler.pkl")

        # Save metrics
        with open(artifacts_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Artifacts saved to {artifacts_dir}")

    def load_artifacts(self) -> None:
        """Load trained models from artifacts."""
        artifacts_dir = Path(self.config.model_path).parent

        self.base_learners = joblib.load(artifacts_dir / "base_learners.pkl")
        self.meta_learner = joblib.load(artifacts_dir / "meta_learner.pkl")
        self.scaler = joblib.load(artifacts_dir / "scaler.pkl")

        with open(artifacts_dir / "metrics.json") as f:
            self.metrics = json.load(f)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions using blending ensemble.

        Args:
            X: Input features (unscaled)

        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if self.meta_learner is None:
            self.load_artifacts()

        X_scaled = self.scaler.transform(X)

        # Generate meta-features from base learners
        meta_features = np.column_stack(
            [learner.predict_proba(X_scaled) for learner in self.base_learners.values()]
        )

        # Get predictions from meta-learner
        predictions = self.meta_learner.predict(meta_features)
        probabilities = self.meta_learner.predict_proba(meta_features)

        return predictions, probabilities
