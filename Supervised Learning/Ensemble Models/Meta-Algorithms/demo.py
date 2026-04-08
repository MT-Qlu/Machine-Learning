"""Quick demonstration of meta-algorithms (Blending Ensemble)."""
from __future__ import annotations

import json

from src.pipeline import BlendingPipeline


def main() -> None:
    """Train and evaluate blending ensemble on Iris dataset."""
    pipeline = BlendingPipeline()
    metrics = pipeline.train()

    print(
        json.dumps(
            {
                "algorithm": "Blending Ensemble",
                "base_learners": [
                    "Random Forest",
                    "Gradient Boosting",
                    "Decision Tree",
                ],
                "meta_learner": "Logistic Regression",
                "holdout_ratio": 0.25,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
