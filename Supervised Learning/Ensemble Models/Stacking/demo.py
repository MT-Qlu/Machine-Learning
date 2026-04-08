"""Quick demonstration of stacking ensemble."""
from __future__ import annotations

import json

from src.pipeline import StackingEnsemblePipeline


def main() -> None:
    pipeline = StackingEnsemblePipeline()
    metrics = pipeline.train()

    print(
        json.dumps(
            {
                "algorithm": "Stacking Ensemble",
                "base_learners": ["Logistic Regression", "Decision Tree", "KNN"],
                "meta_learner": "Logistic Regression",
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
