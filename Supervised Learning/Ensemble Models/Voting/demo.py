"""Demo of voting ensemble."""
from __future__ import annotations

import json

from src.pipeline import VotingEnsemblePipeline


def main() -> None:
    pipeline = VotingEnsemblePipeline()
    metrics = pipeline.train()

    print(
        json.dumps(
            {
                "algorithm": "Voting Ensemble",
                "base_learners": ["Logistic Regression", "Decision Tree", "KNN"],
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
