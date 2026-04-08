"""Demo of One-vs-One."""
from __future__ import annotations

import json

from src.pipeline import OneVsOnePipeline


def main() -> None:
    pipeline = OneVsOnePipeline()
    metrics = pipeline.train()

    print(
        json.dumps(
            {
                "strategy": "One-vs-One",
                "base_classifier": "SVM",
                "dataset": "Iris",
                "metrics": metrics,
                "note": "OvO trains one classifier per pair of classes. More balanced but more models.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
