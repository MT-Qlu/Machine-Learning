"""Demo of One-vs-Rest."""
from __future__ import annotations

import json

from src.pipeline import OneVsRestPipeline


def main() -> None:
    pipeline = OneVsRestPipeline()
    metrics = pipeline.train()

    print(
        json.dumps(
            {
                "strategy": "One-vs-Rest",
                "base_classifier": "SVM",
                "dataset": "Iris",
                "metrics": metrics,
                "note": "OvR trains one classifier per class. Compare metrics with native multi-class.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
