"""Command-line entry point for training the One-vs-Rest model."""
from __future__ import annotations

import json

from .pipeline import OneVsRestPipeline


def main() -> dict[str, dict[str, float]]:
    """Train the One-vs-Rest pipeline and return evaluation metrics."""
    pipeline = OneVsRestPipeline()
    metrics = pipeline.train()
    return metrics


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
