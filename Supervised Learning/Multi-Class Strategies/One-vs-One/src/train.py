"""Command-line entry point for training the One-vs-One model."""
from __future__ import annotations

import json

from .pipeline import OneVsOnePipeline


def main() -> dict[str, dict[str, float]]:
    """Train the One-vs-One pipeline and return evaluation metrics."""
    pipeline = OneVsOnePipeline()
    metrics = pipeline.train()
    return metrics


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
