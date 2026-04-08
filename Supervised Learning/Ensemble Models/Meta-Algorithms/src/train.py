"""Command-line entry point for training meta-algorithms models."""
from __future__ import annotations

import json

from .pipeline import BlendingPipeline


def main() -> dict[str, float]:
    """Train the blending meta-algorithm and return evaluation metrics."""
    pipeline = BlendingPipeline()
    metrics = pipeline.train()
    return metrics


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
