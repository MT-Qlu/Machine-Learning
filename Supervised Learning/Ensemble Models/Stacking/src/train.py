"""Command-line entry point for training the stacking ensemble model."""
from __future__ import annotations

import json

from .pipeline import StackingEnsemblePipeline


def main() -> dict[str, float]:
    """Train the stacking ensemble pipeline and return evaluation metrics."""
    pipeline = StackingEnsemblePipeline()
    metrics = pipeline.train()
    return metrics


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
