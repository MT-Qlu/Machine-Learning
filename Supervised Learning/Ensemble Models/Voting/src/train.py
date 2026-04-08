"""Command-line entry point for training the voting ensemble model."""
from __future__ import annotations

import json

from .pipeline import VotingEnsemblePipeline


def main() -> dict[str, dict[str, float]]:
    """Train the voting ensemble pipeline and return evaluation metrics for both hard and soft voting."""
    pipeline = VotingEnsemblePipeline()
    metrics = pipeline.train()
    return metrics


if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2))
