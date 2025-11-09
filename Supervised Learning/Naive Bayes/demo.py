"""Quick demonstration of the mushroom Naive Bayes pipeline."""
from __future__ import annotations

import json

import pandas as pd

from src.pipeline import MushroomPipeline


def main() -> None:
    pipeline = MushroomPipeline()
    metrics = pipeline.train()

    sample = pd.DataFrame(
        [
            {
                "cap-shape": "x",
                "cap-surface": "s",
                "cap-color": "n",
                "bruises": "t",
                "odor": "a",
                "gill-attachment": "f",
                "gill-spacing": "c",
                "gill-size": "b",
                "gill-color": "k",
                "stalk-shape": "e",
                "stalk-root": "b",
                "stalk-surface-above-ring": "s",
                "stalk-surface-below-ring": "s",
                "stalk-color-above-ring": "w",
                "stalk-color-below-ring": "w",
                "veil-type": "p",
                "veil-color": "w",
                "ring-number": "o",
                "ring-type": "p",
                "spore-print-color": "k",
                "population": "s",
                "habitat": "u",
            }
        ]
    )

    proba_poisonous = float(pipeline.pipeline.predict_proba(sample)[0][1])  # type: ignore[arg-type]
    label = "poisonous" if proba_poisonous >= 0.5 else "edible"

    print(
        json.dumps(
            {
                "sample": sample.iloc[0].to_dict(),
                "predicted_label": label,
                "probability_poisonous": proba_poisonous,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
