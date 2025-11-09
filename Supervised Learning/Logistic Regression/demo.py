"""Quick demonstration of the heart-disease logistic regression pipeline."""
from __future__ import annotations

import json

from src.pipeline import HeartDiseasePipeline


def main() -> None:
    pipeline = HeartDiseasePipeline()
    metrics = pipeline.train()

    sample = [[54, 1, 1, 130, 246, 0, 1, 150, 0, 1.0, 1, 0, 2]]
    proba_positive = float(pipeline.pipeline.predict_proba(sample)[0][1])  # type: ignore[arg-type]
    prediction = int(proba_positive >= 0.5)

    print(
        json.dumps(
            {
                "input": {
                    "age": sample[0][0],
                    "sex": sample[0][1],
                    "cp": sample[0][2],
                    "trestbps": sample[0][3],
                    "chol": sample[0][4],
                    "fbs": sample[0][5],
                    "restecg": sample[0][6],
                    "thalach": sample[0][7],
                    "exang": sample[0][8],
                    "oldpeak": sample[0][9],
                    "slope": sample[0][10],
                    "ca": sample[0][11],
                    "thal": sample[0][12],
                },
                "predicted_class": prediction,
                "probability": proba_positive,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
