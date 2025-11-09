"""Quick demonstration of the salary linear regression pipeline."""
from __future__ import annotations

import json

from src.pipeline import LinearRegressionPipeline


def main() -> None:
    pipeline = LinearRegressionPipeline()
    metrics = pipeline.train()

    sample_years = [[6.5]]  # years of experience
    predicted_salary = float(pipeline.pipeline.predict(sample_years)[0])  # type: ignore[arg-type]

    print(
        json.dumps(
            {
                "years_experience": sample_years[0][0],
                "predicted_salary": predicted_salary,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
