"""Quick demonstration of the AirPassengers Prophet pipeline."""
from __future__ import annotations

import json

from src.pipeline import AirPassengersProphetPipeline


def main() -> None:
    pipeline = AirPassengersProphetPipeline()
    metrics = pipeline.train()
    forecast = pipeline.forecast(12)

    print(
        json.dumps(
            {
                "metrics": metrics,
                "forecast": {
                    "index": [row["ds"].strftime("%Y-%m-%d") for _, row in forecast.iterrows()],
                    "yhat": [float(value) for value in forecast["yhat"].values],
                    "yhat_lower": [float(value) for value in forecast["yhat_lower"].values],
                    "yhat_upper": [float(value) for value in forecast["yhat_upper"].values],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
