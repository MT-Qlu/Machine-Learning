"""Quick demonstration of the AirPassengers Exponential Smoothing pipeline."""
from __future__ import annotations

import json

from src.pipeline import AirPassengersExponentialSmoothingPipeline


def main() -> None:
    pipeline = AirPassengersExponentialSmoothingPipeline()
    metrics = pipeline.train()
    forecast = pipeline.forecast(12)

    print(
        json.dumps(
            {
                "metrics": metrics,
                "forecast": {
                    "index": [dt.strftime("%Y-%m-%d") for dt in forecast.index],
                    "values": [float(value) for value in forecast.values],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
