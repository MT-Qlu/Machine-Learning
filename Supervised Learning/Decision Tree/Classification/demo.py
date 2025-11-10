"""Quick demonstration script for the Iris decision tree classifier."""
from __future__ import annotations

from src.inference import IrisDecisionTreeRequest, get_service


def main() -> None:
    service = get_service()
    sample = IrisDecisionTreeRequest(
        sepal_length_cm=6.0,
        sepal_width_cm=3.1,
        petal_length_cm=4.8,
        petal_width_cm=1.6,
    )
    response = service.predict(sample)
    print(response.model_dump())


if __name__ == "__main__":
    main()
