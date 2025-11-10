"""Quick demonstration script for the California housing decision tree regressor."""
from __future__ import annotations

from src.inference import CaliforniaHousingRequest, get_service


def main() -> None:
    service = get_service()
    sample = CaliforniaHousingRequest(
        median_income=5.8,
        house_age=29.0,
        average_rooms=5.3,
        average_bedrooms=1.1,
        population=780.0,
        average_occupancy=3.2,
        latitude=34.2,
        longitude=-118.1,
    )
    response = service.predict(sample)
    print(response.model_dump())


if __name__ == "__main__":
    main()
