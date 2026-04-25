from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = (
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
)


@dataclass(frozen=True)
class OpenMeteoRequest:
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    timezone: str = "auto"

    def to_params(self) -> dict[str, str | float]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": self.timezone,
        }


def fetch_daily_temperature_data(request: OpenMeteoRequest, max_retries: int = 5) -> dict[str, Any]:
    backoff_seconds = 3

    for attempt in range(max_retries):
        response = requests.get(ARCHIVE_URL, params=request.to_params(), timeout=60)

        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        if attempt == max_retries - 1:
            response.raise_for_status()

        time.sleep(backoff_seconds)
        backoff_seconds *= 2

    raise RuntimeError("Open-Meteo request failed after retries.")
