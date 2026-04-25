from __future__ import annotations

from typing import TypedDict


class CityRecord(TypedDict):
    city: str
    country: str
    continent: str
    latitude: float
    longitude: float


CITIES: list[CityRecord] = [
    {"city": "Delhi", "country": "India", "continent": "Asia", "latitude": 28.6139, "longitude": 77.2090},
    {"city": "Tokyo", "country": "Japan", "continent": "Asia", "latitude": 35.6762, "longitude": 139.6503},
    {"city": "Singapore", "country": "Singapore", "continent": "Asia", "latitude": 1.3521, "longitude": 103.8198},
    {"city": "London", "country": "United Kingdom", "continent": "Europe", "latitude": 51.5072, "longitude": -0.1276},
    {"city": "Paris", "country": "France", "continent": "Europe", "latitude": 48.8566, "longitude": 2.3522},
    {"city": "Berlin", "country": "Germany", "continent": "Europe", "latitude": 52.5200, "longitude": 13.4050},
    {"city": "New York", "country": "USA", "continent": "North America", "latitude": 40.7128, "longitude": -74.0060},
    {"city": "Los Angeles", "country": "USA", "continent": "North America", "latitude": 34.0522, "longitude": -118.2437},
    {"city": "Toronto", "country": "Canada", "continent": "North America", "latitude": 43.6532, "longitude": -79.3832},
    {"city": "Sao Paulo", "country": "Brazil", "continent": "South America", "latitude": -23.5505, "longitude": -46.6333},
    {"city": "Buenos Aires", "country": "Argentina", "continent": "South America", "latitude": -34.6037, "longitude": -58.3816},
    {"city": "Cairo", "country": "Egypt", "continent": "Africa", "latitude": 30.0444, "longitude": 31.2357},
    {"city": "Nairobi", "country": "Kenya", "continent": "Africa", "latitude": -1.2921, "longitude": 36.8219},
    {"city": "Cape Town", "country": "South Africa", "continent": "Africa", "latitude": -33.9249, "longitude": 18.4241},
    {"city": "Sydney", "country": "Australia", "continent": "Australia", "latitude": -33.8688, "longitude": 151.2093},
]
