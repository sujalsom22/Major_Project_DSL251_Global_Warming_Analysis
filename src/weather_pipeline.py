from __future__ import annotations

from datetime import date
from pathlib import Path
import time

import pandas as pd

from src.cities import CITIES, CityRecord
from src.open_meteo_client import OpenMeteoRequest, fetch_daily_temperature_data


PROCESSED_OUTPUT_DIR = Path("data/processed/weather")


def ensure_output_directories() -> None:
    PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def split_date_range_by_year(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    ranges: list[tuple[str, str]] = []

    for year in range(start.year, end.year + 1):
        chunk_start = date(year, 1, 1)
        chunk_end = date(year, 12, 31)
        if year == start.year:
            chunk_start = start
        if year == end.year:
            chunk_end = end
        ranges.append((chunk_start.isoformat(), chunk_end.isoformat()))

    return ranges


def build_city_dataframe(city: CityRecord, start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in split_date_range_by_year(start_date, end_date):
        request = OpenMeteoRequest(
            latitude=city["latitude"],
            longitude=city["longitude"],
            start_date=chunk_start,
            end_date=chunk_end,
        )
        payload = fetch_daily_temperature_data(request)

        daily = payload.get("daily", {})
        chunk_df = pd.DataFrame(daily)
        if not chunk_df.empty:
            frames.append(chunk_df)

        time.sleep(1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["city"] = city["city"]
    df["country"] = city["country"]
    df["continent"] = city["continent"]
    df["latitude"] = city["latitude"]
    df["longitude"] = city["longitude"]
    df["data_source"] = "Open-Meteo Archive API"
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["season"] = df["month"].apply(season_from_month)
    df["month_name"] = df["date"].dt.month_name()

    for column in ("temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["city_mean_temperature"] = df["temperature_2m_mean"].mean()
    df["temperature_anomaly"] = df["temperature_2m_mean"] - df["city_mean_temperature"]
    df["rolling_7d_avg"] = df["temperature_2m_mean"].rolling(window=7, min_periods=1).mean()
    df["rolling_30d_avg"] = df["temperature_2m_mean"].rolling(window=30, min_periods=1).mean()

    return df[
        [
            "date",
            "year",
            "month",
            "month_name",
            "day",
            "season",
            "city",
            "country",
            "continent",
            "latitude",
            "longitude",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "city_mean_temperature",
            "temperature_anomaly",
            "rolling_7d_avg",
            "rolling_30d_avg",
            "data_source",
        ]
    ]


def collect_all_cities(start_date: str, end_date: str, cities: list[CityRecord] | None = None) -> pd.DataFrame:
    ensure_output_directories()
    selected_cities = cities or CITIES
    frames: list[pd.DataFrame] = []

    for city in selected_cities:
        print(f"Collecting {city['city']}...")
        city_df = build_city_dataframe(city, start_date=start_date, end_date=end_date)
        if city_df.empty:
            continue
        frames.append(city_df)
        time.sleep(2)

    if not frames:
        return pd.DataFrame()

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["continent_yearly_mean"] = combined_df.groupby(["continent", "year"])[
        "temperature_2m_mean"
    ].transform("mean")
    combined_df["city_yearly_mean"] = combined_df.groupby(["city", "year"])[
        "temperature_2m_mean"
    ].transform("mean")
    combined_df["global_daily_mean"] = combined_df.groupby("date")["temperature_2m_mean"].transform("mean")

    return combined_df.sort_values(["date", "continent", "city"]).reset_index(drop=True)


def dataframe_schema() -> dict[str, str]:
    return {
        "date": "Observation date",
        "year": "Calendar year extracted from date",
        "month": "Calendar month number",
        "month_name": "Calendar month name",
        "day": "Day of month",
        "season": "Meteorological season derived from month",
        "city": "City name",
        "country": "Country name",
        "continent": "Continent label",
        "latitude": "City latitude",
        "longitude": "City longitude",
        "temperature_2m_mean": "Daily mean temperature in Celsius",
        "temperature_2m_max": "Daily maximum temperature in Celsius",
        "temperature_2m_min": "Daily minimum temperature in Celsius",
        "city_mean_temperature": "Mean temperature across the selected date range for that city",
        "temperature_anomaly": "Daily mean minus city mean temperature",
        "rolling_7d_avg": "7-day rolling average of mean temperature",
        "rolling_30d_avg": "30-day rolling average of mean temperature",
        "data_source": "Source API name",
        "continent_yearly_mean": "Average mean temperature for the continent in that year",
        "city_yearly_mean": "Average mean temperature for the city in that year",
        "global_daily_mean": "Average mean temperature across all collected cities for that date",
    }


def export_dataset(df: pd.DataFrame, filename: str = "global_temperature_dataset.csv") -> Path:
    ensure_output_directories()
    output_path = PROCESSED_OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def export_schema(filename: str = "global_temperature_schema.csv") -> Path:
    ensure_output_directories()
    schema_df = pd.DataFrame(
        [{"column_name": name, "description": description} for name, description in dataframe_schema().items()]
    )
    output_path = PROCESSED_OUTPUT_DIR / filename
    schema_df.to_csv(output_path, index=False)
    return output_path


def export_city_catalog(filename: str = "city_catalog.csv") -> Path:
    ensure_output_directories()
    output_path = PROCESSED_OUTPUT_DIR / filename
    pd.DataFrame(CITIES).to_csv(output_path, index=False)
    return output_path
