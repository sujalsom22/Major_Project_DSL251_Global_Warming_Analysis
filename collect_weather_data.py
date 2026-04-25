from __future__ import annotations

import argparse

from src.weather_pipeline import (
    collect_all_cities,
    dataframe_schema,
    export_city_catalog,
    export_dataset,
    export_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical daily temperature data from Open-Meteo for multiple global cities."
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--output",
        default="global_temperature_dataset.csv",
        help="CSV filename for the merged dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = collect_all_cities(start_date=args.start_date, end_date=args.end_date)
    if df.empty:
        print("No data was returned from the API.")
        return

    dataset_path = export_dataset(df, filename=args.output)
    schema_path = export_schema()
    city_catalog_path = export_city_catalog()

    print(f"Rows collected: {len(df)}")
    print(f"Cities covered: {df['city'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Schema saved to: {schema_path}")
    print(f"City catalog saved to: {city_catalog_path}")
    print("\nSchema preview:")
    for column_name, description in dataframe_schema().items():
        print(f"- {column_name}: {description}")


if __name__ == "__main__":
    main()
