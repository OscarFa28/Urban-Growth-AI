import argparse
from pathlib import Path

from urban_growth.data.boundaries import fetch_and_save_city_boundary
from urban_growth.data.city_registry import get_priority_cities, load_city_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build city boundary GeoJSON files from a city registry."
    )
    parser.add_argument(
        "--registry",
        default="configs/cities/mexico.yaml",
        help="Path to the city registry YAML file.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=1,
        help="City priority level to process.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/external/boundaries",
        help="Directory where boundary files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = load_city_registry(args.registry)
    cities = get_priority_cities(registry, priority=args.priority)

    country = registry["country"]
    country_code = registry["country_code"]

    print(f"Building boundaries for {len(cities)} cities...")

    for city in cities:
        city_id = city["id"]
        city_name = city["name"]

        try:
            path = fetch_and_save_city_boundary(
                city=city,
                country=country,
                country_code=country_code,
                output_dir=Path(args.output_dir),
            )
            print(f"OK   {city_id} - {city_name}: {path}")
        except Exception as exc:
            print(f"FAIL {city_id} - {city_name}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
