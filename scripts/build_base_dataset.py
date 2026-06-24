import argparse
from pathlib import Path

from urban_growth.data.city_registry import get_priority_cities, load_city_registry
from urban_growth.data.dataset_builder import (
    base_dataset_output_path,
    build_base_dataset,
    save_base_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the base geospatial dataset from generated city grids."
    )
    parser.add_argument(
        "--registry",
        default="configs/cities/mexico.yaml",
        help="Path to the city registry YAML file.",
    )
    parser.add_argument(
        "--priorities",
        type=int,
        nargs="+",
        default=[1],
        help="City priority levels to process.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=500,
        help="Grid size in meters.",
    )
    parser.add_argument(
        "--grid-dir",
        default="data/interim/grids",
        help="Directory where generated city grids are stored.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features/base",
        help="Directory where the base dataset will be saved.",
    )
    return parser.parse_args()


def build_priority_label(priorities: list[int]) -> str:
    """Build a stable dataset label from priority levels."""
    unique_priorities = sorted(set(priorities))

    if len(unique_priorities) == 1:
        return f"priority_{unique_priorities[0]}"

    joined_priorities = "_".join(str(priority) for priority in unique_priorities)
    return f"priorities_{joined_priorities}"


def main() -> None:
    args = parse_args()

    registry = load_city_registry(args.registry)
    country_code = registry["country_code"]

    cities = []
    for priority in sorted(set(args.priorities)):
        cities.extend(get_priority_cities(registry, priority=priority))

    if not cities:
        raise ValueError(f"No cities found for priorities: {args.priorities}")

    dataset_label = build_priority_label(args.priorities)

    print(f"Building base dataset for {len(cities)} cities at {args.grid_size}m resolution...")
    print(f"Priorities: {sorted(set(args.priorities))}")

    dataset = build_base_dataset(
        registry=registry,
        cities=cities,
        grid_size_m=args.grid_size,
        grid_dir=Path(args.grid_dir),
    )

    output_path = base_dataset_output_path(
        country_code=country_code,
        grid_size_m=args.grid_size,
        output_dir=Path(args.output_dir),
        dataset_label=dataset_label,
    )

    save_base_dataset(dataset, output_path)

    print(f"OK   rows: {len(dataset):,}")
    print(f"OK   output: {output_path}")


if __name__ == "__main__":
    main()
