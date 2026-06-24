import argparse
from pathlib import Path

from urban_growth.data.city_registry import get_priority_cities, load_city_registry
from urban_growth.geo.grids import (
    DEFAULT_GRID_SIZES_M,
    boundary_input_path,
    generate_multi_resolution_grids,
    grid_output_path,
    read_city_boundary,
    save_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-resolution city grids from boundary GeoJSON files."
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
        "--grid-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_GRID_SIZES_M,
        help="Grid sizes in meters.",
    )
    parser.add_argument(
        "--boundary-dir",
        default="data/external/boundaries",
        help="Directory where city boundary files are stored.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/grids",
        help="Directory where generated grid files will be saved.",
    )
    parser.add_argument(
        "--skip-missing-boundaries",
        action="store_true",
        help="Skip cities with missing boundary files instead of failing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = load_city_registry(args.registry)
    cities = get_priority_cities(registry, priority=args.priority)

    country_code = registry["country_code"]

    print(f"Building grids for {len(cities)} cities...")
    print(f"Grid sizes: {args.grid_sizes}")

    for city in cities:
        city_id = city["id"]
        city_name = city["name"]

        input_path = boundary_input_path(
            city_id=city_id,
            country_code=country_code,
            boundary_dir=Path(args.boundary_dir),
        )

        if not input_path.exists():
            message = f"Missing boundary for {city_id}: {input_path}"

            if args.skip_missing_boundaries:
                print(f"SKIP {city_id} - {city_name}: {message}")
                continue

            raise FileNotFoundError(message)

        try:
            boundary = read_city_boundary(input_path)
            grids = generate_multi_resolution_grids(
                boundary=boundary,
                grid_sizes_m=args.grid_sizes,
                city_id=city_id,
            )

            for grid_size_m, grid in grids.items():
                output_path = grid_output_path(
                    city_id=city_id,
                    country_code=country_code,
                    grid_size_m=grid_size_m,
                    output_dir=Path(args.output_dir),
                )
                save_grid(grid, output_path)

                print(
                    f"OK   {city_id} - {city_name} - {grid_size_m}m: "
                    f"{len(grid):,} cells -> {output_path}"
                )

        except Exception as exc:
            print(f"FAIL {city_id} - {city_name}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
