"""Build OSM service accessibility features."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from urban_growth.features.service_features import (
    SERVICE_CATEGORIES,
    build_osm_service_points,
    build_service_accessibility_features,
    summarize_service_features,
)


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_paths(
    country_code: str,
    priorities: list[int],
    grid_size: int,
) -> dict[str, Path]:
    country = country_code.lower()
    label = priority_label(priorities)

    return {
        "spatial": (
            Path("data/features/spatial")
            / country
            / f"{grid_size}m"
            / f"{country}_spatial_features_{label}_{grid_size}m.parquet"
        ),
        "cache": Path("data/external/services/osm") / country / f"{grid_size}m",
        "output": (
            Path("data/features/services")
            / country
            / f"{grid_size}m"
            / f"{country}_service_features_osm_{label}_{grid_size}m.parquet"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OSM service features.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--spatial-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(SERVICE_CATEGORIES),
        default=list(SERVICE_CATEGORIES),
    )
    parser.add_argument(
        "--city-ids",
        nargs="+",
        default=None,
        help="Optional city ids for partial runs.",
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use local OSM service cache only and do not fetch missing data from Overpass.",
    )
    parser.add_argument(
        "--overpass-url",
        default=None,
        help="Optional OSMnx Overpass base API URL, for example https://overpass.kumi.systems/api",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.overpass_url or args.overwrite_cache:
        import osmnx as ox

        if args.overpass_url:
            ox.settings.overpass_url = args.overpass_url
            print(f"Using Overpass URL: {args.overpass_url}")

        if args.overwrite_cache:
            ox.settings.use_cache = False
            print("Disabled OSMnx cache because --overwrite-cache was provided")

        ox.settings.log_console = True

    paths = default_paths(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
    )

    spatial_path = args.spatial_path or paths["spatial"]
    cache_dir = args.cache_dir or paths["cache"]
    output_path = args.output or paths["output"]

    print(f"Reading spatial features: {spatial_path}")
    cells = gpd.read_parquet(spatial_path)

    if args.city_ids:
        print(f"Filtering city ids: {args.city_ids}")
        cells = cells[cells["city_id"].isin(args.city_ids)].copy()

    print(f"Cells: {len(cells):,}")
    print(f"Cities: {cells['city_id'].nunique():,}")
    print(f"Categories: {args.categories}")
    print(f"Cache dir: {cache_dir}")

    service_points = build_osm_service_points(
        spatial_features=cells,
        cache_dir=cache_dir,
        categories=args.categories,
        city_ids=args.city_ids,
        overwrite_cache=args.overwrite_cache,
        cache_only=args.cache_only,
    )

    print(f"Fetched/cached service points: {len(service_points):,}")

    features = build_service_accessibility_features(
        cells=cells,
        service_points=service_points,
        categories=args.categories,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)

    summary = summarize_service_features(features)

    print(f"Saved service features: {output_path}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Cells: {summary['cells']:,}")
    print(f"Total services assigned to cells: {summary['total_services_assigned_to_cells']:,}")
    print(f"Cells with services: {summary['cells_with_services']:,}")


if __name__ == "__main__":
    main()
