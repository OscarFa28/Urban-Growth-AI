"""Build DENUE service accessibility features."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from urban_growth.features.denue_service_features import (
    DENUE_SERVICE_CATEGORIES,
    build_denue_service_accessibility_features,
    load_denue_service_points,
    summarize_denue_service_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DENUE service accessibility features.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--city-ids", nargs="+", default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DENUE_SERVICE_CATEGORIES),
        choices=list(DENUE_SERVICE_CATEGORIES),
    )
    parser.add_argument(
        "--raw-denue-dir",
        type=Path,
        default=Path("data/raw/inegi/denue"),
    )
    parser.add_argument("--spatial-features-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)

    return parser.parse_args()


def default_spatial_path(
    country_code: str,
    priorities: list[int],
    grid_size: int,
) -> Path:
    country = country_code.lower()
    priorities_suffix = "_".join(str(priority) for priority in priorities)

    return Path(
        "data/features/spatial"
        f"/{country}/{grid_size}m"
        f"/{country}_spatial_features_priorities_{priorities_suffix}_{grid_size}m.parquet"
    )


def default_output_path(
    country_code: str,
    priorities: list[int],
    grid_size: int,
    year: int,
) -> Path:
    country = country_code.lower()
    priorities_suffix = "_".join(str(priority) for priority in priorities)

    return Path(
        "data/features/services_denue"
        f"/{country}/{grid_size}m"
        f"/{country}_service_features_denue{year}"
        f"_priorities_{priorities_suffix}_{grid_size}m.parquet"
    )


def main() -> None:
    args = parse_args()

    spatial_path = args.spatial_features_path or default_spatial_path(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
    )
    output_path = args.output_path or default_output_path(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
        year=args.year,
    )

    print(f"Reading spatial features: {spatial_path}")
    cells = gpd.read_parquet(spatial_path)

    if args.city_ids:
        print(f"Filtering city ids: {args.city_ids}")
        cells = cells[cells["city_id"].isin(args.city_ids)].copy()

    print(f"Cells: {len(cells):,}")
    print(f"Cities: {cells['city_id'].nunique():,}")
    print(f"Categories: {args.categories}")

    print(f"Reading DENUE raw files from: {args.raw_denue_dir / str(args.year)}")
    service_points = load_denue_service_points(
        raw_denue_dir=args.raw_denue_dir,
        year=args.year,
        cells=cells,
        categories=args.categories,
    )

    print(f"DENUE points in selected bounds: {len(service_points):,}")

    features = build_denue_service_accessibility_features(
        cells=cells,
        service_points=service_points,
        categories=args.categories,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)

    summary = summarize_denue_service_features(features)

    print(f"Saved DENUE service features: {output_path}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Cells: {summary['cells']:,}")
    print(f"Total services assigned to cells: {summary['total_services_assigned']:,}")
    print(f"Cells with services: {summary['cells_with_services']:,}")


if __name__ == "__main__":
    main()
