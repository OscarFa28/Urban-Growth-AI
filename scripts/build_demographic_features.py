"""Build INEGI demographic features for grid cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from urban_growth.features.demographic_features import (
    add_demographic_features,
    load_inegi_ageb_census_2020,
)


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_input_path(country_code: str, priorities: list[int], grid_size: int) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)
    return (
        Path("data/features/roads")
        / country
        / f"{grid_size}m"
        / f"{country}_road_features_{label}_{grid_size}m.parquet"
    )


def default_output_path(country_code: str, priorities: list[int], grid_size: int) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)
    return (
        Path("data/features/demographic")
        / country
        / f"{grid_size}m"
        / f"{country}_demographic_features_census2020_{label}_{grid_size}m.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build INEGI census 2020 demographic features.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input cells parquet. Defaults to road features output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--resageburb-dir",
        type=Path,
        default=Path("data/raw/inegi/Estados_2020"),
        help="Directory containing RESAGEBURB_*CSV20.csv files.",
    )
    parser.add_argument(
        "--ageb-geometry-dir",
        type=Path,
        default=Path("data/raw/inegi/marco_geo_2020"),
        help="Directory containing Marco Geoestadistico 2020 shapefiles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input or default_input_path(
        args.country_code,
        args.priorities,
        args.grid_size,
    )
    output_path = args.output or default_output_path(
        args.country_code,
        args.priorities,
        args.grid_size,
    )

    print(f"Reading cells: {input_path}")
    cells = gpd.read_parquet(input_path)

    print(f"Reading INEGI RESAGEBURB CSVs: {args.resageburb_dir}")
    print(f"Reading INEGI AGEB geometries: {args.ageb_geometry_dir}")
    ageb_data = load_inegi_ageb_census_2020(
        resageburb_dir=args.resageburb_dir,
        ageb_geometry_dir=args.ageb_geometry_dir,
    )

    print(f"Cells: {len(cells):,}")
    print(f"AGEB joined rows: {len(ageb_data):,}")

    features = add_demographic_features(cells, ageb_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)

    print(f"Saved demographic features: {output_path}")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns):,}")


if __name__ == "__main__":
    main()
