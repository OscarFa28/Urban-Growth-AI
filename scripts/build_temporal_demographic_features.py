"""Build temporal demographic features from INEGI 2010 and 2020 census data."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from urban_growth.features.temporal_demographic_features import (
    build_municipal_temporal_adjustments,
    build_temporal_demographic_features,
    load_municipal_census_2010,
    load_municipal_census_2020,
)


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_input_path(country_code: str, priorities: list[int], grid_size: int) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)
    return (
        Path("data/features/demographic")
        / country
        / f"{grid_size}m"
        / f"{country}_demographic_features_census2020_blocks_{label}_{grid_size}m.parquet"
    )


def default_output_path(
    country_code: str,
    priorities: list[int],
    grid_size: int,
    start_year: int,
    end_year: int,
) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)
    return (
        Path("data/features/demographic_temporal")
        / country
        / f"{grid_size}m"
        / (
            f"{country}_demographic_temporal_features_"
            f"{start_year}_{end_year}_{label}_{grid_size}m.parquet"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal demographic features.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--census-2010-dir",
        type=Path,
        default=Path("data/raw/inegi/Estados_2010"),
    )
    parser.add_argument(
        "--census-2020-dir",
        type=Path,
        default=Path("data/raw/inegi/Estados_2020"),
    )
    parser.add_argument(
        "--post-2020-method",
        choices=["hold", "extrapolate"],
        default="hold",
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
        args.start_year,
        args.end_year,
    )

    print(f"Reading cell demographics 2020: {input_path}")
    cells_2020 = gpd.read_parquet(input_path)

    print(f"Reading municipal census 2010: {args.census_2010_dir}")
    municipal_2010 = load_municipal_census_2010(args.census_2010_dir)

    print(f"Reading municipal census 2020: {args.census_2020_dir}")
    municipal_2020 = load_municipal_census_2020(args.census_2020_dir)

    print(f"Municipal rows 2010: {len(municipal_2010):,}")
    print(f"Municipal rows 2020: {len(municipal_2020):,}")

    adjustments = build_municipal_temporal_adjustments(
        municipal_2010=municipal_2010,
        municipal_2020=municipal_2020,
        start_year=args.start_year,
        end_year=args.end_year,
        post_2020_method=args.post_2020_method,
    )

    print(f"Municipal temporal adjustment rows: {len(adjustments):,}")

    features = build_temporal_demographic_features(
        cells_2020=cells_2020,
        municipal_adjustments=adjustments,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)

    print(f"Saved temporal demographic features: {output_path}")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns):,}")


if __name__ == "__main__":
    main()
