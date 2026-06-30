"""Build DENUE economic features."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from urban_growth.features.economic_features import build_denue_economic_features


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
    denue_year: int,
) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)
    return (
        Path("data/features/economic")
        / country
        / f"{grid_size}m"
        / f"{country}_economic_features_denue{denue_year}_{label}_{grid_size}m.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DENUE economic features.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--denue-year", type=int, default=2025)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--denue-dir",
        type=Path,
        default=Path("data/raw/inegi/denue/2025"),
    )
    parser.add_argument(
        "--state-codes",
        nargs="+",
        default=["01", "02", "09", "11", "14", "19", "22", "24", "32"],
    )
    parser.add_argument("--bbox-buffer-degrees", type=float, default=0.25)
    parser.add_argument("--chunksize", type=int, default=200_000)
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
        args.denue_year,
    )

    print(f"Reading cells: {input_path}")
    cells = gpd.read_parquet(input_path)

    print(f"Reading DENUE directory: {args.denue_dir}")
    print(f"State codes: {args.state_codes}")

    features = build_denue_economic_features(
        cells=cells,
        denue_dir=args.denue_dir,
        target_state_codes={str(code).zfill(2) for code in args.state_codes},
        bbox_buffer_degrees=args.bbox_buffer_degrees,
        chunksize=args.chunksize,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path)

    print(f"Saved economic features: {output_path}")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns):,}")
    print(
        "Total businesses assigned to cells: "
        f"{int(features['economic_business_count_total'].sum()):,}"
    )


if __name__ == "__main__":
    main()
