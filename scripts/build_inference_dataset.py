"""Build current-year inference dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from urban_growth.inference.dataset import (
    build_inference_dataset,
    read_inference_inputs,
    select_inference_output_columns,
    summarize_inference_dataset,
)


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_paths(
    country_code: str,
    priorities: list[int],
    grid_size: int,
    start_year: int,
    end_year: int,
    denue_year: int,
    inference_year: int,
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
        "roads": (
            Path("data/features/roads")
            / country
            / f"{grid_size}m"
            / f"{country}_road_features_{label}_{grid_size}m.parquet"
        ),
        "land_cover": (
            Path("data/features/land_cover")
            / country
            / f"{grid_size}m"
            / (
                f"{country}_land_cover_dynamic_world_{label}_"
                f"{start_year}_{end_year}_{grid_size}m.parquet"
            )
        ),
        "demographic": (
            Path("data/features/demographic_temporal")
            / country
            / f"{grid_size}m"
            / (
                f"{country}_demographic_temporal_features_"
                f"{start_year}_{end_year}_{label}_{grid_size}m.parquet"
            )
        ),
        "economic": (
            Path("data/features/economic")
            / country
            / f"{grid_size}m"
            / f"{country}_economic_features_denue{denue_year}_{label}_{grid_size}m.parquet"
        ),
        "output": (
            Path("data/features/inference")
            / country
            / f"{grid_size}m"
            / (f"{country}_inference_dataset_{inference_year}_{label}_{grid_size}m.parquet")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inference dataset.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--denue-year", type=int, default=2025)
    parser.add_argument("--inference-year", type=int, default=2025)
    parser.add_argument("--urban-threshold", type=float, default=0.35)

    parser.add_argument("--spatial-path", type=Path, default=None)
    parser.add_argument("--road-path", type=Path, default=None)
    parser.add_argument("--land-cover-path", type=Path, default=None)
    parser.add_argument("--demographic-path", type=Path, default=None)
    parser.add_argument("--economic-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Save only compact inference columns. By default, all features are saved for scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = default_paths(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
        denue_year=args.denue_year,
        inference_year=args.inference_year,
    )

    spatial_path = args.spatial_path or paths["spatial"]
    road_path = args.road_path or paths["roads"]
    land_cover_path = args.land_cover_path or paths["land_cover"]
    demographic_path = args.demographic_path or paths["demographic"]
    economic_path = args.economic_path or paths["economic"]
    output_path = args.output or paths["output"]

    print(f"Reading spatial features: {spatial_path}")
    print(f"Reading road features: {road_path}")
    print(f"Reading land cover features: {land_cover_path}")
    print(f"Reading demographic temporal features: {demographic_path}")
    print(f"Reading economic features: {economic_path}")

    inputs = read_inference_inputs(
        spatial_path=spatial_path,
        road_path=road_path,
        land_cover_path=land_cover_path,
        demographic_path=demographic_path,
        economic_path=economic_path,
    )

    inference = build_inference_dataset(
        **inputs,
        inference_year=args.inference_year,
        urban_threshold=args.urban_threshold,
    )

    if args.compact_output:
        inference = select_inference_output_columns(inference)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inference.to_parquet(output_path)

    summary = summarize_inference_dataset(inference)

    print(f"Saved inference dataset: {output_path}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Cells: {summary['cells']:,}")
    print(f"Years: {summary['years']}")
    print(f"Prediction years: {summary['prediction_years']}")
    print(f"Urban cells: {summary['urban_cells']:,}")
    print(f"Candidate cells: {summary['candidate_cells']:,}")


if __name__ == "__main__":
    main()
