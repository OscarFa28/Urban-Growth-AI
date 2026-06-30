"""Build urban growth potential scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from joblib import load

from urban_growth.scoring.potential import (
    score_modeling_dataset,
    select_potential_output_columns,
    summarize_potential_scores,
)


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_modeling_dataset_path(
    country_code: str,
    priorities: list[int],
    grid_size: int,
    start_year: int,
    end_year: int,
) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)

    return (
        Path("data/features/modeling")
        / country
        / f"{grid_size}m"
        / f"{country}_modeling_dataset_{start_year}_{end_year}_{label}_{grid_size}m.parquet"
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
        Path("data/predictions/potential")
        / country
        / f"{grid_size}m"
        / (
            f"{country}_urban_growth_potential_score_"
            f"{start_year}_{end_year}_{label}_{grid_size}m.parquet"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build urban growth potential scores.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/baselines/baseline_logistic_regression.joblib"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Save all modeling columns instead of compact score columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input or default_modeling_dataset_path(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    output_path = args.output or default_output_path(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print(f"Reading modeling dataset: {input_path}")
    frame = gpd.read_parquet(input_path)

    print(f"Reading model bundle: {args.model_path}")
    model_bundle = load(args.model_path)

    print("Scoring urban growth potential...")
    scored = score_modeling_dataset(frame, model_bundle)

    if not args.full_output:
        scored = select_potential_output_columns(scored)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path)

    summary = summarize_potential_scores(scored)

    print(f"Saved potential scores: {output_path}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Cells: {summary['cells']:,}")
    print(f"Years: {summary['years']}")
    print(f"Mean score: {summary['mean_score']:.4f}")
    print(f"Max score: {summary['max_score']:.4f}")
    print(f"Very high count: {summary['very_high_count']:,}")
    print(f"High count: {summary['high_count']:,}")
    print(f"Medium count: {summary['medium_count']:,}")

    print()
    print("Tier counts:")
    print(scored["urban_growth_potential_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
