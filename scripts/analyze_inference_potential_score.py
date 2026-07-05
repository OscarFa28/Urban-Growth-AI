"""Analyze current-year inference potential scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_input_path(
    country_code: str,
    priorities: list[int],
    grid_size: int,
    inference_year: int,
    model_name: str,
) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)

    return (
        Path("data/predictions/inference")
        / country
        / f"{grid_size}m"
        / (
            f"{country}_inference_potential_score_{inference_year}_"
            f"{model_name}_{label}_{grid_size}m.parquet"
        )
    )


def default_output_dir(
    country_code: str,
    priorities: list[int],
    grid_size: int,
    inference_year: int,
    model_name: str,
) -> Path:
    country = country_code.lower()
    label = priority_label(priorities)

    return (
        Path("reports/inference_analysis")
        / country
        / f"{grid_size}m"
        / f"{inference_year}_{model_name}_{label}"
    )


def summarize_overall(frame: pd.DataFrame, model_name: str) -> dict[str, Any]:
    tier_counts = frame["urban_growth_potential_tier"].value_counts().to_dict()

    return {
        "model_name": model_name,
        "rows": int(len(frame)),
        "cells": int(frame["cell_id"].nunique()),
        "years": sorted(int(year) for year in frame["year"].unique()),
        "cities": int(frame["city_id"].nunique()) if "city_id" in frame.columns else None,
        "mean_score": float(frame["urban_growth_potential_score"].mean()),
        "median_score": float(frame["urban_growth_potential_score"].median()),
        "max_score": float(frame["urban_growth_potential_score"].max()),
        "mean_probability": float(frame["urban_growth_probability"].mean()),
        "max_probability": float(frame["urban_growth_probability"].max()),
        "tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
    }


def summarize_by_city(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["city_id", "city_name"], dropna=False)
        .agg(
            cells=("cell_id", "count"),
            mean_score=("urban_growth_potential_score", "mean"),
            median_score=("urban_growth_potential_score", "median"),
            max_score=("urban_growth_potential_score", "max"),
            mean_probability=("urban_growth_probability", "mean"),
            very_high_cells=(
                "urban_growth_potential_tier",
                lambda values: int((values == "very_high").sum()),
            ),
            high_cells=(
                "urban_growth_potential_tier",
                lambda values: int((values == "high").sum()),
            ),
            medium_cells=(
                "urban_growth_potential_tier",
                lambda values: int((values == "medium").sum()),
            ),
        )
        .reset_index()
        .sort_values(
            ["very_high_cells", "high_cells", "mean_score"],
            ascending=[False, False, False],
        )
    )


def summarize_city_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["city_id", "city_name", "urban_growth_potential_tier"],
            dropna=False,
        )
        .agg(
            cells=("cell_id", "count"),
            mean_score=("urban_growth_potential_score", "mean"),
            max_score=("urban_growth_potential_score", "max"),
            mean_probability=("urban_growth_probability", "mean"),
        )
        .reset_index()
        .sort_values(["city_id", "urban_growth_potential_tier"])
    )


def select_top_cells(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    preferred_columns = [
        "cell_id",
        "year",
        "city_id",
        "city_name",
        "state",
        "municipality_name",
        "urban_growth_potential_score",
        "urban_growth_probability",
        "urban_growth_potential_tier",
        "urban_growth_potential_percentile",
        "urban_growth_potential_rank_overall_year",
        "urban_growth_potential_rank_city_year",
        "distance_to_city_center_m",
        "distance_to_nearest_road_m",
        "population_total",
        "population_density_per_km2",
        "economic_business_count_total",
        "economic_business_density_per_km2",
        "denue_service_total_count",
        "denue_service_total_density_per_km2",
        "denue_service_distance_to_nearest_any_m",
    ]

    columns = [column for column in preferred_columns if column in frame.columns]

    return (
        frame.sort_values("urban_growth_potential_score", ascending=False)
        .head(top_n)[columns]
        .copy()
    )


def compare_scores(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cell_id",
        "urban_growth_potential_score",
        "urban_growth_probability",
        "urban_growth_potential_tier",
    ]

    current_small = current[columns].rename(
        columns={
            "urban_growth_potential_score": "current_score",
            "urban_growth_probability": "current_probability",
            "urban_growth_potential_tier": "current_tier",
        }
    )
    previous_small = previous[columns].rename(
        columns={
            "urban_growth_potential_score": "previous_score",
            "urban_growth_probability": "previous_probability",
            "urban_growth_potential_tier": "previous_tier",
        }
    )

    compared = current_small.merge(previous_small, on="cell_id", how="inner")
    compared["score_delta"] = compared["current_score"] - compared["previous_score"]
    compared["probability_delta"] = (
        compared["current_probability"] - compared["previous_probability"]
    )
    compared["tier_changed"] = compared["current_tier"] != compared["previous_tier"]

    return compared.sort_values("score_delta", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze inference potential scores.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--inference-year", type=int, default=2025)
    parser.add_argument("--model-name", default="hist_gradient_boosting")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--compare-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input or default_input_path(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
        inference_year=args.inference_year,
        model_name=args.model_name,
    )
    output_dir = args.output_dir or default_output_dir(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
        inference_year=args.inference_year,
        model_name=args.model_name,
    )

    print(f"Reading inference potential scores: {input_path}")
    frame = pd.read_parquet(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_overall(frame, model_name=args.model_name)
    by_city = summarize_by_city(frame)
    city_tiers = summarize_city_tiers(frame)
    top_cells = select_top_cells(frame, top_n=args.top_n)

    summary_path = output_dir / "summary.json"
    by_city_path = output_dir / "city_summary.csv"
    city_tiers_path = output_dir / "city_tier_summary.csv"
    top_cells_path = output_dir / f"top_{args.top_n}_cells.csv"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    by_city.to_csv(by_city_path, index=False)
    city_tiers.to_csv(city_tiers_path, index=False)
    top_cells.to_csv(top_cells_path, index=False)

    print()
    print("Overall summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print("Top cities by very high cells:")
    print(by_city.head(15).to_string(index=False))

    print()
    print(f"Saved summary: {summary_path}")
    print(f"Saved city summary: {by_city_path}")
    print(f"Saved city tier summary: {city_tiers_path}")
    print(f"Saved top cells: {top_cells_path}")

    if args.compare_input is not None:
        print()
        print(f"Reading comparison scores: {args.compare_input}")
        previous = pd.read_parquet(args.compare_input)
        comparison = compare_scores(current=frame, previous=previous)

        comparison_path = output_dir / "score_comparison.csv"
        comparison_summary_path = output_dir / "score_comparison_summary.json"

        comparison_summary = {
            "rows_compared": int(len(comparison)),
            "mean_score_delta": float(comparison["score_delta"].mean()),
            "median_score_delta": float(comparison["score_delta"].median()),
            "max_score_delta": float(comparison["score_delta"].max()),
            "min_score_delta": float(comparison["score_delta"].min()),
            "mean_probability_delta": float(comparison["probability_delta"].mean()),
            "tier_changed_cells": int(comparison["tier_changed"].sum()),
        }

        comparison.to_csv(comparison_path, index=False)
        comparison_summary_path.write_text(
            json.dumps(comparison_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print()
        print("Comparison summary:")
        print(json.dumps(comparison_summary, indent=2, ensure_ascii=False))
        print(f"Saved score comparison: {comparison_path}")
        print(f"Saved score comparison summary: {comparison_summary_path}")


if __name__ == "__main__":
    main()
