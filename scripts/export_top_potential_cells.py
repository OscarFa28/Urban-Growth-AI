"""Export top urban growth potential cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

from urban_growth.analysis.potential_exports import (
    add_lon_lat_for_csv,
    filter_potential_scores,
    select_export_columns,
    select_top_n_by_city_year,
    select_top_percentile,
    summarize_export,
    to_geojson_crs,
)


def priority_label(priorities: list[int]) -> str:
    if len(priorities) == 1:
        return f"priority_{priorities[0]}"
    return "priorities_" + "_".join(str(priority) for priority in priorities)


def default_input_path(
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


def export_frame(
    frame: gpd.GeoDataFrame,
    output_base: Path,
    formats: list[str],
) -> None:
    """Export a GeoDataFrame to requested formats."""
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if "parquet" in formats:
        frame.to_parquet(output_base.with_suffix(".parquet"))

    if "geojson" in formats:
        geojson_frame = to_geojson_crs(frame)
        geojson_frame.to_file(output_base.with_suffix(".geojson"), driver="GeoJSON")

    if "csv" in formats:
        csv_frame = add_lon_lat_for_csv(frame)
        csv_frame.to_csv(output_base.with_suffix(".csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top potential cells.")
    parser.add_argument("--country-code", default="MX")
    parser.add_argument("--priorities", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--grid-size", type=int, default=500)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/potential"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--city-ids", nargs="+", default=None)
    parser.add_argument(
        "--top-percentiles",
        nargs="+",
        type=float,
        default=[0.99, 0.95, 0.90],
    )
    parser.add_argument("--top-n-per-city", type=int, default=25)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["parquet", "geojson", "csv"],
        default=["parquet", "geojson", "csv"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input or default_input_path(
        country_code=args.country_code,
        priorities=args.priorities,
        grid_size=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    country = args.country_code.lower()
    label = priority_label(args.priorities)
    year_label = f"year_{args.year}" if args.year is not None else "all_years"

    print(f"Reading potential scores: {input_path}")
    frame = gpd.read_parquet(input_path)

    filtered = filter_potential_scores(
        frame,
        year=args.year,
        city_ids=args.city_ids,
    )
    filtered = select_export_columns(filtered)

    summaries: dict[str, object] = {
        "input": str(input_path),
        "year": args.year,
        "city_ids": args.city_ids,
        "exports": {},
    }

    print(f"Filtered rows: {len(filtered):,}")
    print(f"Filtered cells: {filtered['cell_id'].nunique():,}")

    for percentile in args.top_percentiles:
        top = select_top_percentile(filtered, percentile)
        percentile_label = int(round((1 - percentile) * 100))
        name = (
            f"{country}_potential_top_{percentile_label}pct_{year_label}_{label}_{args.grid_size}m"
        )
        output_base = args.output_dir / name

        export_frame(top, output_base, args.formats)
        summary = summarize_export(top)
        summaries["exports"][name] = summary

        print()
        print(f"Exported {name}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    top_city = select_top_n_by_city_year(filtered, args.top_n_per_city)
    name = (
        f"{country}_potential_top_{args.top_n_per_city}_per_city_"
        f"{year_label}_{label}_{args.grid_size}m"
    )
    output_base = args.output_dir / name

    export_frame(top_city, output_base, args.formats)
    summary = summarize_export(top_city)
    summaries["exports"][name] = summary

    print()
    print(f"Exported {name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_path = args.output_dir / (
        f"{country}_potential_exports_summary_{year_label}_{label}_{args.grid_size}m.json"
    )
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print()
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
