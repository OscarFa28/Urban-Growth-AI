import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from urban_growth.data.city_registry import load_city_registry
from urban_growth.data.metropolitan_registry import (
    DEFAULT_METROPOLITAN_REGISTRY_PATH,
    read_metropolitan_municipalities,
    summarize_metropolitan_coverage,
    validate_metropolitan_municipalities,
)

DEFAULT_CITY_REGISTRY_PATH = Path("configs/cities/mexico.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate metropolitan municipality coverage registry."
    )
    parser.add_argument(
        "--metropolitan-registry",
        type=Path,
        default=DEFAULT_METROPOLITAN_REGISTRY_PATH,
        help="Path to the metropolitan municipalities CSV registry.",
    )
    parser.add_argument(
        "--city-registry",
        type=Path,
        default=DEFAULT_CITY_REGISTRY_PATH,
        help="Path to the current city registry YAML file.",
    )
    parser.add_argument(
        "--country-code",
        default="MX",
        help="Country code filter. Use an empty value to include all countries.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional path where a JSON summary will be written.",
    )

    return parser.parse_args()


def _filter_country(frame: pd.DataFrame, country_code: str | None) -> pd.DataFrame:
    if not country_code:
        return frame.reset_index(drop=True)

    normalized_code = country_code.strip().upper()

    if not normalized_code:
        return frame.reset_index(drop=True)

    return frame.loc[frame["country_code"].eq(normalized_code)].reset_index(drop=True)


def _load_city_ids(city_registry_path: Path) -> list[str]:
    registry = load_city_registry(city_registry_path)
    return sorted(str(city["id"]) for city in registry["cities"])


def _add_city_registry_comparison(
    summary: dict[str, Any],
    frame: pd.DataFrame,
    city_registry_path: Path,
) -> dict[str, Any]:
    city_ids = _load_city_ids(city_registry_path)
    covered_city_ids = sorted(frame["city_id"].drop_duplicates().tolist())
    covered_city_id_set = set(covered_city_ids)
    cities_without_coverage = [
        city_id for city_id in city_ids if city_id not in covered_city_id_set
    ]

    summary["city_registry_path"] = str(city_registry_path)
    summary["city_registry_city_count"] = len(city_ids)
    summary["covered_city_ids"] = covered_city_ids
    summary["cities_without_metropolitan_coverage"] = cities_without_coverage

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("Metropolitan coverage registry")
    print(f"Metro areas: {summary['metro_area_count']}")
    print(f"Unique municipalities: {summary['municipality_count']}")
    print(f"Covered city IDs: {summary['city_count']}")
    print("Municipalities by metro_area_id:")

    for metro_area_id, municipality_count in summary["municipalities_by_metro_area"].items():
        status = summary["coverage_statuses_by_metro_area"][metro_area_id]
        print(f"  - {metro_area_id}: {municipality_count} ({status})")

    manual_review_city_ids = summary["manual_review_required_city_ids"]
    print("Cities requiring manual review:")

    if manual_review_city_ids:
        for city_id in manual_review_city_ids:
            status = summary["coverage_statuses_by_city_id"][city_id]
            print(f"  - {city_id}: {status}")
    else:
        print("  - none")

    if "cities_without_metropolitan_coverage" in summary:
        missing_city_ids = summary["cities_without_metropolitan_coverage"]
        print("Cities without metropolitan coverage:")

        if missing_city_ids:
            for city_id in missing_city_ids:
                print(f"  - {city_id}")
        else:
            print("  - none")


def main() -> None:
    args = parse_args()

    frame = read_metropolitan_municipalities(args.metropolitan_registry)
    validate_metropolitan_municipalities(frame)
    frame = _filter_country(frame, args.country_code)

    summary = summarize_metropolitan_coverage(frame)
    summary["metropolitan_registry_path"] = str(args.metropolitan_registry)
    summary["country_code_filter"] = args.country_code.strip().upper()
    summary = _add_city_registry_comparison(
        summary=summary,
        frame=frame,
        city_registry_path=args.city_registry,
    )

    _print_summary(summary)

    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Summary JSON: {args.output_summary}")


if __name__ == "__main__":
    main()
