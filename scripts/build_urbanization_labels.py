import argparse
from pathlib import Path

from urban_growth.labels.urbanization import (
    DEFAULT_STRONG_GROWTH_THRESHOLD,
    DEFAULT_URBAN_THRESHOLD,
    add_urbanization_labels,
    land_cover_input_path,
    read_temporal_land_cover,
    save_urbanization_labels,
    urbanization_labels_output_path,
)


def build_priority_label(priorities: list[int]) -> str:
    """Build a stable dataset label from priority levels."""
    unique_priorities = sorted(set(priorities))

    if len(unique_priorities) == 1:
        return f"priority_{unique_priorities[0]}"

    joined_priorities = "_".join(str(priority) for priority in unique_priorities)
    return f"priorities_{joined_priorities}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build next-year urbanization labels from land-cover features."
    )
    parser.add_argument(
        "--country-code",
        default="MX",
        help="Country code used in generated paths.",
    )
    parser.add_argument(
        "--priorities",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Priority levels used in the input dataset.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=500,
        help="Grid size in meters.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2016,
        help="First source year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last source year.",
    )
    parser.add_argument(
        "--land-cover-dir",
        default="data/features/land_cover",
        help="Directory where land-cover feature datasets are stored.",
    )
    parser.add_argument(
        "--input-dataset",
        default=None,
        help="Optional explicit land-cover dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/labels/urbanization",
        help="Directory where urbanization labels will be saved.",
    )
    parser.add_argument(
        "--urban-threshold",
        type=float,
        default=DEFAULT_URBAN_THRESHOLD,
        help="Built probability threshold used to define an urban cell.",
    )
    parser.add_argument(
        "--strong-growth-threshold",
        type=float,
        default=DEFAULT_STRONG_GROWTH_THRESHOLD,
        help="Built probability increase threshold used to define strong growth.",
    )
    parser.add_argument(
        "--keep-incomplete-targets",
        action="store_true",
        help="Keep rows without a next-year target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_label = build_priority_label(args.priorities)

    input_path = (
        Path(args.input_dataset)
        if args.input_dataset
        else land_cover_input_path(
            country_code=args.country_code,
            grid_size_m=args.grid_size,
            start_year=args.start_year,
            end_year=args.end_year,
            dataset_label=dataset_label,
            land_cover_dir=Path(args.land_cover_dir),
        )
    )

    output_path = urbanization_labels_output_path(
        country_code=args.country_code,
        grid_size_m=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=Path(args.output_dir),
        dataset_label=dataset_label,
    )

    print(f"Reading temporal land-cover dataset: {input_path}")
    land_cover = read_temporal_land_cover(input_path)

    print(f"Building labels for {len(land_cover):,} cell-year rows...")
    print(f"Urban threshold: {args.urban_threshold}")
    print(f"Strong growth threshold: {args.strong_growth_threshold}")

    labels = add_urbanization_labels(
        dataset=land_cover,
        urban_threshold=args.urban_threshold,
        strong_growth_threshold=args.strong_growth_threshold,
        drop_incomplete_targets=not args.keep_incomplete_targets,
    )

    save_urbanization_labels(labels, output_path)

    print(f"OK   rows: {len(labels):,}")
    print(f"OK   output: {output_path}")
    print()
    print("Label distribution:")
    print(labels["urbanized_next_year"].value_counts(dropna=False).sort_index())
    print()
    print("Transition distribution:")
    print(labels["urban_state_transition"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
