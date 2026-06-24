import argparse
from pathlib import Path

from urban_growth.features.spatial_features import (
    add_basic_spatial_features,
    read_base_dataset,
    save_spatial_features,
    spatial_features_output_path,
)


def build_priority_label(priorities: list[int]) -> str:
    """Build a stable dataset label from priority levels."""
    unique_priorities = sorted(set(priorities))

    if len(unique_priorities) == 1:
        return f"priority_{unique_priorities[0]}"

    joined_priorities = "_".join(str(priority) for priority in unique_priorities)
    return f"priorities_{joined_priorities}"


def base_dataset_input_path(
    country_code: str,
    grid_size_m: int,
    priorities: list[int],
    base_dir: str | Path = "data/features/base",
) -> Path:
    """Build the expected input path for a base cells dataset."""
    dataset_label = build_priority_label(priorities)

    return (
        Path(base_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_base_cells_{dataset_label}_{grid_size_m}m.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build basic spatial features from a base cells dataset."
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
        help="Priority levels used in the base dataset.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=500,
        help="Grid size in meters.",
    )
    parser.add_argument(
        "--base-dir",
        default="data/features/base",
        help="Directory where base datasets are stored.",
    )
    parser.add_argument(
        "--base-dataset",
        default=None,
        help="Optional explicit path to the base dataset.",
    )
    parser.add_argument(
        "--boundary-dir",
        default="data/external/boundaries",
        help="Directory where city boundary files are stored.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features/spatial",
        help="Directory where spatial feature datasets will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_label = build_priority_label(args.priorities)

    input_path = (
        Path(args.base_dataset)
        if args.base_dataset
        else base_dataset_input_path(
            country_code=args.country_code,
            grid_size_m=args.grid_size,
            priorities=args.priorities,
            base_dir=Path(args.base_dir),
        )
    )

    print(f"Reading base dataset: {input_path}")

    base_dataset = read_base_dataset(input_path)

    print(f"Adding spatial features to {len(base_dataset):,} cells...")

    spatial_dataset = add_basic_spatial_features(
        dataset=base_dataset,
        country_code=args.country_code,
        boundary_dir=Path(args.boundary_dir),
    )

    output_path = spatial_features_output_path(
        country_code=args.country_code,
        grid_size_m=args.grid_size,
        output_dir=Path(args.output_dir),
        dataset_label=dataset_label,
    )

    save_spatial_features(spatial_dataset, output_path)

    print(f"OK   rows: {len(spatial_dataset):,}")
    print(f"OK   output: {output_path}")


if __name__ == "__main__":
    main()
