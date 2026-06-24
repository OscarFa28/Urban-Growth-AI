import argparse
from pathlib import Path

from urban_growth.features.road_features import (
    add_road_features,
    read_feature_dataset,
    road_features_output_path,
    save_road_features,
)


def build_priority_label(priorities: list[int]) -> str:
    """Build a stable dataset label from priority levels."""
    unique_priorities = sorted(set(priorities))

    if len(unique_priorities) == 1:
        return f"priority_{unique_priorities[0]}"

    joined_priorities = "_".join(str(priority) for priority in unique_priorities)
    return f"priorities_{joined_priorities}"


def spatial_features_input_path(
    country_code: str,
    grid_size_m: int,
    priorities: list[int],
    spatial_dir: str | Path = "data/features/spatial",
) -> Path:
    """Build the expected input path for a spatial features dataset."""
    dataset_label = build_priority_label(priorities)

    return (
        Path(spatial_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_spatial_features_{dataset_label}_{grid_size_m}m.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build road accessibility features from spatial features."
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
        "--spatial-dir",
        default="data/features/spatial",
        help="Directory where spatial feature datasets are stored.",
    )
    parser.add_argument(
        "--spatial-dataset",
        default=None,
        help="Optional explicit path to the spatial features dataset.",
    )
    parser.add_argument(
        "--boundary-dir",
        default="data/external/boundaries",
        help="Directory where city boundary files are stored.",
    )
    parser.add_argument(
        "--road-dir",
        default="data/external/roads",
        help="Directory where cached road files are stored.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features/roads",
        help="Directory where road feature datasets will be saved.",
    )
    parser.add_argument(
        "--refresh-roads",
        action="store_true",
        help="Refresh road data from OpenStreetMap even if cached files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_label = build_priority_label(args.priorities)

    input_path = (
        Path(args.spatial_dataset)
        if args.spatial_dataset
        else spatial_features_input_path(
            country_code=args.country_code,
            grid_size_m=args.grid_size,
            priorities=args.priorities,
            spatial_dir=Path(args.spatial_dir),
        )
    )

    print(f"Reading spatial features dataset: {input_path}")

    spatial_dataset = read_feature_dataset(input_path)

    print(f"Adding road features to {len(spatial_dataset):,} cells...")
    print("This step may take time because it can download and process OSM roads.")

    road_dataset = add_road_features(
        dataset=spatial_dataset,
        country_code=args.country_code,
        boundary_dir=Path(args.boundary_dir),
        road_dir=Path(args.road_dir),
        refresh_roads=args.refresh_roads,
    )

    output_path = road_features_output_path(
        country_code=args.country_code,
        grid_size_m=args.grid_size,
        output_dir=Path(args.output_dir),
        dataset_label=dataset_label,
    )

    save_road_features(road_dataset, output_path)

    print(f"OK   rows: {len(road_dataset):,}")
    print(f"OK   output: {output_path}")


if __name__ == "__main__":
    main()
