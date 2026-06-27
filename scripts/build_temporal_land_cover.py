import argparse
from pathlib import Path

from urban_growth.features.temporal_land_cover import (
    build_temporal_land_cover_features,
    build_years,
    initialize_earth_engine,
    read_input_dataset,
    road_features_input_path,
    save_temporal_land_cover_features,
    temporal_land_cover_checkpoint_dir,
    temporal_land_cover_output_path,
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
        description="Build temporal land-cover features using Dynamic World."
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
        help="First year to process.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to process.",
    )
    parser.add_argument(
        "--roads-dir",
        default="data/features/roads",
        help="Directory where road feature datasets are stored.",
    )
    parser.add_argument(
        "--input-dataset",
        default=None,
        help="Optional explicit input dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features/land_cover",
        help="Directory where temporal land-cover features will be saved.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/features/land_cover/checkpoints",
        help="Directory where city-year checkpoints will be saved.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute city-year files even if checkpoints already exist.",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="Optional city IDs to process.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Earth Engine reduction scale in meters.",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=4,
        help="Earth Engine tileScale parameter for large reductions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of cells processed per Earth Engine batch.",
    )
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Optional Google Earth Engine project ID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_label = build_priority_label(args.priorities)
    years = build_years(args.start_year, args.end_year)

    input_path = (
        Path(args.input_dataset)
        if args.input_dataset
        else road_features_input_path(
            country_code=args.country_code,
            grid_size_m=args.grid_size,
            dataset_label=dataset_label,
            roads_dir=Path(args.roads_dir),
        )
    )

    output_path = temporal_land_cover_output_path(
        country_code=args.country_code,
        grid_size_m=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=Path(args.output_dir),
        dataset_label=dataset_label,
    )

    run_checkpoint_dir = temporal_land_cover_checkpoint_dir(
        country_code=args.country_code,
        grid_size_m=args.grid_size,
        start_year=args.start_year,
        end_year=args.end_year,
        checkpoint_dir=Path(args.checkpoint_dir),
        dataset_label=dataset_label,
    )

    print(f"Reading input dataset: {input_path}")
    dataset = read_input_dataset(input_path)

    if args.cities:
        selected_rows = dataset.loc[dataset["city_id"].isin(args.cities)]
        selected_cell_count = len(selected_rows)
    else:
        selected_cell_count = len(dataset)

    print("Initializing Earth Engine...")
    initialize_earth_engine(project=args.ee_project)

    print(
        f"Building Dynamic World temporal land-cover features "
        f"for {selected_cell_count:,} selected cells."
    )
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Cities filter: {args.cities if args.cities else 'all'}")
    print(f"Batch size: {args.batch_size:,} cells")
    print(f"Resume enabled: {not args.no_resume}")
    print(f"Checkpoint dir: {run_checkpoint_dir}")

    temporal_features = build_temporal_land_cover_features(
        dataset=dataset,
        years=years,
        city_ids=args.cities,
        scale_m=args.scale,
        tile_scale=args.tile_scale,
        batch_size=args.batch_size,
        checkpoint_dir=run_checkpoint_dir,
        resume=not args.no_resume,
    )

    save_temporal_land_cover_features(temporal_features, output_path)

    print(f"OK   rows: {len(temporal_features):,}")
    print(f"OK   output: {output_path}")
    print(f"OK   checkpoints: {run_checkpoint_dir}")


if __name__ == "__main__":
    main()
