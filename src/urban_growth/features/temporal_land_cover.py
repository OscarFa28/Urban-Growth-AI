from pathlib import Path

import ee
import geemap
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

WGS84_CRS = "EPSG:4326"
DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

DYNAMIC_WORLD_CLASS_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DYNAMIC_WORLD_LABELS = {
    "water": 0,
    "trees": 1,
    "grass": 2,
    "flooded_vegetation": 3,
    "crops": 4,
    "shrub_and_scrub": 5,
    "built": 6,
    "bare": 7,
    "snow_and_ice": 8,
}


def initialize_earth_engine(project: str | None = None) -> None:
    """Initialize Google Earth Engine."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize Earth Engine. Run `earthengine authenticate` "
            "or pass a valid project with `--ee-project`."
        ) from exc


def build_years(start_year: int, end_year: int) -> list[int]:
    """Build an inclusive list of years."""
    if start_year > end_year:
        raise ValueError("start_year must be lower than or equal to end_year.")

    return list(range(start_year, end_year + 1))


def read_input_dataset(path: str | Path) -> gpd.GeoDataFrame:
    """Read the input geospatial feature dataset."""
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Input dataset file not found: {dataset_path}")

    dataset = gpd.read_parquet(dataset_path)

    if dataset.empty:
        raise ValueError(f"Input dataset is empty: {dataset_path}")

    if dataset.crs is None:
        dataset = dataset.set_crs(WGS84_CRS)

    return dataset.to_crs(WGS84_CRS)


def temporal_land_cover_output_path(
    country_code: str,
    grid_size_m: int,
    start_year: int,
    end_year: int,
    output_dir: str | Path = "data/features/land_cover",
    dataset_label: str | None = None,
    source: str = "dynamic_world",
) -> Path:
    """Build the standard output path for temporal land-cover features."""
    label = f"_{dataset_label}" if dataset_label else ""

    return (
        Path(output_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_land_cover_{source}{label}_"
        f"{start_year}_{end_year}_{grid_size_m}m.parquet"
    )


def temporal_land_cover_checkpoint_dir(
    country_code: str,
    grid_size_m: int,
    start_year: int,
    end_year: int,
    checkpoint_dir: str | Path = "data/features/land_cover/checkpoints",
    dataset_label: str | None = None,
    source: str = "dynamic_world",
) -> Path:
    """Build the checkpoint directory for a temporal land-cover run."""
    label = f"_{dataset_label}" if dataset_label else ""

    return (
        Path(checkpoint_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{source}{label}_{start_year}_{end_year}"
    )


def temporal_land_cover_checkpoint_path(
    checkpoint_dir: str | Path,
    city_id: str,
    year: int,
) -> Path:
    """Build the checkpoint path for one city-year."""
    return Path(checkpoint_dir) / city_id / f"{city_id}_{year}.parquet"


def road_features_input_path(
    country_code: str,
    grid_size_m: int,
    dataset_label: str,
    roads_dir: str | Path = "data/features/roads",
) -> Path:
    """Build the expected input path for a road features dataset."""
    return (
        Path(roads_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_road_features_{dataset_label}_{grid_size_m}m.parquet"
    )


def save_temporal_land_cover_features(
    dataset: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save temporal land-cover features as Parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(path, index=False)

    return path


def combine_temporal_land_cover_checkpoints(
    checkpoint_dir: str | Path,
) -> pd.DataFrame:
    """Combine all city-year checkpoint files into one DataFrame."""
    paths = sorted(Path(checkpoint_dir).glob("*/*.parquet"))

    if not paths:
        raise ValueError(f"No checkpoint files found in: {checkpoint_dir}")

    frames = [pd.read_parquet(path) for path in paths]

    dataset = pd.concat(frames, ignore_index=True)

    sort_columns = [
        column for column in ["city_id", "year", "cell_id"] if column in dataset.columns
    ]

    if sort_columns:
        dataset = dataset.sort_values(sort_columns).reset_index(drop=True)

    return dataset


def _build_dynamic_world_annual_image(
    region,
    year: int,
) -> ee.Image:
    """Build an annual Dynamic World image with mean probabilities."""
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"

    collection = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )

    probability_band_names = [f"{band}_probability_mean" for band in DYNAMIC_WORLD_CLASS_BANDS]

    probabilities = (
        collection.select(DYNAMIC_WORLD_CLASS_BANDS).mean().rename(probability_band_names)
    )

    built_label_frequency = (
        collection.select("label")
        .map(lambda image: image.eq(DYNAMIC_WORLD_LABELS["built"]).rename("built_label_frequency"))
        .mean()
    )

    observation_count = collection.select("built").count().rename("dw_observation_count")

    return probabilities.addBands(built_label_frequency).addBands(observation_count)


def _city_cells_to_ee_feature_collection(city_cells: gpd.GeoDataFrame):
    """Convert city cells to an Earth Engine FeatureCollection."""
    required_columns = ["cell_id", "city_id", "geometry"]
    missing_columns = [column for column in required_columns if column not in city_cells.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    cells = city_cells[required_columns].copy()
    cells = cells.to_crs(WGS84_CRS)

    return geemap.geopandas_to_ee(cells)


def _postprocess_ee_result(result: gpd.GeoDataFrame, year: int) -> pd.DataFrame:
    """Clean an Earth Engine result into a tabular feature DataFrame."""
    if result.empty:
        raise ValueError(f"Earth Engine returned an empty result for year {year}.")

    frame = pd.DataFrame(result.drop(columns="geometry", errors="ignore"))
    frame["year"] = year

    numeric_columns = [
        *[f"{band}_probability_mean" for band in DYNAMIC_WORLD_CLASS_BANDS],
        "built_label_frequency",
        "dw_observation_count",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "built_probability_mean" in frame.columns:
        frame["built_area_percentage_approx"] = frame["built_probability_mean"] * 100

    if "built_label_frequency" in frame.columns:
        frame["built_label_percentage"] = frame["built_label_frequency"] * 100

    ordered_columns = [
        "cell_id",
        "city_id",
        "year",
        *[f"{band}_probability_mean" for band in DYNAMIC_WORLD_CLASS_BANDS],
        "built_area_percentage_approx",
        "built_label_frequency",
        "built_label_percentage",
        "dw_observation_count",
    ]

    existing_columns = [column for column in ordered_columns if column in frame.columns]

    return frame[existing_columns]


def _iter_cell_batches(
    city_cells: gpd.GeoDataFrame,
    batch_size: int,
) -> list[gpd.GeoDataFrame]:
    """Split city cells into smaller batches for Earth Engine reductions."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    return [
        city_cells.iloc[start : start + batch_size].copy()
        for start in range(0, len(city_cells), batch_size)
    ]


def build_temporal_land_cover_features_for_city_year(
    city_cells: gpd.GeoDataFrame,
    year: int,
    scale_m: int = 10,
    tile_scale: int = 4,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Build Dynamic World features for one city-year."""
    if city_cells.empty:
        raise ValueError("city_cells cannot be empty.")

    city_cells = city_cells.to_crs(WGS84_CRS)
    city_id = str(city_cells["city_id"].iloc[0])
    batches = _iter_cell_batches(city_cells, batch_size=batch_size)

    batch_results = []

    for batch_index, batch_cells in enumerate(
        tqdm(
            batches,
            desc=f"{city_id} {year} batches",
            leave=False,
        ),
        start=1,
    ):
        tqdm.write(
            f"Processing {city_id} {year} "
            f"batch {batch_index}/{len(batches)} "
            f"({len(batch_cells):,} cells)"
        )

        city_fc = _city_cells_to_ee_feature_collection(batch_cells)
        region = city_fc.geometry()

        annual_image = _build_dynamic_world_annual_image(
            region=region,
            year=year,
        )

        reduced = annual_image.reduceRegions(
            collection=city_fc,
            reducer=ee.Reducer.mean(),
            scale=scale_m,
            tileScale=tile_scale,
        )

        result_gdf = geemap.ee_to_gdf(reduced)
        batch_results.append(_postprocess_ee_result(result_gdf, year=year))

    return pd.concat(batch_results, ignore_index=True)


def build_temporal_land_cover_features_for_city(
    city_cells: gpd.GeoDataFrame,
    years: list[int],
    scale_m: int = 10,
    tile_scale: int = 4,
    batch_size: int = 500,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Build annual Dynamic World features for one city."""
    if city_cells.empty:
        raise ValueError("city_cells cannot be empty.")

    city_cells = city_cells.to_crs(WGS84_CRS)
    city_id = str(city_cells["city_id"].iloc[0])

    yearly_results = []

    for year in tqdm(years, desc=f"{city_id} years", leave=False):
        checkpoint_path = (
            temporal_land_cover_checkpoint_path(
                checkpoint_dir=checkpoint_dir,
                city_id=city_id,
                year=year,
            )
            if checkpoint_dir
            else None
        )

        if checkpoint_path and checkpoint_path.exists() and resume:
            tqdm.write(f"Skipping {city_id} {year}; checkpoint already exists.")
            yearly_results.append(pd.read_parquet(checkpoint_path))
            continue

        try:
            year_result = build_temporal_land_cover_features_for_city_year(
                city_cells=city_cells,
                year=year,
                scale_m=scale_m,
                tile_scale=tile_scale,
                batch_size=batch_size,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed processing {city_id} {year}.") from exc

        if checkpoint_path:
            save_temporal_land_cover_features(year_result, checkpoint_path)
            tqdm.write(f"Saved checkpoint: {checkpoint_path}")

        yearly_results.append(year_result)

    return pd.concat(yearly_results, ignore_index=True)


def build_temporal_land_cover_features(
    dataset: gpd.GeoDataFrame,
    years: list[int],
    city_ids: list[str] | None = None,
    scale_m: int = 10,
    tile_scale: int = 4,
    batch_size: int = 500,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Build annual Dynamic World features for all selected cities."""
    if "city_id" not in dataset.columns:
        raise ValueError("Dataset must contain a 'city_id' column.")

    working_dataset = dataset.copy()

    if city_ids:
        working_dataset = working_dataset.loc[working_dataset["city_id"].isin(city_ids)].copy()

    if working_dataset.empty:
        raise ValueError("No rows left after filtering by city_ids.")

    city_results = []
    grouped = list(working_dataset.groupby("city_id", sort=True))

    for city_id, city_cells in tqdm(grouped, desc="Cities"):
        tqdm.write(f"Processing {city_id}: {len(city_cells):,} cells")

        city_result = build_temporal_land_cover_features_for_city(
            city_cells=gpd.GeoDataFrame(city_cells, crs=dataset.crs),
            years=years,
            scale_m=scale_m,
            tile_scale=tile_scale,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
        )

        city_results.append(city_result)

    return pd.concat(city_results, ignore_index=True)
