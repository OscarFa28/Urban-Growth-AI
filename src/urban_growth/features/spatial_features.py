from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_growth.geo.grids import boundary_input_path

WGS84_CRS = "EPSG:4326"


def read_base_dataset(path: str | Path) -> gpd.GeoDataFrame:
    """Read a base cells dataset from GeoParquet."""
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Base dataset file not found: {dataset_path}")

    dataset = gpd.read_parquet(dataset_path)

    if dataset.empty:
        raise ValueError(f"Base dataset is empty: {dataset_path}")

    if dataset.crs is None:
        dataset = dataset.set_crs(WGS84_CRS)

    return dataset.to_crs(WGS84_CRS)


def spatial_features_output_path(
    country_code: str,
    grid_size_m: int,
    output_dir: str | Path = "data/features/spatial",
    dataset_label: str | None = None,
) -> Path:
    """Build the standard output path for the spatial features dataset."""
    label = f"_{dataset_label}" if dataset_label else ""

    return (
        Path(output_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_spatial_features{label}_{grid_size_m}m.parquet"
    )


def _read_city_boundary(
    city_id: str,
    country_code: str,
    boundary_dir: str | Path,
) -> gpd.GeoDataFrame:
    """Read a city boundary from the standard boundary path."""
    path = boundary_input_path(
        city_id=city_id,
        country_code=country_code,
        boundary_dir=boundary_dir,
    )

    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found for city '{city_id}': {path}")

    boundary = gpd.read_file(path)

    if boundary.empty:
        raise ValueError(f"Boundary file is empty for city '{city_id}': {path}")

    if boundary.crs is None:
        boundary = boundary.set_crs(WGS84_CRS)

    return boundary.to_crs(WGS84_CRS)


def _get_metric_crs(gdf: gpd.GeoDataFrame):
    """Estimate a local metric CRS for distance calculations."""
    metric_crs = gdf.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError("Could not estimate a metric CRS.")

    return metric_crs


def _safe_normalize(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to 0-1 safely."""
    max_value = series.max()

    if pd.isna(max_value) or max_value == 0:
        return pd.Series([0.0] * len(series), index=series.index)

    return series / max_value


def add_spatial_features_for_city(
    city_cells: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add basic spatial features for a single city."""
    if city_cells.empty:
        raise ValueError("city_cells cannot be empty.")

    if boundary.empty:
        raise ValueError("boundary cannot be empty.")

    metric_crs = _get_metric_crs(boundary)

    cells_metric = city_cells.to_crs(metric_crs)
    boundary_metric = boundary.to_crs(metric_crs)

    city_geometry = boundary_metric.geometry.union_all()
    city_boundary_line = city_geometry.boundary
    city_center_metric = city_geometry.centroid

    cell_centroids_metric = cells_metric.geometry.centroid
    cell_centroids_wgs84 = gpd.GeoSeries(cell_centroids_metric, crs=metric_crs).to_crs(WGS84_CRS)
    city_center_wgs84 = gpd.GeoSeries([city_center_metric], crs=metric_crs).to_crs(WGS84_CRS)

    result = city_cells.copy()

    result["cell_centroid_lon"] = cell_centroids_wgs84.x.to_numpy()
    result["cell_centroid_lat"] = cell_centroids_wgs84.y.to_numpy()

    result["city_center_lon"] = float(city_center_wgs84.x.iloc[0])
    result["city_center_lat"] = float(city_center_wgs84.y.iloc[0])

    result["distance_to_city_center_m"] = cell_centroids_metric.distance(
        city_center_metric
    ).to_numpy()
    result["distance_to_city_center_km"] = result["distance_to_city_center_m"] / 1000

    result["distance_to_boundary_m"] = cell_centroids_metric.distance(city_boundary_line).to_numpy()
    result["distance_to_boundary_km"] = result["distance_to_boundary_m"] / 1000

    result["cell_area_km2"] = cells_metric.geometry.area.to_numpy() / 1_000_000

    result["normalized_distance_to_city_center"] = _safe_normalize(
        result["distance_to_city_center_m"]
    )
    result["normalized_distance_to_boundary"] = _safe_normalize(result["distance_to_boundary_m"])

    if "coverage_ratio" in result.columns:
        result["is_boundary_cell"] = result["coverage_ratio"] < 0.999
    else:
        result["is_boundary_cell"] = False

    return result


def add_basic_spatial_features(
    dataset: gpd.GeoDataFrame,
    country_code: str,
    boundary_dir: str | Path = "data/external/boundaries",
) -> gpd.GeoDataFrame:
    """Add basic spatial features to a base cells dataset."""
    if "city_id" not in dataset.columns:
        raise ValueError("Dataset must contain a 'city_id' column.")

    city_datasets = []

    for city_id, city_cells in dataset.groupby("city_id", sort=True):
        boundary = _read_city_boundary(
            city_id=city_id,
            country_code=country_code,
            boundary_dir=boundary_dir,
        )

        city_features = add_spatial_features_for_city(
            city_cells=gpd.GeoDataFrame(city_cells, crs=dataset.crs),
            boundary=boundary,
        )

        city_datasets.append(city_features)

    if not city_datasets:
        raise ValueError("No city datasets were processed.")

    return gpd.GeoDataFrame(
        pd.concat(city_datasets, ignore_index=True),
        crs=WGS84_CRS,
    )


def save_spatial_features(dataset: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Save spatial features as GeoParquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(path, index=False)

    return path
