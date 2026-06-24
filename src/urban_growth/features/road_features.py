from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from urban_growth.geo.grids import boundary_input_path

WGS84_CRS = "EPSG:4326"

ROAD_CLASSES = {
    "major": ["motorway", "trunk", "primary"],
    "medium": ["secondary", "tertiary"],
    "minor": ["unclassified", "residential", "living_street"],
    "service": ["service"],
}

INCLUDED_HIGHWAY_TYPES = sorted(
    {road_type for road_types in ROAD_CLASSES.values() for road_type in road_types}
)

ROAD_TYPE_TO_CLASS = {
    road_type: road_class
    for road_class, road_types in ROAD_CLASSES.items()
    for road_type in road_types
}


def read_feature_dataset(path: str | Path) -> gpd.GeoDataFrame:
    """Read a feature dataset from GeoParquet."""
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Feature dataset file not found: {dataset_path}")

    dataset = gpd.read_parquet(dataset_path)

    if dataset.empty:
        raise ValueError(f"Feature dataset is empty: {dataset_path}")

    if dataset.crs is None:
        dataset = dataset.set_crs(WGS84_CRS)

    return dataset.to_crs(WGS84_CRS)


def road_features_output_path(
    country_code: str,
    grid_size_m: int,
    output_dir: str | Path = "data/features/roads",
    dataset_label: str | None = None,
) -> Path:
    """Build the standard output path for the road features dataset."""
    label = f"_{dataset_label}" if dataset_label else ""

    return (
        Path(output_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_road_features{label}_{grid_size_m}m.parquet"
    )


def roads_output_path(
    city_id: str,
    country_code: str,
    road_dir: str | Path = "data/external/roads",
) -> Path:
    """Build the standard cached road GeoJSON path for a city."""
    return Path(road_dir) / country_code.lower() / f"{city_id}_roads.geojson"


def read_city_boundary(
    city_id: str,
    country_code: str,
    boundary_dir: str | Path = "data/external/boundaries",
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


def normalize_highway_type(value: Any) -> str | None:
    """Normalize an OSM highway value into a single string."""
    if value is None:
        return None

    if isinstance(value, list | tuple | set | np.ndarray):
        values = [str(item) for item in value if item is not None]
        return values[0] if values else None

    return str(value)


def classify_highway_type(highway_type: str | None) -> str | None:
    """Classify an OSM highway type into a road class."""
    if highway_type is None:
        return None

    return ROAD_TYPE_TO_CLASS.get(highway_type)


def _features_from_polygon(polygon, tags: dict[str, list[str]]) -> gpd.GeoDataFrame:
    """Call the OSMnx features API with compatibility across versions."""
    if hasattr(ox, "features_from_polygon"):
        return ox.features_from_polygon(polygon, tags=tags)

    return ox.features.features_from_polygon(polygon, tags=tags)


def _filter_road_geometries(roads: gpd.GeoDataFrame, city_id: str) -> gpd.GeoDataFrame:
    """Keep supported road line geometries and attach normalized road metadata."""
    if roads.empty:
        return gpd.GeoDataFrame(
            columns=["road_id", "city_id", "highway_type", "road_class", "geometry"],
            geometry="geometry",
            crs=WGS84_CRS,
        )

    roads = roads.reset_index(drop=True).copy()

    if roads.crs is None:
        roads = roads.set_crs(WGS84_CRS)

    roads = roads.to_crs(WGS84_CRS)

    if "highway" not in roads.columns:
        return gpd.GeoDataFrame(
            columns=["road_id", "city_id", "highway_type", "road_class", "geometry"],
            geometry="geometry",
            crs=WGS84_CRS,
        )

    line_mask = roads.geometry.apply(lambda geom: isinstance(geom, LineString | MultiLineString))

    roads = roads.loc[line_mask].copy()
    roads["highway_type"] = roads["highway"].apply(normalize_highway_type)
    roads["road_class"] = roads["highway_type"].apply(classify_highway_type)
    roads = roads.loc[roads["road_class"].notna()].copy()

    roads["city_id"] = city_id
    roads = roads.reset_index(drop=True)
    roads["road_id"] = [f"{city_id}_road_{index:06d}" for index in range(1, len(roads) + 1)]

    return roads[
        [
            "road_id",
            "city_id",
            "highway_type",
            "road_class",
            "geometry",
        ]
    ]


def fetch_roads_from_osm(
    boundary: gpd.GeoDataFrame,
    city_id: str,
) -> gpd.GeoDataFrame:
    """Fetch road features from OpenStreetMap for a city boundary."""
    boundary = boundary.to_crs(WGS84_CRS)
    polygon = boundary.geometry.union_all()

    tags = {"highway": INCLUDED_HIGHWAY_TYPES}
    roads = _features_from_polygon(polygon, tags=tags)

    return _filter_road_geometries(roads, city_id=city_id)


def save_roads(roads: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Save cached city roads as GeoJSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    roads.to_file(path, driver="GeoJSON")

    return path


def read_or_fetch_city_roads(
    city_id: str,
    country_code: str,
    boundary: gpd.GeoDataFrame,
    road_dir: str | Path = "data/external/roads",
    refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Read cached city roads or fetch them from OpenStreetMap."""
    path = roads_output_path(
        city_id=city_id,
        country_code=country_code,
        road_dir=road_dir,
    )

    if path.exists() and not refresh:
        roads = gpd.read_file(path)

        if roads.crs is None:
            roads = roads.set_crs(WGS84_CRS)

        return roads.to_crs(WGS84_CRS)

    roads = fetch_roads_from_osm(boundary=boundary, city_id=city_id)
    save_roads(roads, path)

    return roads


def _get_metric_crs(gdf: gpd.GeoDataFrame):
    """Estimate a local metric CRS for distance and length calculations."""
    metric_crs = gdf.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError("Could not estimate a metric CRS.")

    return metric_crs


def _distance_to_road_subset(
    centroids: gpd.GeoSeries,
    roads: gpd.GeoDataFrame,
) -> np.ndarray:
    """Calculate distance from each centroid to a road subset."""
    if roads.empty:
        return np.full(len(centroids), np.nan)

    road_geometry = roads.geometry.union_all()

    return centroids.distance(road_geometry).to_numpy()


def _add_nearest_road_features(
    cells_metric: gpd.GeoDataFrame,
    roads_metric: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Add nearest road features using a nearest spatial join."""
    if roads_metric.empty:
        return pd.DataFrame(
            {
                "distance_to_nearest_road_m": np.full(len(cells_metric), np.nan),
                "nearest_road_type": [None] * len(cells_metric),
                "nearest_road_class": [None] * len(cells_metric),
            },
            index=cells_metric.index,
        )

    centroids = gpd.GeoDataFrame(
        {
            "_cell_pos": cells_metric["_cell_pos"].to_numpy(),
        },
        geometry=cells_metric.geometry.centroid,
        crs=cells_metric.crs,
    )

    roads_for_join = roads_metric[
        [
            "highway_type",
            "road_class",
            "geometry",
        ]
    ].copy()

    nearest = gpd.sjoin_nearest(
        centroids,
        roads_for_join,
        how="left",
        distance_col="distance_to_nearest_road_m",
    )

    nearest = nearest.sort_values("_cell_pos")
    nearest = nearest.drop_duplicates("_cell_pos", keep="first")
    nearest = nearest.set_index("_cell_pos")

    return pd.DataFrame(
        {
            "distance_to_nearest_road_m": nearest["distance_to_nearest_road_m"],
            "nearest_road_type": nearest["highway_type"],
            "nearest_road_class": nearest["road_class"],
        }
    )


def _calculate_road_density_features(
    cells_metric: gpd.GeoDataFrame,
    roads_metric: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Calculate road density features in meters per square kilometer."""
    density = pd.DataFrame(index=cells_metric["_cell_pos"])

    density["road_density_all_m_per_km2"] = 0.0

    for road_class in ROAD_CLASSES:
        density[f"road_density_{road_class}_m_per_km2"] = 0.0

    if roads_metric.empty:
        return density

    cells_for_overlay = cells_metric[
        [
            "_cell_pos",
            "cell_area_km2",
            "geometry",
        ]
    ].copy()

    roads_for_overlay = roads_metric[
        [
            "road_class",
            "geometry",
        ]
    ].copy()

    intersections = gpd.overlay(
        cells_for_overlay,
        roads_for_overlay,
        how="intersection",
        keep_geom_type=False,
    )

    line_mask = intersections.geometry.apply(
        lambda geom: isinstance(geom, LineString | MultiLineString)
    )
    intersections = intersections.loc[line_mask].copy()

    if intersections.empty:
        return density

    intersections["road_length_m"] = intersections.geometry.length

    all_lengths = intersections.groupby("_cell_pos")["road_length_m"].sum()
    density.loc[all_lengths.index, "road_density_all_m_per_km2"] = (
        all_lengths / cells_metric.set_index("_cell_pos").loc[all_lengths.index, "cell_area_km2"]
    )

    class_lengths = (
        intersections.groupby(["_cell_pos", "road_class"])["road_length_m"].sum().reset_index()
    )

    cell_area_lookup = cells_metric.set_index("_cell_pos")["cell_area_km2"]

    for _, row in class_lengths.iterrows():
        cell_pos = row["_cell_pos"]
        road_class = row["road_class"]
        length_m = row["road_length_m"]
        area_km2 = cell_area_lookup.loc[cell_pos]

        density.loc[cell_pos, f"road_density_{road_class}_m_per_km2"] = length_m / area_km2

    return density


def add_road_features_for_city(
    city_cells: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add road accessibility and density features for a single city."""
    if city_cells.empty:
        raise ValueError("city_cells cannot be empty.")

    metric_crs = _get_metric_crs(city_cells)

    cells_metric = city_cells.to_crs(metric_crs).copy()
    roads_metric = roads.to_crs(metric_crs).copy() if not roads.empty else roads.copy()

    cells_metric["_cell_pos"] = np.arange(len(cells_metric))

    if "cell_area_km2" not in cells_metric.columns:
        cells_metric["cell_area_km2"] = cells_metric.geometry.area / 1_000_000

    result = city_cells.copy().reset_index(drop=True)

    nearest_features = _add_nearest_road_features(
        cells_metric=cells_metric,
        roads_metric=roads_metric,
    )

    nearest_features = nearest_features.reindex(cells_metric["_cell_pos"]).reset_index(drop=True)

    result["distance_to_nearest_road_m"] = nearest_features["distance_to_nearest_road_m"].to_numpy()
    result["distance_to_nearest_road_km"] = result["distance_to_nearest_road_m"] / 1000
    result["nearest_road_type"] = nearest_features["nearest_road_type"].to_numpy()
    result["nearest_road_class"] = nearest_features["nearest_road_class"].to_numpy()

    centroids = cells_metric.geometry.centroid

    for road_class in ROAD_CLASSES:
        class_roads = (
            roads_metric.loc[roads_metric["road_class"] == road_class]
            if not roads_metric.empty
            else roads_metric
        )
        distance_m = _distance_to_road_subset(
            centroids=centroids,
            roads=class_roads,
        )

        result[f"distance_to_{road_class}_road_m"] = distance_m
        result[f"distance_to_{road_class}_road_km"] = distance_m / 1000

    density_features = _calculate_road_density_features(
        cells_metric=cells_metric,
        roads_metric=roads_metric,
    )

    density_features = density_features.reindex(cells_metric["_cell_pos"]).reset_index(drop=True)

    for column in density_features.columns:
        result[column] = density_features[column].to_numpy()

    result["road_count_city"] = int(len(roads))

    return result


def add_road_features(
    dataset: gpd.GeoDataFrame,
    country_code: str,
    boundary_dir: str | Path = "data/external/boundaries",
    road_dir: str | Path = "data/external/roads",
    refresh_roads: bool = False,
) -> gpd.GeoDataFrame:
    """Add road features to a full feature dataset."""
    if "city_id" not in dataset.columns:
        raise ValueError("Dataset must contain a 'city_id' column.")

    city_datasets = []

    for city_id, city_cells in dataset.groupby("city_id", sort=True):
        boundary = read_city_boundary(
            city_id=city_id,
            country_code=country_code,
            boundary_dir=boundary_dir,
        )

        roads = read_or_fetch_city_roads(
            city_id=city_id,
            country_code=country_code,
            boundary=boundary,
            road_dir=road_dir,
            refresh=refresh_roads,
        )

        city_features = add_road_features_for_city(
            city_cells=gpd.GeoDataFrame(city_cells, crs=dataset.crs),
            roads=roads,
        )

        city_datasets.append(city_features)

    if not city_datasets:
        raise ValueError("No city datasets were processed.")

    return gpd.GeoDataFrame(
        pd.concat(city_datasets, ignore_index=True),
        crs=WGS84_CRS,
    )


def save_road_features(dataset: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Save road features as GeoParquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(path, index=False)

    return path
