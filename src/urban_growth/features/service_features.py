"""OSM service accessibility features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

CELL_KEY = "cell_id"
CITY_KEY = "city_id"
DEFAULT_METRIC_CRS = "EPSG:6372"

SERVICE_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "education": {
        "amenity": ["school", "kindergarten", "college", "university"],
    },
    "health": {
        "amenity": ["hospital", "clinic", "doctors", "dentist", "pharmacy"],
    },
    "retail": {
        "shop": ["supermarket", "mall", "convenience", "department_store"],
        "amenity": ["marketplace"],
    },
    "transit": {
        "highway": ["bus_stop"],
        "public_transport": ["station", "platform"],
        "railway": ["station", "halt", "tram_stop"],
    },
    "recreation": {
        "leisure": ["park", "garden", "sports_centre", "playground"],
        "tourism": ["attraction"],
    },
}


def to_metric_crs(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project a GeoDataFrame to a metric CRS."""
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326", allow_override=True)

    try:
        return frame.to_crs(DEFAULT_METRIC_CRS)
    except Exception:
        estimated = frame.estimate_utm_crs()

        if estimated is not None:
            return frame.to_crs(estimated)

        return frame.to_crs("EPSG:3857")


def union_geometries(frame: gpd.GeoDataFrame):
    """Return unioned geometry with compatibility across shapely versions."""
    if hasattr(frame.geometry, "union_all"):
        return frame.geometry.union_all()

    return frame.geometry.unary_union


def empty_osm_features() -> gpd.GeoDataFrame:
    """Return an empty OSM features GeoDataFrame."""
    return gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )


def fetch_osm_features_from_polygon(
    polygon,
    tags: dict[str, list[str]],
) -> gpd.GeoDataFrame:
    """Fetch OSM features from a polygon using osmnx."""
    import osmnx as ox

    fetcher = getattr(ox, "features_from_polygon", None)

    if fetcher is None:
        fetcher = ox.geometries_from_polygon

    try:
        features = fetcher(polygon, tags)
    except Exception as exc:
        print(f"OSM polygon fetch failed, trying bbox fallback: {exc}")
        features = fetch_osm_features_from_bbox(polygon, tags)

    if features.empty:
        print("OSM polygon fetch returned 0 rows, trying bbox fallback")
        features = fetch_osm_features_from_bbox(polygon, tags)

    if features.empty:
        return empty_osm_features()

    features = features.reset_index(drop=False)

    if features.crs is None:
        features = features.set_crs("EPSG:4326", allow_override=True)

    return features


def fetch_osm_features_from_bbox(
    polygon,
    tags: dict[str, list[str]],
) -> gpd.GeoDataFrame:
    """Fetch OSM features from polygon bounding box and clip to polygon."""
    import osmnx as ox

    west, south, east, north = polygon.bounds
    bbox_fetcher = getattr(ox, "features_from_bbox", None)

    if bbox_fetcher is None:
        return empty_osm_features()

    try:
        try:
            # OSMnx v2 expects bbox as: west, south, east, north.
            features = bbox_fetcher((west, south, east, north), tags)
        except TypeError:
            # Older OSMnx versions may support explicit bbox kwargs.
            features = bbox_fetcher(
                west=west,
                south=south,
                east=east,
                north=north,
                tags=tags,
            )
    except Exception as exc:
        print(f"OSM bbox fallback failed: {exc}")
        return empty_osm_features()

    if features.empty:
        return empty_osm_features()

    features = features.reset_index(drop=False)

    if features.crs is None:
        features = features.set_crs("EPSG:4326", allow_override=True)

    try:
        polygon_frame = gpd.GeoDataFrame(
            geometry=[polygon],
            crs="EPSG:4326",
        )
        features = gpd.clip(features, polygon_frame)
    except Exception as exc:
        print(f"OSM bbox clipping failed, using unclipped bbox features: {exc}")

    return features


def convert_services_to_points(
    services: gpd.GeoDataFrame,
    category: str,
    city_id: str | None = None,
) -> gpd.GeoDataFrame:
    """Convert service geometries to representative points."""
    if services.empty:
        return gpd.GeoDataFrame(
            {
                "service_category": pd.Series(dtype="object"),
                "city_id": pd.Series(dtype="object"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    source_crs = services.crs or "EPSG:4326"
    services = services.set_crs(source_crs, allow_override=True)

    metric = to_metric_crs(services)
    points_geometry = metric.geometry.representative_point()

    output = metric.copy()
    output = output.set_geometry(points_geometry).to_crs("EPSG:4326")
    output["service_category"] = category

    if city_id is not None:
        output["city_id"] = city_id

    keep_columns = [
        column
        for column in [
            "city_id",
            "service_category",
            "name",
            "amenity",
            "shop",
            "leisure",
            "tourism",
            "highway",
            "public_transport",
            "railway",
            "geometry",
        ]
        if column in output.columns
    ]

    return gpd.GeoDataFrame(output[keep_columns], geometry="geometry", crs="EPSG:4326")


def build_osm_service_points(
    spatial_features: gpd.GeoDataFrame,
    cache_dir: Path,
    categories: list[str] | None = None,
    city_ids: list[str] | None = None,
    overwrite_cache: bool = False,
    cache_only: bool = False,
) -> gpd.GeoDataFrame:
    """Fetch/cache OSM service points for each city."""
    selected_categories = categories or list(SERVICE_CATEGORIES)

    if city_ids:
        spatial_features = spatial_features[spatial_features[CITY_KEY].isin(city_ids)].copy()

    cache_dir.mkdir(parents=True, exist_ok=True)

    all_points = []

    for city_id, city_cells in spatial_features.groupby(CITY_KEY):
        city_polygon = union_geometries(city_cells.to_crs("EPSG:4326"))

        for category in selected_categories:
            cache_path = cache_dir / f"{city_id}_{category}.parquet"

            if cache_path.exists() and not overwrite_cache:
                points = gpd.read_parquet(cache_path)
            elif cache_only:
                print(f"Missing cache, using empty services: {cache_path}")
                points = convert_services_to_points(
                    empty_osm_features(),
                    category=category,
                    city_id=str(city_id),
                )
            else:
                tags = SERVICE_CATEGORIES[category]
                raw_services = fetch_osm_features_from_polygon(city_polygon, tags)
                points = convert_services_to_points(
                    raw_services,
                    category=category,
                    city_id=str(city_id),
                )
                points.to_parquet(cache_path)

            all_points.append(points)

    if not all_points:
        return gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )

    combined = pd.concat(all_points, ignore_index=True)

    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def service_area_km2(cells_metric: gpd.GeoDataFrame) -> pd.Series:
    """Return cell area in square kilometers."""
    if "cell_area_km2" in cells_metric.columns:
        return pd.to_numeric(cells_metric["cell_area_km2"], errors="coerce").fillna(
            cells_metric.geometry.area / 1_000_000
        )

    if "area_m2" in cells_metric.columns:
        return (
            pd.to_numeric(cells_metric["area_m2"], errors="coerce").fillna(
                cells_metric.geometry.area
            )
            / 1_000_000
        )

    return cells_metric.geometry.area / 1_000_000


def count_points_within_cells(
    cells_metric: gpd.GeoDataFrame,
    points_metric: gpd.GeoDataFrame,
) -> pd.Series:
    """Count points within each cell."""
    if points_metric.empty:
        return pd.Series(0, index=cells_metric[CELL_KEY], dtype="int64")

    joined = gpd.sjoin(
        points_metric[[CELL_KEY if CELL_KEY in points_metric.columns else "geometry", "geometry"]]
        if CELL_KEY in points_metric.columns
        else points_metric[["geometry"]],
        cells_metric[[CELL_KEY, "geometry"]],
        how="left",
        predicate="within",
    )

    counts = joined.groupby(CELL_KEY).size()

    return cells_metric[CELL_KEY].map(counts).fillna(0).astype(int)


def nearest_distance_to_points(
    cells_metric: gpd.GeoDataFrame,
    points_metric: gpd.GeoDataFrame,
) -> pd.Series:
    """Calculate nearest point distance for each cell."""
    if points_metric.empty:
        return pd.Series(float("nan"), index=cells_metric[CELL_KEY])

    nearest = gpd.sjoin_nearest(
        cells_metric[[CELL_KEY, "geometry"]],
        points_metric[["geometry"]],
        how="left",
        distance_col="distance_m",
    )

    distances = nearest.groupby(CELL_KEY)["distance_m"].min()

    return cells_metric[CELL_KEY].map(distances)


def nearest_distance_to_points_by_city(
    cells_metric: gpd.GeoDataFrame,
    points_metric: gpd.GeoDataFrame,
) -> pd.Series:
    """Calculate nearest point distance within each city."""
    if points_metric.empty:
        return pd.Series(float("nan"), index=cells_metric[CELL_KEY])

    if CITY_KEY not in cells_metric.columns or CITY_KEY not in points_metric.columns:
        return nearest_distance_to_points(cells_metric, points_metric)

    output = pd.Series(float("nan"), index=cells_metric[CELL_KEY])

    for city_id, city_cells in cells_metric.groupby(CITY_KEY):
        city_points = points_metric[points_metric[CITY_KEY] == city_id].copy()

        if city_points.empty:
            continue

        distances = nearest_distance_to_points(city_cells, city_points)
        output.loc[city_cells[CELL_KEY].values] = distances.values

    return output


def build_service_accessibility_features(
    cells: gpd.GeoDataFrame,
    service_points: gpd.GeoDataFrame,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Build service accessibility features for cells."""
    selected_categories = categories or list(SERVICE_CATEGORIES)

    cells_metric = to_metric_crs(cells)
    services_metric = to_metric_crs(service_points) if not service_points.empty else service_points

    output = pd.DataFrame({CELL_KEY: cells_metric[CELL_KEY].values})
    area_km2 = service_area_km2(cells_metric).replace(0, pd.NA)

    total_count = pd.Series(0, index=cells_metric.index, dtype="int64")

    for category in selected_categories:
        if service_points.empty:
            category_points = services_metric
        else:
            category_points = services_metric[
                services_metric["service_category"] == category
            ].copy()

        count = count_points_within_cells(cells_metric, category_points)
        distance_m = nearest_distance_to_points_by_city(cells_metric, category_points)

        output[f"service_{category}_count"] = count.values
        output[f"service_{category}_density_per_km2"] = (
            count.reset_index(drop=True) / area_km2.reset_index(drop=True)
        ).astype(float)
        output[f"service_distance_to_nearest_{category}_m"] = distance_m.values
        output[f"service_distance_to_nearest_{category}_km"] = (
            distance_m.reset_index(drop=True) / 1000
        ).values

        total_count = total_count.reset_index(drop=True) + count.reset_index(drop=True)

    output["service_total_count"] = total_count.values
    output["service_total_density_per_km2"] = (
        total_count.reset_index(drop=True) / area_km2.reset_index(drop=True)
    ).astype(float)

    any_distance_m = nearest_distance_to_points_by_city(cells_metric, services_metric)
    output["service_distance_to_nearest_any_m"] = any_distance_m.values
    output["service_distance_to_nearest_any_km"] = (
        any_distance_m.reset_index(drop=True) / 1000
    ).values

    return output


def summarize_service_features(features: pd.DataFrame) -> dict[str, Any]:
    """Summarize service features."""
    count_columns = [
        column
        for column in features.columns
        if column.startswith("service_") and column.endswith("_count")
    ]

    return {
        "rows": int(len(features)),
        "cells": int(features[CELL_KEY].nunique()),
        "count_columns": count_columns,
        "total_services_assigned_to_cells": int(features["service_total_count"].sum()),
        "cells_with_services": int((features["service_total_count"] > 0).sum()),
    }
