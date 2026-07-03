"""DENUE-based service accessibility features."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from urban_growth.features.service_features import (
    CELL_KEY,
    CITY_KEY,
    nearest_distance_to_points_by_city,
    to_metric_crs,
)

DENUE_SERVICE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "education": ("61",),
    "health": ("62",),
    "retail": ("46",),
    "transport": ("48-49",),
    "food_lodging": ("72",),
    "recreation": ("71",),
    "government": ("93",),
}

DENUE_USE_COLUMNS = [
    "id",
    "clee",
    "nom_estab",
    "codigo_act",
    "nombre_act",
    "per_ocu",
    "cve_ent",
    "entidad",
    "cve_mun",
    "municipio",
    "latitud",
    "longitud",
]


def denue_code_from_path(path: Path) -> str:
    """Extract DENUE code from raw CSV filename."""
    return path.stem.replace("denue_inegi_", "").strip("_")


def category_for_denue_code(code: str) -> str | None:
    """Map DENUE sector/code file to service category."""
    for category, prefixes in DENUE_SERVICE_CATEGORIES.items():
        if any(code.startswith(prefix) for prefix in prefixes):
            return category

    return None


def find_denue_service_files(
    raw_denue_dir: Path,
    year: int,
    categories: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Find raw DENUE CSV files used for service accessibility."""
    selected_categories = set(categories or DENUE_SERVICE_CATEGORIES)
    year_dir = raw_denue_dir / str(year)

    files: list[tuple[Path, str]] = []

    for path in sorted(year_dir.glob("denue_00_*_csv/conjunto_de_datos/denue_inegi_*.csv")):
        code = denue_code_from_path(path)
        category = category_for_denue_code(code)

        if category is None or category not in selected_categories:
            continue

        files.append((path, category))

    return files


def _bounds_with_padding(
    cells_wgs84: gpd.GeoDataFrame,
    padding_degrees: float,
) -> tuple[float, float, float, float]:
    west, south, east, north = cells_wgs84.total_bounds

    return (
        west - padding_degrees,
        south - padding_degrees,
        east + padding_degrees,
        north + padding_degrees,
    )


def read_denue_points_from_csv(
    path: Path,
    category: str,
    bounds: tuple[float, float, float, float],
    chunksize: int = 200_000,
) -> gpd.GeoDataFrame:
    """Read DENUE points from one CSV and filter to bounds."""
    west, south, east, north = bounds
    frames: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        encoding="latin1",
        encoding_errors="replace",
        dtype=str,
        usecols=lambda column: column in DENUE_USE_COLUMNS,
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk["latitud"] = pd.to_numeric(chunk["latitud"], errors="coerce")
        chunk["longitud"] = pd.to_numeric(chunk["longitud"], errors="coerce")

        chunk = chunk.dropna(subset=["latitud", "longitud"])

        chunk = chunk[
            chunk["longitud"].between(west, east) & chunk["latitud"].between(south, north)
        ].copy()

        if chunk.empty:
            continue

        chunk["denue_service_category"] = category
        chunk["geometry"] = [
            Point(lon, lat)
            for lon, lat in zip(
                chunk["longitud"],
                chunk["latitud"],
                strict=True,
            )
        ]

        frames.append(chunk)

    if not frames:
        return gpd.GeoDataFrame(
            columns=[*DENUE_USE_COLUMNS, "denue_service_category", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    frame = pd.concat(frames, ignore_index=True)

    return gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")


def load_denue_service_points(
    raw_denue_dir: Path,
    year: int,
    cells: gpd.GeoDataFrame,
    categories: list[str] | None = None,
    padding_degrees: float = 0.05,
) -> gpd.GeoDataFrame:
    """Load DENUE service points for the selected categories."""
    cells_wgs84 = cells.to_crs("EPSG:4326")
    bounds = _bounds_with_padding(cells_wgs84, padding_degrees)

    files = find_denue_service_files(
        raw_denue_dir=raw_denue_dir,
        year=year,
        categories=categories,
    )

    if not files:
        raise FileNotFoundError(f"No DENUE service files found in {raw_denue_dir / str(year)}")

    frames: list[gpd.GeoDataFrame] = []

    for path, category in files:
        points = read_denue_points_from_csv(
            path=path,
            category=category,
            bounds=bounds,
        )
        print(f"{category}: {path.name}: {len(points):,} points in bounds")

        if points.empty:
            continue

        frames.append(points)

    if not frames:
        return gpd.GeoDataFrame(
            columns=[*DENUE_USE_COLUMNS, "denue_service_category", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )


def assign_points_to_cells(
    cells_metric: gpd.GeoDataFrame,
    points_metric: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign DENUE points to grid cells."""
    if points_metric.empty:
        return points_metric.assign(**{CELL_KEY: pd.NA, CITY_KEY: pd.NA})

    joined = gpd.sjoin(
        points_metric,
        cells_metric[[CELL_KEY, CITY_KEY, "geometry"]],
        how="inner",
        predicate="within",
    )

    return joined.drop(columns=["index_right"]).reset_index(drop=True)


def _count_points_by_cell(
    cells_metric: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
) -> pd.Series:
    if points.empty:
        return pd.Series(0, index=cells_metric[CELL_KEY], dtype="int64")

    counts = points.groupby(CELL_KEY).size()

    return cells_metric[CELL_KEY].map(counts).fillna(0).astype("int64")


def build_denue_service_accessibility_features(
    cells: gpd.GeoDataFrame,
    service_points: gpd.GeoDataFrame,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Build DENUE service count, density, and nearest-distance features."""
    selected_categories = categories or list(DENUE_SERVICE_CATEGORIES)

    cells_metric = to_metric_crs(cells)
    points_metric = to_metric_crs(service_points)

    area_km2 = cells_metric.geometry.area / 1_000_000

    assigned_points = assign_points_to_cells(
        cells_metric=cells_metric,
        points_metric=points_metric,
    )

    features = pd.DataFrame({CELL_KEY: cells_metric[CELL_KEY].values})

    for category in selected_categories:
        category_points = assigned_points[
            assigned_points["denue_service_category"] == category
        ].copy()

        count_column = f"denue_service_{category}_count"
        density_column = f"denue_service_{category}_density_per_km2"
        distance_m_column = f"denue_service_distance_to_nearest_{category}_m"
        distance_km_column = f"denue_service_distance_to_nearest_{category}_km"

        counts = _count_points_by_cell(cells_metric, category_points)
        distance_m = nearest_distance_to_points_by_city(cells_metric, category_points)

        features[count_column] = counts.values
        features[density_column] = counts.values / area_km2.values
        features[distance_m_column] = distance_m.values
        features[distance_km_column] = distance_m.values / 1_000

    total_counts = _count_points_by_cell(cells_metric, assigned_points)
    any_distance_m = nearest_distance_to_points_by_city(cells_metric, assigned_points)

    features["denue_service_total_count"] = total_counts.values
    features["denue_service_total_density_per_km2"] = total_counts.values / area_km2.values
    features["denue_service_distance_to_nearest_any_m"] = any_distance_m.values
    features["denue_service_distance_to_nearest_any_km"] = any_distance_m.values / 1_000

    return features


def summarize_denue_service_features(features: pd.DataFrame) -> dict[str, float | int]:
    """Summarize DENUE service features."""
    return {
        "rows": len(features),
        "cells": features[CELL_KEY].nunique(),
        "total_services_assigned": int(features["denue_service_total_count"].sum()),
        "cells_with_services": int((features["denue_service_total_count"] > 0).sum()),
    }
