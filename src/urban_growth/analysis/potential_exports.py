"""Potential score export helpers."""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import pandas as pd

DEFAULT_EXPORT_COLUMNS = [
    "cell_id",
    "year",
    "city_id",
    "city_name",
    "state",
    "municipality_name",
    "urban_growth_potential_score",
    "urban_growth_potential_percentile",
    "urban_growth_potential_tier",
    "urban_growth_potential_rank_year",
    "urban_growth_potential_rank_city_year",
    "urban_growth_probability",
    "urban_growth_model_threshold",
    "model_probability_score",
    "land_availability_score",
    "road_accessibility_score",
    "economic_access_score",
    "population_pressure_score",
    "built_probability_mean",
    "distance_to_nearest_road_m",
    "nearest_road_class",
    "economic_business_count_total",
    "economic_distance_to_nearest_business_m",
    "population_density_per_km2",
    "urbanized_next_year",
    "geometry",
]


def filter_potential_scores(
    frame: gpd.GeoDataFrame,
    year: int | None = None,
    city_ids: Iterable[str] | None = None,
) -> gpd.GeoDataFrame:
    """Filter potential score rows by year and city IDs."""
    output = frame.copy()

    if year is not None:
        output = output[output["year"] == year].copy()

    if city_ids is not None:
        city_id_set = set(city_ids)
        output = output[output["city_id"].isin(city_id_set)].copy()

    return gpd.GeoDataFrame(output, geometry="geometry", crs=frame.crs)


def select_top_percentile(
    frame: gpd.GeoDataFrame,
    percentile_threshold: float,
) -> gpd.GeoDataFrame:
    """Select rows above a potential percentile threshold."""
    if not 0 <= percentile_threshold <= 1:
        msg = "percentile_threshold must be between 0 and 1."
        raise ValueError(msg)

    output = frame[frame["urban_growth_potential_percentile"] >= percentile_threshold].copy()

    return gpd.GeoDataFrame(output, geometry="geometry", crs=frame.crs)


def select_top_n_by_city_year(
    frame: gpd.GeoDataFrame,
    top_n: int,
) -> gpd.GeoDataFrame:
    """Select top N potential cells per city-year."""
    if top_n <= 0:
        msg = "top_n must be greater than zero."
        raise ValueError(msg)

    output = frame[frame["urban_growth_potential_rank_city_year"] <= top_n].copy()

    return gpd.GeoDataFrame(output, geometry="geometry", crs=frame.crs)


def select_export_columns(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select compact export columns."""
    columns = [column for column in DEFAULT_EXPORT_COLUMNS if column in frame.columns]
    output = frame[columns].copy()

    if "geometry" in output.columns:
        return gpd.GeoDataFrame(output, geometry="geometry", crs=frame.crs)

    return gpd.GeoDataFrame(output, crs=frame.crs)


def add_lon_lat_for_csv(frame: gpd.GeoDataFrame) -> pd.DataFrame:
    """Add representative longitude and latitude columns and drop geometry."""
    if "geometry" not in frame.columns:
        return pd.DataFrame(frame)

    output = frame.copy()

    if output.crs is not None:
        output = output.to_crs("EPSG:4326")

    points = output.geometry.representative_point()

    output["lon"] = points.x
    output["lat"] = points.y

    return pd.DataFrame(output.drop(columns="geometry"))


def to_geojson_crs(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert export frame to GeoJSON-compatible CRS."""
    if frame.crs is None:
        return frame

    return frame.to_crs("EPSG:4326")


def summarize_export(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize exported potential score rows."""
    return {
        "rows": int(len(frame)),
        "cells": int(frame["cell_id"].nunique()) if "cell_id" in frame else None,
        "years": sorted(int(year) for year in frame["year"].unique()) if "year" in frame else [],
        "cities": int(frame["city_id"].nunique()) if "city_id" in frame else None,
        "positives": int(frame["urbanized_next_year"].sum())
        if "urbanized_next_year" in frame
        else None,
        "positive_rate": float(frame["urbanized_next_year"].mean())
        if "urbanized_next_year" in frame and len(frame) > 0
        else None,
        "mean_score": float(frame["urban_growth_potential_score"].mean())
        if "urban_growth_potential_score" in frame and len(frame) > 0
        else None,
    }
