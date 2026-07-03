"""Inference dataset assembly for current-year urban growth scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from urban_growth.modeling.dataset import (
    _merge_one_to_one,
    _prepare_demographic_features,
    _prepare_denue_service_features,
    _prepare_economic_features,
    _prepare_land_cover_features,
    _prepare_road_features,
    _prepare_static_features,
)

CELL_KEY = "cell_id"
TIME_KEY = "year"


INFERENCE_OUTPUT_COLUMNS = [
    "cell_id",
    "year",
    "prediction_year",
    "predicted_target_year",
    "target_year",
    "is_inference",
    "city_id",
    "city_name",
    "country",
    "country_code",
    "state",
    "spatial_unit_id",
    "spatial_unit_type",
    "municipality_id",
    "municipality_name",
    "municipality_cvegeo",
    "metro_area_id",
    "metro_area_name",
    "size_category",
    "density_category",
    "primary_behavior_category",
    "grid_size_m",
    "row",
    "col",
    "is_urban",
    "urban_threshold",
    "built_probability_mean",
    "built_area_percentage_approx",
    "built_label_frequency",
    "built_label_percentage",
    "distance_to_city_center_m",
    "distance_to_boundary_m",
    "distance_to_nearest_road_m",
    "distance_to_nearest_road_km",
    "nearest_road_class",
    "road_density_all_m_per_km2",
    "population_total",
    "population_density_per_km2",
    "economic_business_count_total",
    "economic_business_density_per_km2",
    "economic_distance_to_nearest_business_m",
    "economic_distance_to_nearest_business_km",
    "denue_service_total_count",
    "denue_service_total_density_per_km2",
    "denue_service_distance_to_nearest_any_m",
    "denue_service_distance_to_nearest_any_km",
    "geometry",
]


def build_inference_dataset(
    spatial_features: gpd.GeoDataFrame,
    road_features: pd.DataFrame,
    land_cover_features: pd.DataFrame,
    demographic_features: pd.DataFrame,
    economic_features: pd.DataFrame,
    denue_service_features: pd.DataFrame | None = None,
    inference_year: int = 2025,
    urban_threshold: float = 0.35,
) -> gpd.GeoDataFrame:
    """Build current-year inference dataset without future target labels."""
    land_cover = _prepare_land_cover_features(land_cover_features)
    land_cover = land_cover[land_cover[TIME_KEY] == inference_year].copy()

    if land_cover.empty:
        msg = f"No land cover rows found for inference_year={inference_year}"
        raise ValueError(msg)

    demographics = _prepare_demographic_features(demographic_features)
    demographics = demographics[demographics[TIME_KEY] == inference_year].copy()

    if demographics.empty:
        msg = f"No demographic rows found for inference_year={inference_year}"
        raise ValueError(msg)

    static = _prepare_static_features(spatial_features)
    roads = _prepare_road_features(road_features)
    economic = _prepare_economic_features(economic_features)
    denue_services = None

    if denue_service_features is not None:
        denue_services = _prepare_denue_service_features(denue_service_features)

    inference = land_cover

    inference = _merge_one_to_one(
        inference,
        demographics,
        on=[CELL_KEY, TIME_KEY],
        source_name="demographic_features",
    )
    inference = _merge_one_to_one(
        inference,
        roads,
        on=[CELL_KEY],
        source_name="road_features",
    )
    inference = _merge_one_to_one(
        inference,
        economic,
        on=[CELL_KEY],
        source_name="economic_features",
    )

    if denue_services is not None:
        inference = _merge_one_to_one(
            inference,
            denue_services,
            on=[CELL_KEY],
            source_name="denue_service_features",
        )

    inference = _merge_one_to_one(
        inference,
        static,
        on=[CELL_KEY],
        source_name="spatial_features",
    )

    inference["prediction_year"] = inference_year
    inference["predicted_target_year"] = inference_year + 1
    inference["target_year"] = inference_year + 1
    inference["is_inference"] = True
    inference["urban_threshold"] = urban_threshold

    inference["is_urban"] = (
        pd.to_numeric(inference["built_probability_mean"], errors="coerce").fillna(0)
        >= urban_threshold
    )

    # Placeholder columns required by the existing scoring pipeline.
    # They are not historical labels for inference output.
    inference["has_next_year_target"] = False
    inference["urbanized_next_year"] = False
    inference["strong_urban_growth_next_year"] = False

    return gpd.GeoDataFrame(
        inference,
        geometry="geometry",
        crs=spatial_features.crs,
    )


def read_inference_inputs(
    spatial_path: Path,
    road_path: Path,
    land_cover_path: Path,
    demographic_path: Path,
    economic_path: Path,
    denue_service_path: Path | None = None,
) -> dict[str, Any]:
    """Read all inference dataset inputs."""
    inputs = {
        "spatial_features": gpd.read_parquet(spatial_path),
        "road_features": pd.read_parquet(road_path),
        "land_cover_features": pd.read_parquet(land_cover_path),
        "demographic_features": pd.read_parquet(demographic_path),
        "economic_features": pd.read_parquet(economic_path),
    }

    if denue_service_path is not None:
        inputs["denue_service_features"] = pd.read_parquet(denue_service_path)

    return inputs


def select_inference_output_columns(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select compact inference columns."""
    columns = [column for column in INFERENCE_OUTPUT_COLUMNS if column in frame.columns]
    output = frame[columns].copy()

    if "geometry" in output.columns:
        return gpd.GeoDataFrame(output, geometry="geometry", crs=frame.crs)

    return gpd.GeoDataFrame(output, crs=frame.crs)


def summarize_inference_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize inference dataset."""
    return {
        "rows": int(len(frame)),
        "cells": int(frame["cell_id"].nunique()),
        "years": sorted(int(year) for year in frame["year"].unique()),
        "prediction_years": sorted(int(year) for year in frame["prediction_year"].unique())
        if "prediction_year" in frame
        else [],
        "urban_cells": int(frame["is_urban"].sum()) if "is_urban" in frame else None,
        "candidate_cells": int((~frame["is_urban"]).sum()) if "is_urban" in frame else None,
    }
