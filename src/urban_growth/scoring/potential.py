"""Urban growth potential scoring."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from urban_growth.modeling.training import (
    TARGET_COLUMN,
    predict_scores,
    prepare_modeling_data,
)

POTENTIAL_COMPONENT_WEIGHTS = {
    "model_probability_score": 0.70,
    "land_availability_score": 0.10,
    "road_accessibility_score": 0.08,
    "economic_access_score": 0.07,
    "population_pressure_score": 0.05,
}

POTENTIAL_OUTPUT_COLUMNS = [
    "cell_id",
    "year",
    "target_year",
    "city_id",
    "city_name",
    "state",
    "spatial_unit_id",
    "spatial_unit_type",
    "municipality_id",
    "municipality_name",
    "metro_area_id",
    "metro_area_name",
    "size_category",
    "density_category",
    "primary_behavior_category",
    "grid_size_m",
    "row",
    "col",
    "is_urban",
    "urbanized_next_year",
    "strong_urban_growth_next_year",
    "built_probability_mean",
    "built_label_frequency",
    "distance_to_nearest_road_m",
    "distance_to_nearest_road_km",
    "nearest_road_class",
    "population_total",
    "population_density_per_km2",
    "economic_business_count_total",
    "economic_business_density_per_km2",
    "economic_distance_to_nearest_business_m",
    "economic_distance_to_nearest_business_km",
    "urban_growth_probability",
    "urban_growth_model_threshold",
    "model_probability_score",
    "land_availability_score",
    "road_accessibility_score",
    "economic_access_score",
    "population_pressure_score",
    "urban_growth_potential_score",
    "urban_growth_potential_percentile",
    "urban_growth_potential_tier",
    "urban_growth_potential_rank_year",
    "urban_growth_potential_rank_city_year",
    "geometry",
]


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_numeric_column(
    frame: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)

    return _clean_numeric(frame[column]).fillna(default)


def _percentile_score_by_year(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = _clean_numeric(frame[column])

    if "year" not in frame.columns:
        return values.rank(pct=True).fillna(0.0)

    return values.groupby(frame["year"]).rank(pct=True).fillna(0.0)


def inverse_distance_score(
    distance_m: pd.Series,
    scale_m: float = 1000.0,
) -> pd.Series:
    """Convert distance in meters into a 0-100 accessibility score."""
    distance = _clean_numeric(distance_m).clip(lower=0)
    score = 1 / (1 + (distance / scale_m))
    return (score * 100).fillna(0.0)


def calculate_potential_components(
    frame: pd.DataFrame,
    urban_threshold: float = 0.35,
    model_threshold: float = 0.5,
) -> pd.DataFrame:
    """Calculate interpretable urban growth potential components."""
    output = frame.copy()

    model_probability = _safe_numeric_column(
        output,
        "urban_growth_probability",
        default=0.0,
    ).clip(lower=0)

    safe_model_threshold = max(float(model_threshold), 1e-9)

    threshold_relative_score = (model_probability / safe_model_threshold).clip(0, 1) * 100
    probability_rank_score = _percentile_score_by_year(output, "urban_growth_probability") * 100

    output["urban_growth_model_threshold"] = safe_model_threshold
    output["model_probability_score"] = (
        0.80 * threshold_relative_score + 0.20 * probability_rank_score
    ).clip(0, 100)

    built_probability = _safe_numeric_column(
        output,
        "built_probability_mean",
        default=urban_threshold,
    )
    output["land_availability_score"] = (
        (urban_threshold - built_probability) / urban_threshold
    ).clip(0, 1) * 100

    output["road_accessibility_score"] = inverse_distance_score(
        _safe_numeric_column(output, "distance_to_nearest_road_m"),
        scale_m=1000.0,
    )

    business_density_score = (
        np.log1p(_safe_numeric_column(output, "economic_business_count_total"))
        .groupby(output["year"])
        .rank(pct=True)
        .fillna(0.0)
        * 100
    )

    business_distance_score = inverse_distance_score(
        _safe_numeric_column(output, "economic_distance_to_nearest_business_m"),
        scale_m=1000.0,
    )

    output["economic_access_score"] = 0.5 * business_density_score + 0.5 * business_distance_score

    population_density = np.log1p(_safe_numeric_column(output, "population_density_per_km2"))
    output["population_pressure_score"] = (
        population_density.groupby(output["year"]).rank(pct=True).fillna(0.0) * 100
    )

    output["urban_growth_potential_score"] = 0.0
    for column, weight in POTENTIAL_COMPONENT_WEIGHTS.items():
        output["urban_growth_potential_score"] += output[column].fillna(0) * weight

    output["urban_growth_potential_score"] = output["urban_growth_potential_score"].clip(0, 100)

    return output


def add_potential_ranks_and_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    """Add percentile, ranks and tier labels to potential scores."""
    output = frame.copy()

    output["urban_growth_potential_percentile"] = (
        output["urban_growth_potential_score"].groupby(output["year"]).rank(pct=True).fillna(0.0)
    )

    output["urban_growth_potential_rank_year"] = (
        output.groupby("year")["urban_growth_potential_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    if "city_id" in output.columns:
        output["urban_growth_potential_rank_city_year"] = (
            output.groupby(["city_id", "year"])["urban_growth_potential_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    else:
        output["urban_growth_potential_rank_city_year"] = pd.NA

    percentile = output["urban_growth_potential_percentile"]

    output["urban_growth_potential_tier"] = np.select(
        [
            percentile >= 0.99,
            percentile >= 0.95,
            percentile >= 0.80,
            percentile >= 0.50,
        ],
        [
            "very_high",
            "high",
            "medium",
            "low",
        ],
        default="very_low",
    )

    return output


def score_modeling_dataset(
    frame: gpd.GeoDataFrame,
    model_bundle: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Score urban growth potential using a trained model bundle."""
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    candidate_only = bool(model_bundle.get("candidate_only", True))

    prepared = prepare_modeling_data(
        frame,
        target_column=TARGET_COLUMN,
        candidate_only=candidate_only,
    )

    features = prepared.features.copy()

    for column in feature_columns:
        if column not in features.columns:
            features[column] = np.nan

    features = features[feature_columns]

    scores = predict_scores(model, features)

    scored = prepared.frame.copy()
    scored["urban_growth_probability"] = scores

    model_threshold = float(model_bundle.get("threshold", 0.5))
    scored = calculate_potential_components(
        scored,
        model_threshold=model_threshold,
    )
    scored = add_potential_ranks_and_tiers(scored)

    if "geometry" in scored.columns:
        return gpd.GeoDataFrame(scored, geometry="geometry", crs=frame.crs)

    return gpd.GeoDataFrame(scored, crs=frame.crs)


def select_potential_output_columns(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select compact output columns for potential score files."""
    columns = [column for column in POTENTIAL_OUTPUT_COLUMNS if column in frame.columns]
    output = frame[columns].copy()

    if "geometry" in output.columns:
        return gpd.GeoDataFrame(output, geometry="geometry", crs=frame.crs)

    return gpd.GeoDataFrame(output, crs=frame.crs)


def summarize_potential_scores(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize potential score output."""
    return {
        "rows": int(len(frame)),
        "cells": int(frame["cell_id"].nunique()) if "cell_id" in frame else None,
        "years": sorted(int(year) for year in frame["year"].unique()),
        "mean_score": float(frame["urban_growth_potential_score"].mean()),
        "max_score": float(frame["urban_growth_potential_score"].max()),
        "very_high_count": int((frame["urban_growth_potential_tier"] == "very_high").sum()),
        "high_count": int((frame["urban_growth_potential_tier"] == "high").sum()),
        "medium_count": int((frame["urban_growth_potential_tier"] == "medium").sum()),
    }
