"""Modeling dataset assembly."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

CELL_KEY = "cell_id"
TIME_KEY = "year"


STATIC_ID_COLUMNS = [
    "cell_id",
    "cell_index",
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
    "behavior_categories",
    "primary_behavior_category",
    "grid_size_m",
    "row",
    "col",
    "nominal_area_m2",
    "area_m2",
    "coverage_ratio",
]

SPATIAL_FEATURE_COLUMNS = [
    "cell_centroid_lon",
    "cell_centroid_lat",
    "city_center_lon",
    "city_center_lat",
    "distance_to_city_center_m",
    "distance_to_city_center_km",
    "distance_to_boundary_m",
    "distance_to_boundary_km",
    "cell_area_km2",
    "normalized_distance_to_city_center",
    "normalized_distance_to_boundary",
    "is_boundary_cell",
]

ROAD_FEATURE_PREFIXES = (
    "road_",
    "distance_to_nearest_road",
    "distance_to_major_road",
    "distance_to_medium_road",
    "distance_to_minor_road",
    "distance_to_service_road",
    "nearest_road_",
)

LAND_COVER_FEATURE_COLUMNS = [
    "water_probability_mean",
    "trees_probability_mean",
    "grass_probability_mean",
    "flooded_vegetation_probability_mean",
    "crops_probability_mean",
    "shrub_and_scrub_probability_mean",
    "built_probability_mean",
    "bare_probability_mean",
    "snow_and_ice_probability_mean",
    "built_area_percentage_approx",
    "built_label_frequency",
    "built_label_percentage",
    "dw_observation_count",
]

LABEL_COLUMNS = [
    "target_year",
    "built_probability_next_year",
    "built_probability_change_next_year",
    "built_label_frequency_next_year",
    "built_label_frequency_change_next_year",
    "is_urban",
    "is_urban_next_year",
    "has_next_year_target",
    "urbanized_next_year",
    "strong_urban_growth_next_year",
    "urban_state_transition",
    "urban_threshold",
    "strong_growth_threshold",
]

DEMOGRAPHIC_PREFIXES = (
    "population_",
    "female_population",
    "male_population",
    "economically_",
    "employed_",
    "unemployed_",
    "health_",
    "housing_",
    "occupied_",
    "private_",
    "electricity_",
    "piped_",
    "drainage_",
    "internet_",
    "car_",
    "motorcycle_",
    "bicycle_",
    "computer_",
    "cellphone_",
    "avg_",
    "average_",
    "source_average_",
    "dependency_",
    "working_age_",
    "children_",
    "elderly_",
    "young_",
    "school_",
    "demographic_",
)

ECONOMIC_PREFIXES = ("economic_",)
DENUE_SERVICE_PREFIXES = ("denue_service_",)


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _columns_by_prefix(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [
        column for column in df.columns if any(column.startswith(prefix) for prefix in prefixes)
    ]


def _drop_geometry_if_present(df: pd.DataFrame) -> pd.DataFrame:
    if "geometry" in df.columns:
        return df.drop(columns="geometry")
    return df


def _normalize_year(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    if TIME_KEY in output.columns:
        output[TIME_KEY] = output[TIME_KEY].astype(int)

    if "target_year" in output.columns:
        output["target_year"] = output["target_year"].astype("Int64")

    return output


def _prepare_static_features(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    columns = _available_columns(df, STATIC_ID_COLUMNS + SPATIAL_FEATURE_COLUMNS)
    columns = list(dict.fromkeys(columns))

    if "geometry" in df.columns:
        columns.append("geometry")

    return df[columns].copy()


def _repair_nearest_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """Repair nearest-road fields using class-specific road distances."""
    output = df.copy()

    distance_columns = {
        "major": "distance_to_major_road_m",
        "medium": "distance_to_medium_road_m",
        "minor": "distance_to_minor_road_m",
        "service": "distance_to_service_road_m",
    }
    available_distance_columns = [
        column for column in distance_columns.values() if column in output.columns
    ]

    if not available_distance_columns:
        return output

    distances = output[available_distance_columns].astype(float)
    fallback_distance = distances.min(axis=1)

    if "distance_to_nearest_road_m" not in output.columns:
        output["distance_to_nearest_road_m"] = fallback_distance
    else:
        output["distance_to_nearest_road_m"] = output["distance_to_nearest_road_m"].fillna(
            fallback_distance
        )

    output["distance_to_nearest_road_km"] = output["distance_to_nearest_road_m"] / 1000

    class_by_column = {
        column_name: road_class
        for road_class, column_name in distance_columns.items()
        if column_name in output.columns
    }

    nearest_class = distances.idxmin(axis=1).map(class_by_column)

    if "nearest_road_class" not in output.columns:
        output["nearest_road_class"] = nearest_class
    else:
        output["nearest_road_class"] = output["nearest_road_class"].fillna(nearest_class)

    return output


def _prepare_road_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [CELL_KEY]
    columns += _columns_by_prefix(df, ROAD_FEATURE_PREFIXES)
    columns = list(dict.fromkeys(_available_columns(df, columns)))

    road_features = _drop_geometry_if_present(df[columns].copy())
    return _repair_nearest_road_features(road_features)


def _prepare_land_cover_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [CELL_KEY, TIME_KEY]
    columns += _available_columns(df, LAND_COVER_FEATURE_COLUMNS)
    columns = list(dict.fromkeys(columns))

    return _normalize_year(_drop_geometry_if_present(df[columns].copy()))


def _prepare_label_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [CELL_KEY, TIME_KEY]
    columns += _available_columns(df, LABEL_COLUMNS)
    columns = list(dict.fromkeys(columns))

    return _normalize_year(_drop_geometry_if_present(df[columns].copy()))


def _prepare_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [CELL_KEY, TIME_KEY]
    columns += _columns_by_prefix(df, DEMOGRAPHIC_PREFIXES)
    columns = list(dict.fromkeys(_available_columns(df, columns)))

    return _normalize_year(_drop_geometry_if_present(df[columns].copy()))


def _prepare_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = [CELL_KEY]
    columns += _columns_by_prefix(df, ECONOMIC_PREFIXES)
    columns = list(dict.fromkeys(_available_columns(df, columns)))

    return _drop_geometry_if_present(df[columns].copy())


def _prepare_denue_service_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare static DENUE service accessibility features."""
    columns = [CELL_KEY]
    columns += _columns_by_prefix(df, DENUE_SERVICE_PREFIXES)
    columns = list(dict.fromkeys(_available_columns(df, columns)))

    return _drop_geometry_if_present(df[columns].copy())


def _merge_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    source_name: str,
) -> pd.DataFrame:
    duplicated = right.duplicated(subset=on).sum()

    if duplicated:
        msg = f"{source_name} has {duplicated} duplicated rows for keys {on}"
        raise ValueError(msg)

    return left.merge(right, on=on, how="left")


def build_modeling_dataset(
    spatial_features: gpd.GeoDataFrame,
    road_features: pd.DataFrame,
    land_cover_features: pd.DataFrame,
    label_features: pd.DataFrame,
    demographic_features: pd.DataFrame,
    economic_features: pd.DataFrame,
    denue_service_features: pd.DataFrame | None = None,
    start_year: int = 2016,
    end_year: int = 2024,
) -> gpd.GeoDataFrame:
    """Build final modeling dataset by cell-year."""
    labels = _prepare_label_features(label_features)
    labels = labels[
        labels[TIME_KEY].between(start_year, end_year) & labels["has_next_year_target"].astype(bool)
    ].copy()

    modeling = labels

    land_cover = _prepare_land_cover_features(land_cover_features)
    land_cover = land_cover[land_cover[TIME_KEY].between(start_year, end_year)].copy()

    demographics = _prepare_demographic_features(demographic_features)
    demographics = demographics[demographics[TIME_KEY].between(start_year, end_year)].copy()

    static = _prepare_static_features(spatial_features)
    roads = _prepare_road_features(road_features)
    economic = _prepare_economic_features(economic_features)
    denue_services = None

    if denue_service_features is not None:
        denue_services = _prepare_denue_service_features(denue_service_features)

    modeling = _merge_one_to_one(
        modeling,
        land_cover,
        on=[CELL_KEY, TIME_KEY],
        source_name="land_cover_features",
    )
    modeling = _merge_one_to_one(
        modeling,
        demographics,
        on=[CELL_KEY, TIME_KEY],
        source_name="demographic_features",
    )
    modeling = _merge_one_to_one(
        modeling,
        roads,
        on=[CELL_KEY],
        source_name="road_features",
    )
    modeling = _merge_one_to_one(
        modeling,
        economic,
        on=[CELL_KEY],
        source_name="economic_features",
    )

    if denue_services is not None:
        modeling = _merge_one_to_one(
            modeling,
            denue_services,
            on=[CELL_KEY],
            source_name="denue_service_features",
        )

    modeling = _merge_one_to_one(
        modeling,
        static,
        on=[CELL_KEY],
        source_name="spatial_features",
    )

    if "geometry" in modeling.columns:
        return gpd.GeoDataFrame(modeling, geometry="geometry", crs=spatial_features.crs)

    return gpd.GeoDataFrame(modeling, crs=spatial_features.crs)


def read_modeling_inputs(
    spatial_path: Path,
    road_path: Path,
    land_cover_path: Path,
    labels_path: Path,
    demographic_path: Path,
    economic_path: Path,
    denue_service_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Read all modeling dataset inputs."""
    inputs = {
        "spatial_features": gpd.read_parquet(spatial_path),
        "road_features": pd.read_parquet(road_path),
        "land_cover_features": pd.read_parquet(land_cover_path),
        "label_features": pd.read_parquet(labels_path),
        "demographic_features": pd.read_parquet(demographic_path),
        "economic_features": pd.read_parquet(economic_path),
    }

    if denue_service_path is not None:
        inputs["denue_service_features"] = pd.read_parquet(denue_service_path)

    return inputs
