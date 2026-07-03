import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from urban_growth.features.service_features import (
    build_service_accessibility_features,
    convert_services_to_points,
)


def test_convert_services_to_points_from_polygons():
    services = gpd.GeoDataFrame(
        {"name": ["School"], "amenity": ["school"]},
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:3857",
    )

    points = convert_services_to_points(
        services,
        category="education",
        city_id="mx_test",
    )

    assert len(points) == 1
    assert points.iloc[0]["service_category"] == "education"
    assert points.iloc[0]["city_id"] == "mx_test"
    assert points.geometry.iloc[0].geom_type == "Point"


def test_build_service_accessibility_features_counts_and_distances():
    cells = gpd.GeoDataFrame(
        {
            "cell_id": ["cell_a", "cell_b"],
            "cell_area_km2": [1.0, 1.0],
        },
        geometry=[
            box(0, 0, 1000, 1000),
            box(1000, 0, 2000, 1000),
        ],
        crs="EPSG:3857",
    )

    services = gpd.GeoDataFrame(
        {
            "service_category": ["education", "education", "health"],
        },
        geometry=[
            Point(100, 100),
            Point(600, 600),
            Point(1500, 500),
        ],
        crs="EPSG:3857",
    )

    features = build_service_accessibility_features(
        cells,
        services,
        categories=["education", "health"],
    )

    cell_a = features[features["cell_id"] == "cell_a"].iloc[0]
    cell_b = features[features["cell_id"] == "cell_b"].iloc[0]

    assert cell_a["service_education_count"] == 2
    assert cell_a["service_health_count"] == 0
    assert cell_a["service_total_count"] == 2
    assert cell_a["service_distance_to_nearest_education_m"] == 0

    assert cell_b["service_education_count"] == 0
    assert cell_b["service_health_count"] == 1
    assert cell_b["service_total_count"] == 1
    assert cell_b["service_distance_to_nearest_health_m"] == 0


def test_build_service_accessibility_features_handles_empty_services():
    cells = gpd.GeoDataFrame(
        {"cell_id": ["cell_a"], "cell_area_km2": [1.0]},
        geometry=[box(0, 0, 1000, 1000)],
        crs="EPSG:3857",
    )

    services = gpd.GeoDataFrame(
        {"service_category": pd.Series(dtype="object")},
        geometry=gpd.GeoSeries([], crs="EPSG:3857"),
        crs="EPSG:3857",
    )

    features = build_service_accessibility_features(
        cells,
        services,
        categories=["education"],
    )

    assert features.iloc[0]["service_education_count"] == 0
    assert features.iloc[0]["service_total_count"] == 0
    assert pd.isna(features.iloc[0]["service_distance_to_nearest_education_m"])
