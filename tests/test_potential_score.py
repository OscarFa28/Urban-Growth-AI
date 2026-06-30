import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from urban_growth.scoring.potential import (
    add_potential_ranks_and_tiers,
    calculate_potential_components,
    inverse_distance_score,
    select_potential_output_columns,
)


def test_inverse_distance_score():
    distances = pd.Series([0.0, 1000.0, 3000.0])
    scores = inverse_distance_score(distances, scale_m=1000.0)

    assert scores.iloc[0] == 100.0
    assert scores.iloc[1] == 50.0
    assert scores.iloc[2] == 25.0


def test_calculate_potential_components_creates_score():
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "year": [2024, 2024, 2024],
            "urban_growth_probability": [0.1, 0.9, 0.5],
            "built_probability_mean": [0.05, 0.3, 0.15],
            "distance_to_nearest_road_m": [100.0, 2000.0, 500.0],
            "economic_business_count_total": [1, 20, 5],
            "economic_distance_to_nearest_business_m": [1000.0, 100.0, 500.0],
            "population_density_per_km2": [50.0, 1000.0, 250.0],
        }
    )

    result = calculate_potential_components(frame)

    assert "model_probability_score" in result.columns
    assert "land_availability_score" in result.columns
    assert "road_accessibility_score" in result.columns
    assert "economic_access_score" in result.columns
    assert "population_pressure_score" in result.columns
    assert "urban_growth_potential_score" in result.columns

    assert result["urban_growth_potential_score"].between(0, 100).all()


def test_add_potential_ranks_and_tiers():
    frame = pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(100)],
            "city_id": ["mx_test"] * 100,
            "year": [2024] * 100,
            "urban_growth_potential_score": list(range(100)),
        }
    )

    result = add_potential_ranks_and_tiers(frame)

    top = result.sort_values("urban_growth_potential_score", ascending=False).iloc[0]

    assert top["urban_growth_potential_rank_year"] == 1
    assert top["urban_growth_potential_rank_city_year"] == 1
    assert top["urban_growth_potential_tier"] == "very_high"


def test_select_potential_output_columns_keeps_geometry():
    frame = gpd.GeoDataFrame(
        {
            "cell_id": ["a"],
            "year": [2024],
            "city_id": ["mx_test"],
            "urban_growth_probability": [0.8],
            "urban_growth_potential_score": [90.0],
            "extra_column": ["drop_me"],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )

    result = select_potential_output_columns(frame)

    assert "cell_id" in result.columns
    assert "urban_growth_probability" in result.columns
    assert "urban_growth_potential_score" in result.columns
    assert "geometry" in result.columns
    assert "extra_column" not in result.columns
    assert result.crs == frame.crs
