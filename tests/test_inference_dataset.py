import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from urban_growth.inference.dataset import build_inference_dataset


def test_build_inference_dataset_joins_current_year_features():
    spatial = gpd.GeoDataFrame(
        {
            "cell_id": ["cell_a"],
            "city_id": ["mx_test"],
            "city_name": ["Test City"],
            "cell_area_km2": [1.0],
            "distance_to_city_center_m": [1000.0],
        },
        geometry=[box(0, 0, 1000, 1000)],
        crs="EPSG:3857",
    )

    roads = pd.DataFrame(
        {
            "cell_id": ["cell_a"],
            "distance_to_nearest_road_m": [50.0],
            "distance_to_major_road_m": [100.0],
            "distance_to_medium_road_m": [50.0],
            "distance_to_minor_road_m": [75.0],
            "distance_to_service_road_m": [200.0],
            "nearest_road_class": ["medium"],
        }
    )

    land_cover = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a"],
            "year": [2024, 2025],
            "built_probability_mean": [0.2, 0.3],
            "built_label_frequency": [0.1, 0.2],
        }
    )

    demographic = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a"],
            "year": [2024, 2025],
            "population_total": [100.0, 110.0],
            "population_density_per_km2": [100.0, 110.0],
        }
    )

    economic = pd.DataFrame(
        {
            "cell_id": ["cell_a"],
            "economic_business_count_total": [10],
            "economic_business_density_per_km2": [10.0],
            "economic_distance_to_nearest_business_m": [25.0],
        }
    )

    result = build_inference_dataset(
        spatial_features=spatial,
        road_features=roads,
        land_cover_features=land_cover,
        demographic_features=demographic,
        economic_features=economic,
        inference_year=2025,
        urban_threshold=0.35,
    )

    assert len(result) == 1
    assert result.iloc[0]["year"] == 2025
    assert result.iloc[0]["prediction_year"] == 2025
    assert result.iloc[0]["predicted_target_year"] == 2026
    assert result.iloc[0]["target_year"] == 2026
    assert result.iloc[0]["is_inference"]
    assert not result.iloc[0]["is_urban"]
    assert not result.iloc[0]["urbanized_next_year"]
    assert result.iloc[0]["population_total"] == 110.0
    assert result.iloc[0]["economic_business_count_total"] == 10
    assert result.crs == spatial.crs


def test_build_inference_dataset_marks_urban_cells():
    spatial = gpd.GeoDataFrame(
        {"cell_id": ["cell_a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:3857",
    )
    roads = pd.DataFrame(
        {
            "cell_id": ["cell_a"],
            "distance_to_major_road_m": [1.0],
            "distance_to_medium_road_m": [2.0],
            "distance_to_minor_road_m": [3.0],
            "distance_to_service_road_m": [4.0],
        }
    )
    land_cover = pd.DataFrame(
        {"cell_id": ["cell_a"], "year": [2025], "built_probability_mean": [0.5]}
    )
    demographic = pd.DataFrame({"cell_id": ["cell_a"], "year": [2025], "population_total": [10.0]})
    economic = pd.DataFrame({"cell_id": ["cell_a"], "economic_business_count_total": [1]})

    result = build_inference_dataset(
        spatial_features=spatial,
        road_features=roads,
        land_cover_features=land_cover,
        demographic_features=demographic,
        economic_features=economic,
        inference_year=2025,
        urban_threshold=0.35,
    )

    assert result.iloc[0]["is_urban"]
