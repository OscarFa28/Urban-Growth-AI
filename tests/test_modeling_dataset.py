import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from urban_growth.modeling.dataset import build_modeling_dataset


def test_build_modeling_dataset_joins_cell_year_features():
    spatial = gpd.GeoDataFrame(
        {
            "cell_id": ["cell_a"],
            "city_id": ["mx_test"],
            "state": ["Aguascalientes"],
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
            "road_density_all_m_per_km2": [2000.0],
        }
    )

    land_cover = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a"],
            "year": [2016, 2017],
            "built_probability_mean": [0.2, 0.3],
            "built_label_frequency": [0.1, 0.2],
            "dw_observation_count": [20, 21],
        }
    )

    labels = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a"],
            "year": [2016, 2017],
            "target_year": [2017, 2018],
            "has_next_year_target": [True, True],
            "urbanized_next_year": [False, True],
            "strong_urban_growth_next_year": [False, False],
            "built_probability_change_next_year": [0.1, 0.2],
        }
    )

    demographic = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a"],
            "year": [2016, 2017],
            "population_total": [100.0, 110.0],
            "population_density_per_km2": [100.0, 110.0],
            "internet_access_rate": [0.5, 0.6],
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

    result = build_modeling_dataset(
        spatial_features=spatial,
        road_features=roads,
        land_cover_features=land_cover,
        label_features=labels,
        demographic_features=demographic,
        economic_features=economic,
        start_year=2016,
        end_year=2017,
    )

    assert len(result) == 2
    assert result.crs == spatial.crs
    assert "geometry" in result.columns

    row_2016 = result[result["year"] == 2016].iloc[0]
    row_2017 = result[result["year"] == 2017].iloc[0]

    assert row_2016["built_probability_mean"] == 0.2
    assert row_2016["population_total"] == 100.0
    assert row_2016["economic_business_count_total"] == 10
    assert row_2016["distance_to_nearest_road_m"] == 50.0
    assert not row_2016["urbanized_next_year"]

    assert row_2017["built_probability_mean"] == 0.3
    assert row_2017["population_total"] == 110.0
    assert row_2017["urbanized_next_year"]


def test_build_modeling_dataset_filters_years_and_incomplete_targets():
    spatial = gpd.GeoDataFrame(
        {"cell_id": ["cell_a"], "cell_area_km2": [1.0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:3857",
    )

    roads = pd.DataFrame({"cell_id": ["cell_a"]})
    economic = pd.DataFrame({"cell_id": ["cell_a"]})

    land_cover = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a", "cell_a"],
            "year": [2015, 2016, 2025],
            "built_probability_mean": [0.1, 0.2, 0.9],
        }
    )

    labels = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a", "cell_a"],
            "year": [2015, 2016, 2025],
            "target_year": [2016, 2017, None],
            "has_next_year_target": [True, True, False],
            "urbanized_next_year": [False, True, False],
        }
    )

    demographic = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_a", "cell_a"],
            "year": [2015, 2016, 2025],
            "population_total": [90.0, 100.0, 200.0],
        }
    )

    result = build_modeling_dataset(
        spatial_features=spatial,
        road_features=roads,
        land_cover_features=land_cover,
        label_features=labels,
        demographic_features=demographic,
        economic_features=economic,
        start_year=2016,
        end_year=2024,
    )

    assert len(result) == 1
    assert result.iloc[0]["year"] == 2016
    assert result.iloc[0]["population_total"] == 100.0
