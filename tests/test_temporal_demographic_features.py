import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from urban_growth.features.temporal_demographic_features import (
    build_cell_municipality_cvegeo,
    build_municipal_temporal_adjustments,
    build_temporal_demographic_features,
)


def test_build_cell_municipality_cvegeo_uses_state_when_needed():
    cells = pd.DataFrame(
        {
            "state": ["Aguascalientes"],
            "municipality_id": ["001"],
        }
    )

    result = build_cell_municipality_cvegeo(cells)

    assert result.iloc[0] == "01001"


def test_build_cell_municipality_cvegeo_uses_city_id_mapping():
    cells = pd.DataFrame(
        {
            "city_id": ["mx_leon", "mx_cdmx"],
            "state": ["Guanajuato", "Ciudad de México"],
            "municipality_id": ["mx_leon", None],
        }
    )

    result = build_cell_municipality_cvegeo(cells)

    assert result.iloc[0] == "11020"
    assert result.iloc[1] == "09000"


def test_build_municipal_temporal_adjustments_interpolates_counts():
    municipal_2010 = pd.DataFrame(
        {
            "municipality_cvegeo": ["01001"],
            "population_total": [100.0],
            "housing_units_total": [50.0],
            "average_schooling_years": [8.0],
        }
    )
    municipal_2020 = pd.DataFrame(
        {
            "municipality_cvegeo": ["01001"],
            "population_total": [200.0],
            "housing_units_total": [100.0],
            "average_schooling_years": [10.0],
        }
    )

    result = build_municipal_temporal_adjustments(
        municipal_2010=municipal_2010,
        municipal_2020=municipal_2020,
        start_year=2015,
        end_year=2020,
    )

    row_2015 = result[result["year"] == 2015].iloc[0]
    row_2020 = result[result["year"] == 2020].iloc[0]

    assert row_2015["population_total_municipal_ratio_to_2020"] == 0.75
    assert row_2015["housing_units_total_municipal_ratio_to_2020"] == 0.75
    assert row_2015["average_schooling_years_municipal_delta_from_2020"] == -1.0
    assert row_2020["population_total_municipal_ratio_to_2020"] == 1.0
    assert row_2020["average_schooling_years_municipal_delta_from_2020"] == 0.0


def test_build_temporal_demographic_features_scales_cell_counts():
    cells_2020 = gpd.GeoDataFrame(
        {
            "cell_id": ["cell_a"],
            "state": ["Aguascalientes"],
            "municipality_id": ["001"],
            "population_total": [20.0],
            "female_population": [12.0],
            "male_population": [8.0],
            "housing_units_total": [10.0],
            "occupied_housing_units": [8.0],
            "population_0_14": [4.0],
            "population_15_64": [14.0],
            "population_65_plus": [2.0],
            "average_schooling_years": [10.0],
            "demographic_cell_area_m2": [1_000_000.0],
        },
        geometry=[box(0, 0, 1000, 1000)],
        crs="EPSG:3857",
    )

    adjustments = pd.DataFrame(
        {
            "municipality_cvegeo": ["01001", "01001"],
            "year": [2015, 2020],
            "population_total_municipal_ratio_to_2020": [0.75, 1.0],
            "female_population_municipal_ratio_to_2020": [0.75, 1.0],
            "male_population_municipal_ratio_to_2020": [0.75, 1.0],
            "housing_units_total_municipal_ratio_to_2020": [0.8, 1.0],
            "occupied_housing_units_municipal_ratio_to_2020": [0.8, 1.0],
            "population_0_14_municipal_ratio_to_2020": [0.5, 1.0],
            "population_15_64_municipal_ratio_to_2020": [0.8, 1.0],
            "population_65_plus_municipal_ratio_to_2020": [1.2, 1.0],
            "average_schooling_years_municipal_delta_from_2020": [-1.0, 0.0],
        }
    )

    result = build_temporal_demographic_features(
        cells_2020=cells_2020,
        municipal_adjustments=adjustments,
        start_year=2015,
        end_year=2020,
    )

    row_2015 = result[result["year"] == 2015].iloc[0]
    row_2020 = result[result["year"] == 2020].iloc[0]

    assert row_2015["population_total"] == 15.0
    assert row_2015["housing_units_total"] == 8.0
    assert row_2015["average_schooling_years"] == 9.0
    assert row_2015["population_density_per_km2"] == 15.0
    assert row_2015["demographic_is_temporal_estimate"]

    assert row_2020["population_total"] == 20.0
    assert not row_2020["demographic_is_temporal_estimate"]
