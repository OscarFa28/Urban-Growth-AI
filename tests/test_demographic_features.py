from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from urban_growth.features.demographic_features import (
    add_demographic_features,
    build_inegi_ageb_cvegeo,
    clean_inegi_numeric,
    filter_ageb_total_rows,
    find_ageb_shapefiles,
)


def test_build_inegi_ageb_cvegeo_handles_padded_values_and_letters():
    df = pd.DataFrame(
        {
            "ENTIDAD": ["1"],
            "MUN": ["1"],
            "LOC": ["1"],
            "AGEB": ["006A"],
        }
    )

    result = build_inegi_ageb_cvegeo(df)

    assert result.iloc[0] == "010010001006A"


def test_filter_ageb_total_rows_keeps_only_ageb_totals():
    df = pd.DataFrame(
        {
            "AGEB": ["0000", "0017", "0017", "006A"],
            "MZA": ["000", "000", "001", "000"],
        }
    )

    result = filter_ageb_total_rows(df)

    assert result[["AGEB", "MZA"]].values.tolist() == [
        ["0017", "000"],
        ["006A", "000"],
    ]


def test_clean_inegi_numeric_converts_reserved_values_to_zero():
    series = pd.Series(["10", "*", "1,250", "", "N/D"])

    result = clean_inegi_numeric(series)

    assert result.tolist() == [10.0, 0.0, 1250.0, 0.0, 0.0]


def test_find_ageb_shapefiles_ignores_integrated_00_file(tmp_path: Path):
    state_dir = tmp_path / "01_aguascalientes" / "conjunto_de_datos"
    integrated_dir = tmp_path / "MG_2020_Integrado" / "conjunto_de_datos"
    state_dir.mkdir(parents=True)
    integrated_dir.mkdir(parents=True)

    state_path = state_dir / "01a.shp"
    integrated_path = integrated_dir / "00a.shp"
    state_path.touch()
    integrated_path.touch()

    result = find_ageb_shapefiles(tmp_path)

    assert result == [state_path]


def test_add_demographic_features_allocates_counts_by_ageb_area():
    cells = gpd.GeoDataFrame(
        {
            "cell_id": ["cell_a"],
        },
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:3857",
    )

    ageb_data = gpd.GeoDataFrame(
        {
            "CVEGEO": ["0100100010017"],
            "POBTOT": [100.0],
            "POBFEM": [60.0],
            "POBMAS": [40.0],
            "POB0_14": [20.0],
            "POB15_64": [70.0],
            "POB65_MAS": [10.0],
            "P_12YMAS": [80.0],
            "PEA": [50.0],
            "POCUPADA": [45.0],
            "PDESOCUP": [5.0],
            "VIVTOT": [40.0],
            "TVIVHAB": [30.0],
            "VPH_INTER": [15.0],
            "VPH_AUTOM": [12.0],
            "GRAPROES": [10.0],
        },
        geometry=[box(0, 0, 2, 1)],
        crs="EPSG:3857",
    )

    result = add_demographic_features(cells, ageb_data)

    row = result.iloc[0]

    assert row["population_total"] == 50.0
    assert row["female_population"] == 30.0
    assert row["male_population"] == 20.0
    assert row["housing_units_total"] == 20.0
    assert row["occupied_housing_units"] == 15.0
    assert row["female_share"] == 0.6
    assert row["employment_rate"] == 0.9
    assert row["internet_access_rate"] == 0.5
    assert row["average_schooling_years"] == 10.0
    assert row["demographic_source_ageb_count"] == 1
    assert row["demographic_source_area_coverage_ratio"] == 1.0


def test_build_inegi_block_cvegeo_handles_padded_values():
    from urban_growth.features.demographic_features import build_inegi_block_cvegeo

    df = pd.DataFrame(
        {
            "ENTIDAD": ["1"],
            "MUN": ["1"],
            "LOC": ["1"],
            "AGEB": ["006A"],
            "MZA": ["7"],
        }
    )

    result = build_inegi_block_cvegeo(df)

    assert result.iloc[0] == "010010001006A007"


def test_filter_block_rows_keeps_only_blocks():
    from urban_growth.features.demographic_features import filter_block_rows

    df = pd.DataFrame(
        {
            "AGEB": ["0000", "0017", "0017", "006A"],
            "MZA": ["000", "000", "001", "012"],
        }
    )

    result = filter_block_rows(df)

    assert result[["AGEB", "MZA"]].values.tolist() == [
        ["0017", "001"],
        ["006A", "012"],
    ]
