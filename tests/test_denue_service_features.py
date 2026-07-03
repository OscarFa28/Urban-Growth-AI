from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from urban_growth.features.denue_service_features import (
    build_denue_service_accessibility_features,
    category_for_denue_code,
    denue_code_from_path,
)


def test_denue_code_from_path() -> None:
    path = Path(
        "data/raw/inegi/denue/2025/"
        "denue_00_46591-46911_0525_csv/"
        "conjunto_de_datos/denue_inegi_46591-46911_.csv"
    )

    assert denue_code_from_path(path) == "46591-46911"


def test_category_for_denue_code() -> None:
    assert category_for_denue_code("61") == "education"
    assert category_for_denue_code("62") == "health"
    assert category_for_denue_code("46111") == "retail"
    assert category_for_denue_code("48-49") == "transport"
    assert category_for_denue_code("72_1") == "food_lodging"
    assert category_for_denue_code("71") == "recreation"
    assert category_for_denue_code("93") == "government"
    assert category_for_denue_code("11") is None


def test_build_denue_service_accessibility_features() -> None:
    cells = gpd.GeoDataFrame(
        {
            "cell_id": ["cell_a", "cell_b", "cell_c"],
            "city_id": ["city_1", "city_1", "city_2"],
        },
        geometry=[
            box(-102.30, 21.80, -102.29, 21.81),
            box(-102.29, 21.80, -102.28, 21.81),
            box(-103.00, 22.00, -102.99, 22.01),
        ],
        crs="EPSG:4326",
    )

    points = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "denue_service_category": ["education", "health"],
                "id": ["1", "2"],
            }
        ),
        geometry=[
            Point(-102.295, 21.805),
            Point(-102.285, 21.805),
        ],
        crs="EPSG:4326",
    )

    features = build_denue_service_accessibility_features(
        cells=cells,
        service_points=points,
        categories=["education", "health"],
    )

    assert len(features) == 3
    assert features["cell_id"].tolist() == ["cell_a", "cell_b", "cell_c"]
    assert features["denue_service_education_count"].tolist() == [1, 0, 0]
    assert features["denue_service_health_count"].tolist() == [0, 1, 0]
    assert features["denue_service_total_count"].tolist() == [1, 1, 0]

    city_2 = features[features["cell_id"] == "cell_c"].iloc[0]
    assert pd.isna(city_2["denue_service_distance_to_nearest_any_m"])
