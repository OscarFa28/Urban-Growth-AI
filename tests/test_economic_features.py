import geopandas as gpd
from shapely.geometry import Point, box

from urban_growth.features.economic_features import (
    build_economic_features,
    classify_denue_sector,
    classify_employee_size,
)


def test_classify_denue_sector():
    assert classify_denue_sector("461110") == "retail"
    assert classify_denue_sector("722511") == "food_lodging"
    assert classify_denue_sector("621111") == "health"
    assert classify_denue_sector("611111") == "education"
    assert classify_denue_sector("315229") == "manufacturing"
    assert classify_denue_sector("bad") == "unknown"


def test_classify_employee_size():
    assert classify_employee_size("0 a 5 personas") == "micro"
    assert classify_employee_size("11 a 30 personas") == "small"
    assert classify_employee_size("101 a 250 personas") == "medium"
    assert classify_employee_size("251 y más personas") == "large"
    assert classify_employee_size("") == "unknown"


def test_build_economic_features_counts_businesses():
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

    points = gpd.GeoDataFrame(
        {
            "id": ["1", "2", "3"],
            "codigo_act": ["461110", "722511", "621111"],
            "per_ocu": [
                "0 a 5 personas",
                "11 a 30 personas",
                "251 y más personas",
            ],
            "economic_sector": ["retail", "food_lodging", "health"],
            "employee_size_class": ["micro", "small", "large"],
        },
        geometry=[
            Point(100, 100),
            Point(200, 200),
            Point(1500, 500),
        ],
        crs="EPSG:3857",
    )

    result = build_economic_features(cells=cells, denue_points=points)

    row_a = result[result["cell_id"] == "cell_a"].iloc[0]
    row_b = result[result["cell_id"] == "cell_b"].iloc[0]

    assert row_a["economic_business_count_total"] == 2
    assert row_a["economic_retail_business_count"] == 1
    assert row_a["economic_food_lodging_business_count"] == 1
    assert row_a["economic_micro_business_count"] == 1
    assert row_a["economic_small_business_count"] == 1
    assert row_a["economic_business_density_per_km2"] == 2.0

    assert row_b["economic_business_count_total"] == 1
    assert row_b["economic_health_business_count"] == 1
    assert row_b["economic_large_business_count"] == 1
