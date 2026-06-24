import geopandas as gpd
from shapely.geometry import LineString, Polygon

from urban_growth.features.road_features import (
    WGS84_CRS,
    add_road_features_for_city,
    classify_highway_type,
    normalize_highway_type,
    road_features_output_path,
    roads_output_path,
    save_road_features,
)


def build_test_cells() -> gpd.GeoDataFrame:
    cells = []

    polygons = [
        Polygon(
            [
                (-102.300, 21.800),
                (-102.295, 21.800),
                (-102.295, 21.805),
                (-102.300, 21.805),
                (-102.300, 21.800),
            ]
        ),
        Polygon(
            [
                (-102.295, 21.800),
                (-102.290, 21.800),
                (-102.290, 21.805),
                (-102.295, 21.805),
                (-102.295, 21.800),
            ]
        ),
    ]

    for index, polygon in enumerate(polygons, start=1):
        cells.append(
            {
                "cell_id": f"mx_test_city_500m_r00000_c{index:05d}",
                "city_id": "mx_test_city",
                "grid_size_m": 500,
                "cell_area_km2": 0.25,
                "geometry": polygon,
            }
        )

    return gpd.GeoDataFrame(cells, crs=WGS84_CRS)


def build_test_roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "road_id": "mx_test_city_road_000001",
                "city_id": "mx_test_city",
                "highway_type": "primary",
                "road_class": "major",
                "geometry": LineString(
                    [
                        (-102.301, 21.802),
                        (-102.289, 21.802),
                    ]
                ),
            },
            {
                "road_id": "mx_test_city_road_000002",
                "city_id": "mx_test_city",
                "highway_type": "residential",
                "road_class": "minor",
                "geometry": LineString(
                    [
                        (-102.292, 21.799),
                        (-102.292, 21.806),
                    ]
                ),
            },
        ],
        crs=WGS84_CRS,
    )


def test_road_features_output_path() -> None:
    path = road_features_output_path(
        country_code="MX",
        grid_size_m=500,
        dataset_label="priorities_1_2",
    )

    assert str(path) == ("data/features/roads/mx/500m/mx_road_features_priorities_1_2_500m.parquet")


def test_roads_output_path() -> None:
    path = roads_output_path(
        city_id="mx_aguascalientes",
        country_code="MX",
    )

    assert str(path) == "data/external/roads/mx/mx_aguascalientes_roads.geojson"


def test_normalize_highway_type() -> None:
    assert normalize_highway_type("primary") == "primary"
    assert normalize_highway_type(["residential", "service"]) == "residential"
    assert normalize_highway_type(None) is None


def test_classify_highway_type() -> None:
    assert classify_highway_type("motorway") == "major"
    assert classify_highway_type("secondary") == "medium"
    assert classify_highway_type("residential") == "minor"
    assert classify_highway_type("service") == "service"
    assert classify_highway_type("footway") is None


def test_add_road_features_for_city() -> None:
    cells = build_test_cells()
    roads = build_test_roads()

    result = add_road_features_for_city(cells, roads)

    expected_columns = {
        "distance_to_nearest_road_m",
        "distance_to_nearest_road_km",
        "nearest_road_type",
        "nearest_road_class",
        "distance_to_major_road_m",
        "distance_to_medium_road_m",
        "distance_to_minor_road_m",
        "distance_to_service_road_m",
        "road_density_all_m_per_km2",
        "road_density_major_m_per_km2",
        "road_density_medium_m_per_km2",
        "road_density_minor_m_per_km2",
        "road_density_service_m_per_km2",
        "road_count_city",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == 2
    assert result["distance_to_nearest_road_m"].notna().all()
    assert (result["road_density_all_m_per_km2"] > 0).any()
    assert result["road_count_city"].iloc[0] == 2


def test_save_road_features(tmp_path) -> None:
    cells = build_test_cells()
    roads = build_test_roads()
    dataset = add_road_features_for_city(cells, roads)

    output_path = tmp_path / "road_features.parquet"
    saved_path = save_road_features(dataset, output_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".parquet"
