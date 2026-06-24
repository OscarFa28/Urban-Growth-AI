import geopandas as gpd
from shapely.geometry import Polygon

from urban_growth.features.spatial_features import (
    WGS84_CRS,
    add_basic_spatial_features,
    add_spatial_features_for_city,
    save_spatial_features,
    spatial_features_output_path,
)


def build_test_boundary() -> gpd.GeoDataFrame:
    polygon = Polygon(
        [
            (-102.30, 21.80),
            (-102.28, 21.80),
            (-102.28, 21.82),
            (-102.30, 21.82),
            (-102.30, 21.80),
        ]
    )

    return gpd.GeoDataFrame(
        [
            {
                "city_id": "mx_test_city",
                "city_name": "Test City",
                "geometry": polygon,
            }
        ],
        crs=WGS84_CRS,
    )


def build_test_cells() -> gpd.GeoDataFrame:
    polygon = Polygon(
        [
            (-102.299, 21.801),
            (-102.295, 21.801),
            (-102.295, 21.805),
            (-102.299, 21.805),
            (-102.299, 21.801),
        ]
    )

    return gpd.GeoDataFrame(
        [
            {
                "cell_id": "mx_test_city_500m_r00000_c00000",
                "city_id": "mx_test_city",
                "grid_size_m": 500,
                "coverage_ratio": 1.0,
                "geometry": polygon,
            }
        ],
        crs=WGS84_CRS,
    )


def test_spatial_features_output_path() -> None:
    path = spatial_features_output_path(
        country_code="MX",
        grid_size_m=500,
        dataset_label="priorities_1_2",
    )

    assert str(path) == (
        "data/features/spatial/mx/500m/mx_spatial_features_priorities_1_2_500m.parquet"
    )


def test_add_spatial_features_for_city() -> None:
    cells = build_test_cells()
    boundary = build_test_boundary()

    result = add_spatial_features_for_city(cells, boundary)

    expected_columns = {
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
    }

    assert expected_columns.issubset(result.columns)
    assert result["distance_to_city_center_m"].iloc[0] >= 0
    assert result["distance_to_boundary_m"].iloc[0] >= 0
    assert result["cell_area_km2"].iloc[0] > 0
    assert result["cell_centroid_lat"].iloc[0] != 0
    assert result["cell_centroid_lon"].iloc[0] != 0


def test_add_basic_spatial_features(tmp_path) -> None:
    cells = build_test_cells()
    boundary = build_test_boundary()

    boundary_path = tmp_path / "mx" / "mx_test_city.geojson"
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_file(boundary_path, driver="GeoJSON")

    result = add_basic_spatial_features(
        dataset=cells,
        country_code="MX",
        boundary_dir=tmp_path,
    )

    assert len(result) == 1
    assert "distance_to_city_center_m" in result.columns
    assert "distance_to_boundary_m" in result.columns


def test_save_spatial_features(tmp_path) -> None:
    cells = build_test_cells()
    boundary = build_test_boundary()
    dataset = add_spatial_features_for_city(cells, boundary)

    output_path = tmp_path / "spatial_features.parquet"
    saved_path = save_spatial_features(dataset, output_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".parquet"
