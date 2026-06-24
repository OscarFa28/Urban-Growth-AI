import geopandas as gpd
from shapely.geometry import Polygon

from urban_growth.data.dataset_builder import (
    WGS84_CRS,
    base_dataset_output_path,
    build_base_dataset,
    enrich_grid_with_city_metadata,
    grid_input_path,
    save_base_dataset,
)


def build_test_grid() -> gpd.GeoDataFrame:
    polygon = Polygon(
        [
            (-102.30, 21.80),
            (-102.29, 21.80),
            (-102.29, 21.81),
            (-102.30, 21.81),
            (-102.30, 21.80),
        ]
    )

    return gpd.GeoDataFrame(
        [
            {
                "city_id": "mx_test_city",
                "grid_size_m": 500,
                "cell_id": "mx_test_city_500m_r00000_c00000",
                "cell_index": 1,
                "row": 0,
                "col": 0,
                "nominal_area_m2": 250000.0,
                "area_m2": 250000.0,
                "coverage_ratio": 1.0,
                "geometry": polygon,
            }
        ],
        crs=WGS84_CRS,
    )


def build_test_registry() -> dict:
    return {
        "country": "Mexico",
        "country_code": "MX",
        "cities": [
            {
                "id": "mx_test_city",
                "name": "Test City",
                "state": "Test State",
                "spatial_unit_type": "municipality",
                "municipality_id": "mx_test_city",
                "municipality_name": "Test City",
                "metro_area_id": "mx_test_metro",
                "metro_area_name": "Test Metro Area",
                "size_category": "city",
                "density_category": "medium",
                "behavior_category": ["industrial", "administrative"],
                "priority": 1,
            }
        ],
    }


def test_grid_input_path() -> None:
    path = grid_input_path(
        city_id="mx_aguascalientes",
        country_code="MX",
        grid_size_m=500,
    )

    assert str(path) == ("data/interim/grids/mx/500m/mx_aguascalientes_grid_500m.geojson")


def test_base_dataset_output_path() -> None:
    path = base_dataset_output_path(
        country_code="MX",
        grid_size_m=500,
    )

    assert str(path) == "data/features/base/mx/500m/mx_base_cells_500m.parquet"


def test_enrich_grid_with_city_metadata() -> None:
    grid = build_test_grid()
    registry = build_test_registry()
    city = registry["cities"][0]

    enriched = enrich_grid_with_city_metadata(
        grid=grid,
        city=city,
        country=registry["country"],
        country_code=registry["country_code"],
    )

    assert enriched["city_name"].iloc[0] == "Test City"
    assert enriched["country_code"].iloc[0] == "MX"
    assert enriched["spatial_unit_type"].iloc[0] == "municipality"
    assert enriched["metro_area_id"].iloc[0] == "mx_test_metro"
    assert enriched["behavior_categories"].iloc[0] == "industrial|administrative"
    assert enriched["primary_behavior_category"].iloc[0] == "industrial"


def test_build_base_dataset(tmp_path) -> None:
    registry = build_test_registry()
    city = registry["cities"][0]

    grid_dir = tmp_path / "grids"
    path = grid_input_path(
        city_id=city["id"],
        country_code=registry["country_code"],
        grid_size_m=500,
        grid_dir=grid_dir,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    build_test_grid().to_file(path, driver="GeoJSON")

    dataset = build_base_dataset(
        registry=registry,
        cities=[city],
        grid_size_m=500,
        grid_dir=grid_dir,
    )

    assert len(dataset) == 1
    assert dataset.crs.to_string() == WGS84_CRS
    assert dataset["cell_id"].iloc[0] == "mx_test_city_500m_r00000_c00000"
    assert dataset["city_id"].iloc[0] == "mx_test_city"
    assert dataset["grid_size_m"].iloc[0] == 500


def test_save_base_dataset(tmp_path) -> None:
    dataset = build_test_grid()
    output_path = tmp_path / "base_dataset.parquet"

    saved_path = save_base_dataset(dataset, output_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".parquet"


def test_base_dataset_output_path_with_label() -> None:
    path = base_dataset_output_path(
        country_code="MX",
        grid_size_m=500,
        dataset_label="priorities_1_2",
    )

    assert str(path) == ("data/features/base/mx/500m/mx_base_cells_priorities_1_2_500m.parquet")
