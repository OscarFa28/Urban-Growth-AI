import geopandas as gpd
from shapely.geometry import Polygon

from urban_growth.geo.grids import (
    WGS84_CRS,
    boundary_input_path,
    generate_grid,
    generate_multi_resolution_grids,
    grid_output_path,
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
                "state": "Test State",
                "country": "Mexico",
                "geometry": polygon,
            }
        ],
        crs=WGS84_CRS,
    )


def test_generate_grid_500m() -> None:
    boundary = build_test_boundary()

    grid = generate_grid(boundary, cell_size_m=500)

    assert not grid.empty
    assert grid.crs.to_string() == WGS84_CRS
    assert set(grid["grid_size_m"].unique()) == {500}
    assert grid["cell_id"].str.startswith("mx_test_city_500m_").all()
    assert (grid["area_m2"] > 0).all()
    assert (grid["coverage_ratio"] > 0).all()
    assert (grid["coverage_ratio"] <= 1).all()


def test_generate_multi_resolution_grids() -> None:
    boundary = build_test_boundary()

    grids = generate_multi_resolution_grids(
        boundary=boundary,
        grid_sizes_m=[200, 500, 1000],
    )

    assert set(grids.keys()) == {200, 500, 1000}
    assert all(not grid.empty for grid in grids.values())
    assert set(grids[200]["grid_size_m"].unique()) == {200}
    assert set(grids[500]["grid_size_m"].unique()) == {500}
    assert set(grids[1000]["grid_size_m"].unique()) == {1000}


def test_grid_output_path() -> None:
    path = grid_output_path(
        city_id="mx_aguascalientes",
        country_code="MX",
        grid_size_m=500,
        output_dir="data/interim/grids",
    )

    assert str(path) == ("data/interim/grids/mx/500m/mx_aguascalientes_grid_500m.geojson")


def test_boundary_input_path() -> None:
    path = boundary_input_path(
        city_id="mx_aguascalientes",
        country_code="MX",
        boundary_dir="data/external/boundaries",
    )

    assert str(path) == "data/external/boundaries/mx/mx_aguascalientes.geojson"
