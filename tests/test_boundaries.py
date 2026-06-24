import geopandas as gpd
from shapely.geometry import Polygon

from urban_growth.data.boundaries import (
    WGS84_CRS,
    boundary_output_path,
    calculate_area_km2,
    get_city_osm_queries,
    get_city_osm_query,
    save_city_boundary,
)


def test_get_city_osm_query_from_config() -> None:
    city = {
        "id": "mx_aguascalientes",
        "name": "Aguascalientes",
        "state": "Aguascalientes",
        "osm_query": "Aguascalientes, Aguascalientes, Mexico",
    }

    query = get_city_osm_query(city, country="Mexico")

    assert query == "Aguascalientes, Aguascalientes, Mexico"


def test_get_city_osm_query_fallback() -> None:
    city = {
        "id": "mx_aguascalientes",
        "name": "Aguascalientes",
        "state": "Aguascalientes",
    }

    query = get_city_osm_query(city, country="Mexico")

    assert query == "Aguascalientes, Aguascalientes, Mexico"


def test_boundary_output_path() -> None:
    path = boundary_output_path(
        city_id="mx_aguascalientes",
        country_code="MX",
        output_dir="data/external/boundaries",
    )

    assert str(path) == "data/external/boundaries/mx/mx_aguascalientes.geojson"


def test_save_city_boundary(tmp_path) -> None:
    polygon = Polygon(
        [
            (-102.3, 21.8),
            (-102.2, 21.8),
            (-102.2, 21.9),
            (-102.3, 21.9),
            (-102.3, 21.8),
        ]
    )

    gdf = gpd.GeoDataFrame(
        [
            {
                "city_id": "mx_aguascalientes",
                "city_name": "Aguascalientes",
                "state": "Aguascalientes",
                "country": "Mexico",
                "osm_query": "Aguascalientes, Aguascalientes, Mexico",
                "geometry": polygon,
            }
        ],
        crs=WGS84_CRS,
    )

    output_path = tmp_path / "mx_aguascalientes.geojson"
    saved_path = save_city_boundary(gdf, output_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".geojson"


def test_get_city_osm_queries_from_list() -> None:
    city = {
        "id": "mx_aguascalientes",
        "name": "Aguascalientes",
        "state": "Aguascalientes",
        "osm_queries": [
            "Municipio de Aguascalientes, Aguascalientes, Mexico",
            "Aguascalientes, Aguascalientes, Mexico",
        ],
    }

    queries = get_city_osm_queries(city, country="Mexico")

    assert queries == [
        "Municipio de Aguascalientes, Aguascalientes, Mexico",
        "Aguascalientes, Aguascalientes, Mexico",
    ]


def test_calculate_area_km2() -> None:
    polygon = Polygon(
        [
            (-102.30, 21.80),
            (-102.28, 21.80),
            (-102.28, 21.82),
            (-102.30, 21.82),
            (-102.30, 21.80),
        ]
    )

    gdf = gpd.GeoDataFrame(
        [{"geometry": polygon}],
        crs=WGS84_CRS,
    )

    area_km2 = calculate_area_km2(gdf)

    assert area_km2 > 0
