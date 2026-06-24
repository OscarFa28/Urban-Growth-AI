from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiPolygon, Polygon

WGS84_CRS = "EPSG:4326"
DEFAULT_MIN_BOUNDARY_AREA_KM2 = 1.0


def get_city_osm_queries(city: dict[str, Any], country: str) -> list[str]:
    """Build the OSM geocoding queries for a city."""
    if city.get("osm_queries"):
        return [str(query) for query in city["osm_queries"]]

    if city.get("osm_query"):
        return [str(city["osm_query"])]

    return [f'{city["name"]}, {city["state"]}, {country}']


def get_city_osm_query(city: dict[str, Any], country: str) -> str:
    """Build the primary OSM geocoding query for a city.

    Kept for backwards compatibility with tests and simple callers.
    """
    return get_city_osm_queries(city, country)[0]


def _filter_polygonal_geometries(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only polygonal geometries."""
    polygon_mask = boundary.geometry.apply(
        lambda geom: isinstance(geom, Polygon | MultiPolygon)
    )
    return boundary.loc[polygon_mask].copy()


def calculate_area_km2(boundary: gpd.GeoDataFrame) -> float:
    """Calculate the area of a boundary in square kilometers."""
    metric_crs = boundary.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError("Could not estimate a metric CRS to calculate area.")

    return float(boundary.to_crs(metric_crs).geometry.area.sum() / 1_000_000)


def _get_min_boundary_area_km2(city: dict[str, Any]) -> float:
    """Get the minimum acceptable boundary area for a city."""
    return float(city.get("min_boundary_area_km2", DEFAULT_MIN_BOUNDARY_AREA_KM2))


def fetch_city_boundary(city: dict[str, Any], country: str) -> gpd.GeoDataFrame:
    """Fetch a city boundary from OpenStreetMap using OSMnx."""
    queries = get_city_osm_queries(city, country)
    min_area_km2 = _get_min_boundary_area_km2(city)
    errors = []

    for query in queries:
        try:
            boundary = ox.geocode_to_gdf(query)
        except Exception as exc:
            errors.append(f"{query}: {exc}")
            continue

        if boundary.empty:
            errors.append(f"{query}: empty result")
            continue

        boundary = boundary.to_crs(WGS84_CRS)
        boundary = _filter_polygonal_geometries(boundary)

        if boundary.empty:
            errors.append(f"{query}: no polygonal geometry")
            continue

        area_km2 = calculate_area_km2(boundary)

        if area_km2 < min_area_km2:
            errors.append(
                f"{query}: area too small ({area_km2:.2f} km² < {min_area_km2:.2f} km²)"
            )
            continue

        boundary["city_id"] = city["id"]
        boundary["city_name"] = city["name"]
        boundary["state"] = city["state"]
        boundary["country"] = country
        boundary["osm_query"] = query
        boundary["area_km2"] = area_km2

        return boundary[
            [
                "city_id",
                "city_name",
                "state",
                "country",
                "osm_query",
                "area_km2",
                "geometry",
            ]
        ]

    joined_errors = " | ".join(errors)
    raise ValueError(
        f"No valid polygonal boundary found for city '{city['id']}'. "
        f"Tried: {joined_errors}"
    )


def save_city_boundary(boundary: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Save a city boundary as GeoJSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    boundary.to_file(path, driver="GeoJSON")

    return path


def boundary_output_path(
    city_id: str,
    country_code: str,
    output_dir: str | Path = "data/external/boundaries",
) -> Path:
    """Build the standard output path for a city boundary."""
    return Path(output_dir) / country_code.lower() / f"{city_id}.geojson"


def fetch_and_save_city_boundary(
    city: dict[str, Any],
    country: str,
    country_code: str,
    output_dir: str | Path = "data/external/boundaries",
) -> Path:
    """Fetch and save a city boundary."""
    boundary = fetch_city_boundary(city=city, country=country)
    output_path = boundary_output_path(
        city_id=city["id"],
        country_code=country_code,
        output_dir=output_dir,
    )
    return save_city_boundary(boundary, output_path)
