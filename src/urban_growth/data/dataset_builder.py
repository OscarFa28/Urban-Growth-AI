from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

WGS84_CRS = "EPSG:4326"


def grid_input_path(
    city_id: str,
    country_code: str,
    grid_size_m: int,
    grid_dir: str | Path = "data/interim/grids",
) -> Path:
    """Build the standard input path for a generated city grid."""
    return (
        Path(grid_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{city_id}_grid_{grid_size_m}m.geojson"
    )


def base_dataset_output_path(
    country_code: str,
    grid_size_m: int,
    output_dir: str | Path = "data/features/base",
    dataset_label: str | None = None,
) -> Path:
    """Build the standard output path for the base cells dataset."""
    label = f"_{dataset_label}" if dataset_label else ""

    return (
        Path(output_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_base_cells{label}_{grid_size_m}m.parquet"
    )


def read_grid(path: str | Path) -> gpd.GeoDataFrame:
    """Read a generated city grid."""
    grid_path = Path(path)

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    grid = gpd.read_file(grid_path)

    if grid.empty:
        raise ValueError(f"Grid file is empty: {grid_path}")

    if grid.crs is None:
        grid = grid.set_crs(WGS84_CRS)

    return grid.to_crs(WGS84_CRS)


def build_city_lookup(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a city lookup dictionary indexed by city_id."""
    return {city["id"]: city for city in registry["cities"]}


def _serialize_behavior_categories(city: dict[str, Any]) -> str:
    """Serialize behavior categories into a stable string."""
    categories = city.get("behavior_category", [])

    if isinstance(categories, list):
        return "|".join(str(category) for category in categories)

    if categories is None:
        return ""

    return str(categories)


def _get_primary_behavior_category(city: dict[str, Any]) -> str | None:
    """Get the first behavior category for simple modeling features."""
    categories = city.get("behavior_category", [])

    if isinstance(categories, list) and categories:
        return str(categories[0])

    if isinstance(categories, str):
        return categories

    return None


def enrich_grid_with_city_metadata(
    grid: gpd.GeoDataFrame,
    city: dict[str, Any],
    country: str,
    country_code: str,
) -> gpd.GeoDataFrame:
    """Attach registry metadata to a city grid."""
    enriched = grid.copy()

    enriched["country"] = country
    enriched["country_code"] = country_code
    enriched["state"] = city.get("state")
    enriched["city_name"] = city.get("name")

    enriched["spatial_unit_id"] = city.get("id")
    enriched["spatial_unit_type"] = city.get("spatial_unit_type")
    enriched["municipality_id"] = city.get("municipality_id")
    enriched["municipality_name"] = city.get("municipality_name")
    enriched["metro_area_id"] = city.get("metro_area_id")
    enriched["metro_area_name"] = city.get("metro_area_name")

    enriched["size_category"] = city.get("size_category")
    enriched["density_category"] = city.get("density_category")
    enriched["behavior_categories"] = _serialize_behavior_categories(city)
    enriched["primary_behavior_category"] = _get_primary_behavior_category(city)

    return enriched


def build_base_dataset(
    registry: dict[str, Any],
    cities: list[dict[str, Any]],
    grid_size_m: int = 500,
    grid_dir: str | Path = "data/interim/grids",
) -> gpd.GeoDataFrame:
    """Build the base geospatial dataset from generated city grids."""
    country = registry["country"]
    country_code = registry["country_code"]

    datasets = []

    for city in cities:
        city_id = city["id"]
        path = grid_input_path(
            city_id=city_id,
            country_code=country_code,
            grid_size_m=grid_size_m,
            grid_dir=grid_dir,
        )

        grid = read_grid(path)
        enriched_grid = enrich_grid_with_city_metadata(
            grid=grid,
            city=city,
            country=country,
            country_code=country_code,
        )

        datasets.append(enriched_grid)

    if not datasets:
        raise ValueError("No city grids were loaded. Cannot build base dataset.")

    dataset = gpd.GeoDataFrame(
        pd.concat(datasets, ignore_index=True),
        crs=WGS84_CRS,
    )

    ordered_columns = [
        "cell_id",
        "cell_index",
        "city_id",
        "city_name",
        "country",
        "country_code",
        "state",
        "spatial_unit_id",
        "spatial_unit_type",
        "municipality_id",
        "municipality_name",
        "metro_area_id",
        "metro_area_name",
        "size_category",
        "density_category",
        "behavior_categories",
        "primary_behavior_category",
        "grid_size_m",
        "row",
        "col",
        "nominal_area_m2",
        "area_m2",
        "coverage_ratio",
        "geometry",
    ]

    return dataset[ordered_columns]


def save_base_dataset(dataset: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Save the base dataset as GeoParquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(path, index=False)

    return path
