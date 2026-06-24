from pathlib import Path
from typing import Any

import yaml


def load_city_registry(path: str | Path) -> dict[str, Any]:
    """Load a city registry YAML file."""
    registry_path = Path(path)

    if not registry_path.exists():
        raise FileNotFoundError(f"City registry file not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("City registry must be a YAML dictionary.")

    if "country" not in data:
        raise ValueError("City registry must contain a 'country' field.")

    if "cities" not in data:
        raise ValueError("City registry must contain a 'cities' list.")

    if not isinstance(data["cities"], list):
        raise ValueError("'cities' must be a list.")

    return data


def get_priority_cities(registry: dict[str, Any], priority: int = 1) -> list[dict[str, Any]]:
    """Return cities matching the selected priority."""
    return [city for city in registry["cities"] if city.get("priority") == priority]


def get_cities_by_metro_area(
    registry: dict[str, Any],
    metro_area_id: str,
) -> list[dict[str, Any]]:
    """Return cities belonging to a given metropolitan area."""
    return [
        city
        for city in registry["cities"]
        if city.get("metro_area_id") == metro_area_id
    ]


def get_spatial_units_by_type(
    registry: dict[str, Any],
    spatial_unit_type: str,
) -> list[dict[str, Any]]:
    """Return cities/spatial units matching a given spatial unit type."""
    return [
        city
        for city in registry["cities"]
        if city.get("spatial_unit_type") == spatial_unit_type
    ]