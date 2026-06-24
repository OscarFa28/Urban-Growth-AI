from urban_growth.data.city_registry import get_priority_cities, load_city_registry


def test_load_mexico_city_registry() -> None:
    registry = load_city_registry("configs/cities/mexico.yaml")

    assert registry["country"] == "Mexico"
    assert registry["country_code"] == "MX"
    assert len(registry["cities"]) >= 10


def test_get_priority_cities() -> None:
    registry = load_city_registry("configs/cities/mexico.yaml")
    priority_cities = get_priority_cities(registry, priority=1)

    assert len(priority_cities) == 10
    assert all(city["priority"] == 1 for city in priority_cities)


def test_city_registry_contains_spatial_unit_fields() -> None:
    registry = load_city_registry("configs/cities/mexico.yaml")

    required_fields = {
        "spatial_unit_type",
        "municipality_id",
        "municipality_name",
        "metro_area_id",
        "metro_area_name",
    }

    for city in registry["cities"]:
        assert required_fields.issubset(city.keys())


def test_cdmx_is_not_marked_as_municipality() -> None:
    registry = load_city_registry("configs/cities/mexico.yaml")
    cdmx = next(city for city in registry["cities"] if city["id"] == "mx_cdmx")

    assert cdmx["spatial_unit_type"] == "federal_entity"
    assert cdmx["municipality_id"] is None
    assert cdmx["metro_area_id"] == "mx_zm_valle_de_mexico"


def test_get_cities_by_metro_area() -> None:
    from urban_growth.data.city_registry import get_cities_by_metro_area

    registry = load_city_registry("configs/cities/mexico.yaml")

    guadalajara_metro = get_cities_by_metro_area(
        registry,
        metro_area_id="mx_zm_guadalajara",
    )

    assert len(guadalajara_metro) >= 1
    assert guadalajara_metro[0]["id"] == "mx_guadalajara"


def test_get_spatial_units_by_type() -> None:
    from urban_growth.data.city_registry import get_spatial_units_by_type

    registry = load_city_registry("configs/cities/mexico.yaml")

    municipalities = get_spatial_units_by_type(
        registry,
        spatial_unit_type="municipality",
    )

    assert len(municipalities) >= 10
    assert all(city["spatial_unit_type"] == "municipality" for city in municipalities)
