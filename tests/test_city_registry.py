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
