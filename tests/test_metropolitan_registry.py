import pandas as pd
import pytest

from urban_growth.data.metropolitan_registry import (
    DEFAULT_METROPOLITAN_REGISTRY_PATH,
    get_city_metro_coverage,
    list_metro_areas,
    read_metropolitan_municipalities,
    summarize_metropolitan_coverage,
    validate_metropolitan_municipalities,
)


def build_example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country_code": "mx",
                "metro_area_id": "mx_metro_alpha",
                "metro_area_name": "Metro Alpha",
                "city_id": "mx_alpha",
                "city_name": "Alpha",
                "municipality_cvegeo": "01001",
                "municipality_name": "Alpha",
                "state_name": "State A",
                "state_code": "01",
                "is_core_municipality": "true",
                "coverage_status": "official_2020",
                "source_name": "Test source",
                "source_year": "2020",
            },
            {
                "country_code": "mx",
                "metro_area_id": "mx_metro_alpha",
                "metro_area_name": "Metro Alpha",
                "city_id": "mx_alpha",
                "city_name": "Alpha",
                "municipality_cvegeo": "01002",
                "municipality_name": "Alpha Norte",
                "state_name": "State A",
                "state_code": "01",
                "is_core_municipality": "false",
                "coverage_status": "official_2020",
                "source_name": "Test source",
                "source_year": "2020",
            },
            {
                "country_code": "mx",
                "metro_area_id": "mx_local_beta",
                "metro_area_name": "Local Beta",
                "city_id": "mx_beta",
                "city_name": "Beta",
                "municipality_cvegeo": "02001",
                "municipality_name": "Beta",
                "state_name": "State B",
                "state_code": "02",
                "is_core_municipality": "true",
                "coverage_status": "standalone_manual_review_required",
                "source_name": "Test source",
                "source_year": "2020",
            },
        ]
    )


def test_read_metropolitan_municipalities_preserves_ids_as_strings(tmp_path) -> None:
    path = tmp_path / "metropolitan_municipalities.csv"
    build_example_frame().to_csv(path, index=False)

    frame = read_metropolitan_municipalities(path)

    assert frame["country_code"].iloc[0] == "MX"
    assert frame["municipality_cvegeo"].iloc[0] == "01001"
    assert isinstance(frame["municipality_cvegeo"].iloc[0], str)


def test_validate_metropolitan_municipalities_requires_columns() -> None:
    frame = build_example_frame().drop(columns=["municipality_name"])

    with pytest.raises(ValueError, match="Missing required metropolitan registry columns"):
        validate_metropolitan_municipalities(frame)


def test_validate_metropolitan_municipalities_rejects_empty_required_values() -> None:
    frame = build_example_frame()
    frame.loc[0, "city_id"] = ""

    with pytest.raises(ValueError, match="Column 'city_id' cannot be empty"):
        validate_metropolitan_municipalities(frame)


def test_validate_metropolitan_municipalities_rejects_duplicate_municipalities() -> None:
    frame = pd.concat([build_example_frame(), build_example_frame().iloc[[0]]])

    with pytest.raises(ValueError, match="Duplicate rows"):
        validate_metropolitan_municipalities(frame)


def test_validate_metropolitan_municipalities_rejects_invalid_cvegeo() -> None:
    frame = build_example_frame()
    frame.loc[0, "municipality_cvegeo"] = "1001"

    with pytest.raises(ValueError, match="5-digit string"):
        validate_metropolitan_municipalities(frame)


def test_validate_metropolitan_municipalities_rejects_invalid_status() -> None:
    frame = build_example_frame()
    frame.loc[0, "coverage_status"] = "draft"

    with pytest.raises(ValueError, match="Invalid coverage_status"):
        validate_metropolitan_municipalities(frame)


def test_get_city_metro_coverage_returns_all_metro_municipalities() -> None:
    frame = build_example_frame()

    coverage = get_city_metro_coverage(frame, city_id="mx_alpha")

    assert len(coverage) == 2
    assert set(coverage["municipality_cvegeo"]) == {"01001", "01002"}
    assert set(coverage["metro_area_id"]) == {"mx_metro_alpha"}


def test_summarize_metropolitan_coverage_returns_expected_counts() -> None:
    frame = build_example_frame()

    summary = summarize_metropolitan_coverage(frame)

    assert summary["metro_area_count"] == 2
    assert summary["municipality_count"] == 3
    assert summary["city_count"] == 2
    assert summary["row_count"] == 3
    assert summary["municipalities_by_metro_area"] == {
        "mx_local_beta": 1,
        "mx_metro_alpha": 2,
    }
    assert summary["manual_review_required_city_ids"] == ["mx_beta"]


def test_list_metro_areas_returns_compact_area_table() -> None:
    frame = build_example_frame()

    metro_areas = list_metro_areas(frame)

    assert list(metro_areas["metro_area_id"]) == ["mx_local_beta", "mx_metro_alpha"]
    assert list(metro_areas["municipality_count"]) == [1, 2]


def test_default_registry_has_v027_coverage_counts() -> None:
    frame = read_metropolitan_municipalities(DEFAULT_METROPOLITAN_REGISTRY_PATH)
    summary = summarize_metropolitan_coverage(frame)

    assert summary["municipality_count"] == 109
    assert summary["metro_area_count"] == 12
    assert summary["city_count"] == 12
    assert summary["municipalities_by_metro_area"]["mx_zm_valle_de_mexico"] == 63
    assert summary["municipalities_by_metro_area"]["mx_zm_monterrey"] == 16
    assert summary["municipalities_by_metro_area"]["mx_zm_guadalajara"] == 7
    assert summary["manual_review_required_city_ids"] == [
        "mx_arandas",
        "mx_san_juan_de_los_lagos",
    ]
