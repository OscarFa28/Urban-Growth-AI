import pandas as pd
import pytest

from urban_growth.data.metropolitan_registry import (
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
                "source_name": "Test source",
                "source_year": "2020",
                "coverage_status": "complete",
                "notes": "",
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
                "source_name": "Test source",
                "source_year": "2020",
                "coverage_status": "complete",
                "notes": "",
            },
            {
                "country_code": "mx",
                "metro_area_id": "mx_metro_beta",
                "metro_area_name": "Metro Beta",
                "city_id": "mx_beta",
                "city_name": "Beta",
                "municipality_cvegeo": "02001",
                "municipality_name": "Beta",
                "state_name": "State B",
                "state_code": "02",
                "is_core_municipality": "true",
                "source_name": "Test source",
                "source_year": "2020",
                "coverage_status": "partial_manual_review_required",
                "notes": "",
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


def test_validate_metropolitan_municipalities_rejects_duplicate_municipalities() -> None:
    frame = pd.concat([build_example_frame(), build_example_frame().iloc[[0]]])

    with pytest.raises(ValueError, match="Duplicate rows"):
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
        "mx_metro_alpha": 2,
        "mx_metro_beta": 1,
    }


def test_list_metro_areas_returns_compact_area_table() -> None:
    frame = build_example_frame()

    metro_areas = list_metro_areas(frame)

    assert list(metro_areas["metro_area_id"]) == ["mx_metro_alpha", "mx_metro_beta"]
    assert list(metro_areas["municipality_count"]) == [2, 1]
