import pandas as pd
import pytest

from urban_growth.features.temporal_land_cover import (
    DYNAMIC_WORLD_CLASS_BANDS,
    DYNAMIC_WORLD_COLLECTION,
    DYNAMIC_WORLD_LABELS,
    build_years,
    road_features_input_path,
    save_temporal_land_cover_features,
    temporal_land_cover_output_path,
)


def test_dynamic_world_constants() -> None:
    assert DYNAMIC_WORLD_COLLECTION == "GOOGLE/DYNAMICWORLD/V1"
    assert "built" in DYNAMIC_WORLD_CLASS_BANDS
    assert DYNAMIC_WORLD_LABELS["built"] == 6


def test_build_years() -> None:
    assert build_years(2016, 2025) == [
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
    ]


def test_build_years_invalid_range() -> None:
    with pytest.raises(ValueError):
        build_years(2025, 2016)


def test_temporal_land_cover_output_path() -> None:
    path = temporal_land_cover_output_path(
        country_code="MX",
        grid_size_m=500,
        start_year=2016,
        end_year=2025,
        dataset_label="priorities_1_2",
    )

    assert str(path) == (
        "data/features/land_cover/mx/500m/"
        "mx_land_cover_dynamic_world_priorities_1_2_2016_2025_500m.parquet"
    )


def test_road_features_input_path() -> None:
    path = road_features_input_path(
        country_code="MX",
        grid_size_m=500,
        dataset_label="priorities_1_2",
    )

    assert str(path) == ("data/features/roads/mx/500m/mx_road_features_priorities_1_2_500m.parquet")


def test_save_temporal_land_cover_features(tmp_path) -> None:
    dataset = pd.DataFrame(
        [
            {
                "cell_id": "mx_test_city_500m_r00000_c00000",
                "city_id": "mx_test_city",
                "year": 2020,
                "built_probability_mean": 0.42,
            }
        ]
    )

    output_path = tmp_path / "temporal_land_cover.parquet"
    saved_path = save_temporal_land_cover_features(dataset, output_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".parquet"
