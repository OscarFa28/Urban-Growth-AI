import pandas as pd
import pytest

from urban_growth.labels.urbanization import (
    DEFAULT_STRONG_GROWTH_THRESHOLD,
    DEFAULT_URBAN_THRESHOLD,
    add_urbanization_labels,
    land_cover_input_path,
    save_urbanization_labels,
    urbanization_labels_output_path,
)


def build_test_land_cover() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": "cell_1",
                "city_id": "mx_test",
                "year": 2020,
                "built_probability_mean": 0.10,
                "built_label_frequency": 0.00,
            },
            {
                "cell_id": "cell_1",
                "city_id": "mx_test",
                "year": 2021,
                "built_probability_mean": 0.55,
                "built_label_frequency": 0.75,
            },
            {
                "cell_id": "cell_2",
                "city_id": "mx_test",
                "year": 2020,
                "built_probability_mean": 0.80,
                "built_label_frequency": 0.90,
            },
            {
                "cell_id": "cell_2",
                "city_id": "mx_test",
                "year": 2021,
                "built_probability_mean": 0.82,
                "built_label_frequency": 0.95,
            },
        ]
    )


def test_land_cover_input_path() -> None:
    path = land_cover_input_path(
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


def test_urbanization_labels_output_path() -> None:
    path = urbanization_labels_output_path(
        country_code="MX",
        grid_size_m=500,
        start_year=2016,
        end_year=2025,
        dataset_label="priorities_1_2",
    )

    assert str(path) == (
        "data/labels/urbanization/mx/500m/"
        "mx_urbanization_labels_dynamic_world_priorities_1_2_2016_2025_500m.parquet"
    )


def test_add_urbanization_labels() -> None:
    dataset = build_test_land_cover()

    labels = add_urbanization_labels(
        dataset,
        urban_threshold=DEFAULT_URBAN_THRESHOLD,
        strong_growth_threshold=DEFAULT_STRONG_GROWTH_THRESHOLD,
    )

    assert len(labels) == 2
    assert labels["target_year"].tolist() == [2021, 2021]

    cell_1 = labels.loc[labels["cell_id"] == "cell_1"].iloc[0]
    cell_2 = labels.loc[labels["cell_id"] == "cell_2"].iloc[0]

    assert not bool(cell_1["is_urban"])
    assert bool(cell_1["is_urban_next_year"])
    assert cell_1["urbanized_next_year"] == 1
    assert cell_1["strong_urban_growth_next_year"] == 1
    assert cell_1["urban_state_transition"] == "non_urban_to_urban"

    assert bool(cell_2["is_urban"])
    assert bool(cell_2["is_urban_next_year"])
    assert cell_2["urbanized_next_year"] == 0
    assert cell_2["urban_state_transition"] == "urban_stable"


def test_add_urbanization_labels_keep_incomplete_targets() -> None:
    dataset = build_test_land_cover()

    labels = add_urbanization_labels(
        dataset,
        drop_incomplete_targets=False,
    )

    assert len(labels) == 4
    assert labels["has_next_year_target"].sum() == 2


def test_add_urbanization_labels_missing_columns() -> None:
    dataset = pd.DataFrame(
        [
            {
                "cell_id": "cell_1",
                "year": 2020,
            }
        ]
    )

    with pytest.raises(ValueError):
        add_urbanization_labels(dataset)


def test_save_urbanization_labels(tmp_path) -> None:
    dataset = build_test_land_cover()
    labels = add_urbanization_labels(dataset)

    output_path = tmp_path / "urbanization_labels.parquet"
    saved_path = save_urbanization_labels(labels, output_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".parquet"
