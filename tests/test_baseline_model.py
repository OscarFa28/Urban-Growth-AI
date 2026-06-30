import pandas as pd

from urban_growth.modeling.baseline import create_logistic_regression_baseline
from urban_growth.modeling.training import (
    evaluate_binary_classifier,
    find_best_f1_threshold,
    prepare_modeling_data,
    select_feature_columns,
    temporal_train_validation_test_split,
)


def test_select_feature_columns_excludes_leakage_and_identifiers():
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "year": [2016, 2017],
            "built_probability_mean": [0.1, 0.2],
            "population_total": [100, 200],
            "city_id": ["mx_a", "mx_b"],
            "built_probability_next_year": [0.3, 0.4],
            "built_probability_change_next_year": [0.2, 0.2],
            "urbanized_next_year": [0, 1],
        }
    )

    columns = select_feature_columns(frame)

    assert "built_probability_mean" in columns
    assert "population_total" in columns
    assert "year" in columns
    assert "city_id" in columns

    assert "cell_id" not in columns
    assert "built_probability_next_year" not in columns
    assert "built_probability_change_next_year" not in columns
    assert "urbanized_next_year" not in columns


def test_prepare_modeling_data_filters_to_non_urban_candidates():
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "year": [2016, 2016, 2016],
            "is_urban": [False, True, False],
            "built_probability_mean": [0.1, 0.8, 0.2],
            "urbanized_next_year": [1, 0, 0],
        }
    )

    prepared = prepare_modeling_data(frame, candidate_only=True)

    assert len(prepared.frame) == 2
    assert prepared.target.tolist() == [1, 0]


def test_temporal_train_validation_test_split():
    frame = pd.DataFrame(
        {
            "year": [2016, 2022, 2023, 2024],
            "cell_id": ["a", "a", "a", "a"],
            "urbanized_next_year": [0, 1, 0, 1],
        }
    )

    split = temporal_train_validation_test_split(frame)

    assert split.train["year"].tolist() == [2016, 2022]
    assert split.validation["year"].tolist() == [2023]
    assert split.test["year"].tolist() == [2024]


def test_find_best_f1_threshold_and_evaluate():
    target = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.1, 0.2, 0.8, 0.9]).to_numpy()

    threshold = find_best_f1_threshold(target, scores)
    metrics = evaluate_binary_classifier(target, scores, threshold)

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["f1"] == 1.0


def test_logistic_regression_baseline_trains():
    frame = pd.DataFrame(
        {
            "year": [2016, 2017, 2018, 2019, 2020, 2021],
            "built_probability_mean": [0.1, 0.2, 0.8, 0.7, 0.15, 0.9],
            "population_total": [100, 120, 500, 450, 130, 600],
            "nearest_road_class": [
                "minor",
                "minor",
                "major",
                "major",
                "minor",
                "major",
            ],
            "urbanized_next_year": [0, 0, 1, 1, 0, 1],
        }
    )

    prepared = prepare_modeling_data(frame, candidate_only=False)
    model = create_logistic_regression_baseline(prepared.features)

    model.fit(prepared.features, prepared.target)
    probabilities = model.predict_proba(prepared.features)

    assert probabilities.shape == (6, 2)
