import pandas as pd

from urban_growth.modeling.advanced import (
    create_advanced_model,
    create_hist_gradient_boosting_model,
    create_random_forest_model,
)


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
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
        }
    )


def sample_target() -> pd.Series:
    return pd.Series([0, 0, 1, 1, 0, 1])


def test_create_hist_gradient_boosting_model_trains():
    features = sample_features()
    target = sample_target()

    model = create_hist_gradient_boosting_model(features)
    model.fit(features, target)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (6, 2)


def test_create_random_forest_model_trains():
    features = sample_features()
    target = sample_target()

    model = create_random_forest_model(features)
    model.fit(features, target)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (6, 2)


def test_create_advanced_model_by_name():
    features = sample_features()

    hist_model = create_advanced_model("hist_gradient_boosting", features)
    forest_model = create_advanced_model("random_forest", features)

    assert hist_model is not None
    assert forest_model is not None
