"""Advanced model definitions."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def get_numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric feature columns."""
    return features.select_dtypes(include=["number", "bool"]).columns.tolist()


def get_categorical_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return categorical feature columns."""
    return features.select_dtypes(
        include=["object", "category", "string"],
    ).columns.tolist()


def create_tree_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for tree-based models."""
    numeric_features = get_numeric_feature_columns(features)
    categorical_features = get_categorical_feature_columns(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ],
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ],
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def create_hist_gradient_boosting_model(
    features: pd.DataFrame,
    random_state: int = 42,
    verbose: int = 0,
) -> Pipeline:
    """Create HistGradientBoostingClassifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_tree_preprocessor(features)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=random_state,
                    verbose=verbose,
                ),
            ),
        ],
    )


def create_random_forest_model(
    features: pd.DataFrame,
    random_state: int = 42,
    verbose: int = 0,
) -> Pipeline:
    """Create RandomForestClassifier pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_tree_preprocessor(features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    verbose=verbose,
                ),
            ),
        ],
    )


def create_advanced_model(
    model_name: str,
    features: pd.DataFrame,
    random_state: int = 42,
    verbose: int = 0,
) -> Pipeline:
    """Create advanced model by name."""
    if model_name == "hist_gradient_boosting":
        return create_hist_gradient_boosting_model(
            features,
            random_state=random_state,
            verbose=verbose,
        )

    if model_name == "random_forest":
        return create_random_forest_model(
            features,
            random_state=random_state,
            verbose=verbose,
        )

    msg = f"Unsupported model_name: {model_name}"
    raise ValueError(msg)
