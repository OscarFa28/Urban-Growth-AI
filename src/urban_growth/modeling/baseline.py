"""Baseline model definitions."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric feature columns."""
    return features.select_dtypes(include=["number", "bool"]).columns.tolist()


def get_categorical_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return categorical feature columns."""
    return features.select_dtypes(include=["object", "category", "string"]).columns.tolist()


def create_logistic_regression_baseline(
    features: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """Create a class-balanced logistic regression baseline."""
    numeric_features = get_numeric_feature_columns(features)
    categorical_features = get_categorical_feature_columns(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="liblinear",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
