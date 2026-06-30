"""Training helpers for urban growth models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "urbanized_next_year"

LEAKAGE_AND_TARGET_COLUMNS = {
    "target_year",
    "built_probability_next_year",
    "built_probability_change_next_year",
    "built_label_frequency_next_year",
    "built_label_frequency_change_next_year",
    "is_urban_next_year",
    "has_next_year_target",
    "urbanized_next_year",
    "strong_urban_growth_next_year",
    "urban_state_transition",
}

IDENTIFIER_COLUMNS = {
    "cell_id",
    "cell_index",
    "city_name",
    "country",
    "country_code",
    "state",
    "spatial_unit_id",
    "municipality_id",
    "municipality_name",
    "municipality_cvegeo",
    "metro_area_id",
    "metro_area_name",
    "geometry",
}


@dataclass(frozen=True)
class TemporalSplit:
    """Temporal train/validation/test split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared modeling data."""

    features: pd.DataFrame
    target: pd.Series
    feature_columns: list[str]
    frame: pd.DataFrame


def read_modeling_dataset(path: Path) -> gpd.GeoDataFrame:
    """Read modeling dataset."""
    return gpd.read_parquet(path)


def select_feature_columns(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> list[str]:
    """Select model feature columns while excluding leakage and identifiers."""
    excluded = set(IDENTIFIER_COLUMNS)
    excluded.update(LEAKAGE_AND_TARGET_COLUMNS)
    excluded.add(target_column)

    candidates = frame.drop(columns=list(excluded), errors="ignore")
    selected = candidates.select_dtypes(
        include=["number", "bool", "object", "category", "string"]
    ).columns.tolist()

    return selected


def prepare_modeling_data(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    candidate_only: bool = True,
) -> PreparedDataset:
    """Prepare features and target for binary classification."""
    prepared = frame.copy()

    if candidate_only and "is_urban" in prepared.columns:
        prepared = prepared[~prepared["is_urban"].fillna(False).astype(bool)].copy()

    prepared = prepared.dropna(subset=[target_column]).copy()
    prepared[target_column] = prepared[target_column].astype(int)

    feature_columns = select_feature_columns(prepared, target_column=target_column)
    features = prepared[feature_columns].copy()

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns
    categorical_columns = features.select_dtypes(include=["object", "category", "string"]).columns

    features[numeric_columns] = features[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    features[numeric_columns] = features[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in categorical_columns:
        features[column] = features[column].astype("string").fillna("missing")
        features[column] = features[column].astype(str)

    target = prepared[target_column].copy()

    return PreparedDataset(
        features=features,
        target=target,
        feature_columns=feature_columns,
        frame=prepared,
    )


def temporal_train_validation_test_split(
    frame: pd.DataFrame,
    train_start_year: int = 2016,
    train_end_year: int = 2022,
    validation_year: int = 2023,
    test_year: int = 2024,
) -> TemporalSplit:
    """Split frame using time-aware years."""
    train = frame[frame["year"].between(train_start_year, train_end_year)].copy()
    validation = frame[frame["year"] == validation_year].copy()
    test = frame[frame["year"] == test_year].copy()

    if train.empty:
        msg = "Training split is empty."
        raise ValueError(msg)

    if validation.empty:
        msg = "Validation split is empty."
        raise ValueError(msg)

    if test.empty:
        msg = "Test split is empty."
        raise ValueError(msg)

    return TemporalSplit(train=train, validation=validation, test=test)


def predict_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict positive-class scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return 1 / (1 + np.exp(-scores))

    msg = "Model must implement predict_proba or decision_function."
    raise TypeError(msg)


def find_best_f1_threshold(
    target: pd.Series,
    scores: np.ndarray,
) -> float:
    """Find threshold that maximizes F1 on a validation split."""
    thresholds = np.unique(scores)

    if len(thresholds) == 0:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = scores >= threshold
        score = f1_score(target, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def evaluate_binary_classifier(
    target: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate binary classifier scores."""
    predictions = scores >= threshold

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(
        target,
        predictions,
        labels=labels,
    ).ravel()

    has_two_classes = target.nunique() == 2

    return {
        "rows": int(len(target)),
        "positives": int(target.sum()),
        "positive_rate": float(target.mean()),
        "threshold": float(threshold),
        "predicted_positives": int(predictions.sum()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": (float(roc_auc_score(target, scores)) if has_two_classes else None),
        "average_precision": (
            float(average_precision_score(target, scores)) if has_two_classes else None
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def summarize_split(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    """Summarize a modeling split."""
    return {
        "rows": int(len(frame)),
        "years": sorted(int(year) for year in frame["year"].unique()),
        "cells": int(frame["cell_id"].nunique()) if "cell_id" in frame else None,
        "positives": int(frame[target_column].sum()),
        "positive_rate": float(frame[target_column].mean()),
    }
