"""Model comparison helpers for urban growth potential scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_top_percentiles(
    scored: pd.DataFrame,
    model_name: str,
    percentile_thresholds: list[float],
    target_column: str = "urbanized_next_year",
) -> list[dict[str, Any]]:
    """Summarize ranking quality for top percentile buckets."""
    total_positives = int(scored[target_column].sum())
    rows = []

    for threshold in percentile_thresholds:
        top = scored[scored["urban_growth_potential_percentile"] >= threshold].copy()

        positives = int(top[target_column].sum())
        row = {
            "model": model_name,
            "bucket": f"top_{int(round((1 - threshold) * 100))}pct",
            "percentile_threshold": float(threshold),
            "rows": int(len(top)),
            "positives": positives,
            "positive_rate": float(top[target_column].mean()) if len(top) else 0.0,
            "recall_of_all_positives": (
                float(positives / total_positives) if total_positives else 0.0
            ),
            "mean_score": float(top["urban_growth_potential_score"].mean()) if len(top) else 0.0,
            "mean_probability": float(top["urban_growth_probability"].mean()) if len(top) else 0.0,
        }
        rows.append(row)

    return rows


def summarize_tiers(
    scored: pd.DataFrame,
    model_name: str,
    target_column: str = "urbanized_next_year",
) -> list[dict[str, Any]]:
    """Summarize ranking quality by potential tier."""
    summary = (
        scored.groupby("urban_growth_potential_tier")
        .agg(
            rows=("cell_id", "count"),
            positives=(target_column, "sum"),
            positive_rate=(target_column, "mean"),
            mean_score=("urban_growth_potential_score", "mean"),
            mean_probability=("urban_growth_probability", "mean"),
        )
        .reset_index()
    )

    summary["model"] = model_name

    ordered_columns = [
        "model",
        "urban_growth_potential_tier",
        "rows",
        "positives",
        "positive_rate",
        "mean_score",
        "mean_probability",
    ]

    return summary[ordered_columns].to_dict(orient="records")


def summarize_model_overall(
    scored: pd.DataFrame,
    model_name: str,
    target_column: str = "urbanized_next_year",
) -> dict[str, Any]:
    """Summarize overall model scoring output."""
    return {
        "model": model_name,
        "rows": int(len(scored)),
        "cells": int(scored["cell_id"].nunique()) if "cell_id" in scored else None,
        "years": sorted(int(year) for year in scored["year"].unique()),
        "positives": int(scored[target_column].sum()),
        "positive_rate": float(scored[target_column].mean()),
        "mean_score": float(scored["urban_growth_potential_score"].mean()),
        "max_score": float(scored["urban_growth_potential_score"].max()),
        "mean_probability": float(scored["urban_growth_probability"].mean()),
        "max_probability": float(scored["urban_growth_probability"].max()),
    }


def choose_best_model(
    top_percentile_summary: pd.DataFrame,
    bucket: str = "top_5pct",
) -> dict[str, Any]:
    """Choose best model by recall, then positive rate, then mean probability."""
    candidates = top_percentile_summary[top_percentile_summary["bucket"] == bucket].copy()

    if candidates.empty:
        msg = f"No rows found for bucket: {bucket}"
        raise ValueError(msg)

    candidates = candidates.sort_values(
        ["recall_of_all_positives", "positive_rate", "mean_probability"],
        ascending=[False, False, False],
    )

    best = candidates.iloc[0].to_dict()

    return {
        "selected_model": best["model"],
        "selection_bucket": bucket,
        "selection_reason": (
            "Highest recall of positives in the selected top-percentile bucket, "
            "using positive rate and mean probability as tie-breakers."
        ),
        "metrics": best,
    }
