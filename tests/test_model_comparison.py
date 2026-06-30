import pandas as pd

from urban_growth.scoring.model_comparison import (
    choose_best_model,
    summarize_model_overall,
    summarize_tiers,
    summarize_top_percentiles,
)


def sample_scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "year": [2024, 2024, 2024, 2024],
            "urbanized_next_year": [1, 1, 0, 0],
            "urban_growth_potential_percentile": [0.99, 0.96, 0.90, 0.10],
            "urban_growth_potential_tier": ["very_high", "high", "medium", "very_low"],
            "urban_growth_potential_score": [95.0, 90.0, 60.0, 10.0],
            "urban_growth_probability": [0.9, 0.8, 0.4, 0.1],
        }
    )


def test_summarize_top_percentiles():
    scored = sample_scored_frame()

    rows = summarize_top_percentiles(
        scored,
        model_name="test_model",
        percentile_thresholds=[0.95],
    )

    assert rows[0]["model"] == "test_model"
    assert rows[0]["bucket"] == "top_5pct"
    assert rows[0]["rows"] == 2
    assert rows[0]["positives"] == 2
    assert rows[0]["positive_rate"] == 1.0
    assert rows[0]["recall_of_all_positives"] == 1.0


def test_summarize_tiers():
    scored = sample_scored_frame()

    rows = summarize_tiers(scored, model_name="test_model")
    summary = pd.DataFrame(rows)

    very_high = summary[summary["urban_growth_potential_tier"] == "very_high"].iloc[0]

    assert very_high["model"] == "test_model"
    assert very_high["rows"] == 1
    assert very_high["positives"] == 1
    assert very_high["positive_rate"] == 1.0


def test_summarize_model_overall():
    scored = sample_scored_frame()

    summary = summarize_model_overall(scored, model_name="test_model")

    assert summary["model"] == "test_model"
    assert summary["rows"] == 4
    assert summary["cells"] == 4
    assert summary["years"] == [2024]
    assert summary["positives"] == 2
    assert summary["positive_rate"] == 0.5


def test_choose_best_model():
    top_percentile_summary = pd.DataFrame(
        {
            "model": ["a", "b"],
            "bucket": ["top_5pct", "top_5pct"],
            "recall_of_all_positives": [0.8, 0.9],
            "positive_rate": [0.5, 0.4],
            "mean_probability": [0.7, 0.6],
        }
    )

    best = choose_best_model(top_percentile_summary, bucket="top_5pct")

    assert best["selected_model"] == "b"
    assert best["selection_bucket"] == "top_5pct"
