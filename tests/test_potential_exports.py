import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from urban_growth.analysis.potential_exports import (
    add_lon_lat_for_csv,
    filter_potential_scores,
    select_export_columns,
    select_top_n_by_city_year,
    select_top_percentile,
    summarize_export,
)


def test_filter_potential_scores_by_year_and_city():
    frame = gpd.GeoDataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "year": [2023, 2024, 2024],
            "city_id": ["mx_a", "mx_a", "mx_b"],
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )

    result = filter_potential_scores(frame, year=2024, city_ids=["mx_a"])

    assert result["cell_id"].tolist() == ["b"]
    assert result.crs == frame.crs


def test_select_top_percentile():
    frame = gpd.GeoDataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "urban_growth_potential_percentile": [0.5, 0.95, 0.99],
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )

    result = select_top_percentile(frame, 0.95)

    assert result["cell_id"].tolist() == ["b", "c"]


def test_select_top_n_by_city_year():
    frame = gpd.GeoDataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "city_id": ["mx_a", "mx_a", "mx_b", "mx_b"],
            "year": [2024, 2024, 2024, 2024],
            "urban_growth_potential_rank_city_year": [1, 2, 1, 3],
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        crs="EPSG:4326",
    )

    result = select_top_n_by_city_year(frame, 1)

    assert result["cell_id"].tolist() == ["a", "c"]


def test_select_export_columns_and_csv_lon_lat():
    frame = gpd.GeoDataFrame(
        {
            "cell_id": ["a"],
            "year": [2024],
            "city_id": ["mx_a"],
            "urban_growth_potential_score": [90.0],
            "extra": ["drop"],
        },
        geometry=[Point(-102, 21)],
        crs="EPSG:4326",
    )

    selected = select_export_columns(frame)
    csv_frame = add_lon_lat_for_csv(selected)

    assert "extra" not in selected.columns
    assert "lon" in csv_frame.columns
    assert "lat" in csv_frame.columns
    assert "geometry" not in csv_frame.columns


def test_summarize_export():
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "year": [2024, 2024],
            "city_id": ["mx_a", "mx_b"],
            "urbanized_next_year": [1, 0],
            "urban_growth_potential_score": [90.0, 80.0],
        }
    )

    summary = summarize_export(frame)

    assert summary["rows"] == 2
    assert summary["cells"] == 2
    assert summary["years"] == [2024]
    assert summary["cities"] == 2
    assert summary["positives"] == 1
    assert summary["positive_rate"] == 0.5
    assert summary["mean_score"] == 85.0
