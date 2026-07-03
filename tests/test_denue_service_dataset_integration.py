import pandas as pd

from urban_growth.modeling.dataset import _prepare_denue_service_features


def test_prepare_denue_service_features_selects_only_service_columns() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["cell_a", "cell_b"],
            "denue_service_total_count": [10, 20],
            "denue_service_distance_to_nearest_any_m": [100.0, 200.0],
            "economic_business_count_total": [99, 88],
            "geometry": [None, None],
        }
    )

    prepared = _prepare_denue_service_features(frame)

    assert prepared.columns.tolist() == [
        "cell_id",
        "denue_service_total_count",
        "denue_service_distance_to_nearest_any_m",
    ]
    assert prepared["denue_service_total_count"].tolist() == [10, 20]
