from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_URBAN_THRESHOLD = 0.35
DEFAULT_STRONG_GROWTH_THRESHOLD = 0.20


def land_cover_input_path(
    country_code: str,
    grid_size_m: int,
    start_year: int,
    end_year: int,
    dataset_label: str,
    land_cover_dir: str | Path = "data/features/land_cover",
    source: str = "dynamic_world",
) -> Path:
    """Build the expected input path for temporal land-cover features."""
    return (
        Path(land_cover_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_land_cover_{source}_{dataset_label}_"
        f"{start_year}_{end_year}_{grid_size_m}m.parquet"
    )


def urbanization_labels_output_path(
    country_code: str,
    grid_size_m: int,
    start_year: int,
    end_year: int,
    output_dir: str | Path = "data/labels/urbanization",
    dataset_label: str | None = None,
    source: str = "dynamic_world",
) -> Path:
    """Build the standard output path for urbanization labels."""
    label = f"_{dataset_label}" if dataset_label else ""

    return (
        Path(output_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{country_code.lower()}_urbanization_labels_{source}{label}_"
        f"{start_year}_{end_year}_{grid_size_m}m.parquet"
    )


def read_temporal_land_cover(path: str | Path) -> pd.DataFrame:
    """Read temporal land-cover features from Parquet."""
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Temporal land-cover file not found: {dataset_path}")

    dataset = pd.read_parquet(dataset_path)

    if dataset.empty:
        raise ValueError(f"Temporal land-cover dataset is empty: {dataset_path}")

    return dataset


def _validate_land_cover_columns(dataset: pd.DataFrame) -> None:
    """Validate required columns for label generation."""
    required_columns = [
        "cell_id",
        "city_id",
        "year",
        "built_probability_mean",
    ]

    missing_columns = [column for column in required_columns if column not in dataset.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _build_transition_label(is_urban: bool, is_urban_next_year: bool) -> str:
    """Build a readable transition label."""
    if not is_urban and is_urban_next_year:
        return "non_urban_to_urban"

    if is_urban and is_urban_next_year:
        return "urban_stable"

    if not is_urban and not is_urban_next_year:
        return "non_urban_stable"

    return "urban_to_non_urban"


def add_urbanization_labels(
    dataset: pd.DataFrame,
    urban_threshold: float = DEFAULT_URBAN_THRESHOLD,
    strong_growth_threshold: float = DEFAULT_STRONG_GROWTH_THRESHOLD,
    drop_incomplete_targets: bool = True,
) -> pd.DataFrame:
    """Create next-year urbanization labels from temporal land-cover features."""
    _validate_land_cover_columns(dataset)

    if not 0 <= urban_threshold <= 1:
        raise ValueError("urban_threshold must be between 0 and 1.")

    if not 0 <= strong_growth_threshold <= 1:
        raise ValueError("strong_growth_threshold must be between 0 and 1.")

    labels = dataset.copy()
    labels["year"] = labels["year"].astype(int)
    labels = labels.sort_values(["cell_id", "year"]).reset_index(drop=True)

    labels["is_urban"] = labels["built_probability_mean"] >= urban_threshold

    group = labels.groupby("cell_id", sort=False)

    labels["target_year"] = group["year"].shift(-1)
    labels["built_probability_next_year"] = group["built_probability_mean"].shift(-1)
    labels["built_probability_change_next_year"] = (
        labels["built_probability_next_year"] - labels["built_probability_mean"]
    )

    if "built_label_frequency" in labels.columns:
        labels["built_label_frequency_next_year"] = group["built_label_frequency"].shift(-1)
        labels["built_label_frequency_change_next_year"] = (
            labels["built_label_frequency_next_year"] - labels["built_label_frequency"]
        )

    labels["is_urban_next_year"] = labels["built_probability_next_year"] >= urban_threshold

    labels["has_next_year_target"] = labels["target_year"].notna()

    if drop_incomplete_targets:
        labels = labels.loc[labels["has_next_year_target"]].copy()

    labels["target_year"] = labels["target_year"].astype("Int64")

    labels["urbanized_next_year"] = ((~labels["is_urban"]) & labels["is_urban_next_year"]).astype(
        int
    )

    labels["strong_urban_growth_next_year"] = (
        labels["built_probability_change_next_year"] >= strong_growth_threshold
    ).astype(int)

    labels["urban_state_transition"] = [
        _build_transition_label(is_urban, is_urban_next_year)
        for is_urban, is_urban_next_year in zip(
            labels["is_urban"],
            labels["is_urban_next_year"],
            strict=True,
        )
    ]

    labels["urban_threshold"] = urban_threshold
    labels["strong_growth_threshold"] = strong_growth_threshold

    bool_columns = ["is_urban", "is_urban_next_year", "has_next_year_target"]
    for column in bool_columns:
        labels[column] = labels[column].astype(bool)

    numeric_columns = [
        "built_probability_mean",
        "built_probability_next_year",
        "built_probability_change_next_year",
    ]

    for column in numeric_columns:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")

    labels = labels.replace([np.inf, -np.inf], np.nan)

    ordered_columns = [
        "cell_id",
        "city_id",
        "year",
        "target_year",
        "built_probability_mean",
        "built_probability_next_year",
        "built_probability_change_next_year",
        "built_label_frequency",
        "built_label_frequency_next_year",
        "built_label_frequency_change_next_year",
        "is_urban",
        "is_urban_next_year",
        "urbanized_next_year",
        "strong_urban_growth_next_year",
        "urban_state_transition",
        "has_next_year_target",
        "urban_threshold",
        "strong_growth_threshold",
    ]

    existing_columns = [column for column in ordered_columns if column in labels.columns]
    extra_columns = [column for column in labels.columns if column not in existing_columns]

    return labels[existing_columns + extra_columns]


def save_urbanization_labels(
    dataset: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save urbanization labels as Parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(path, index=False)

    return path
