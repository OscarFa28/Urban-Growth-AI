from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_METROPOLITAN_REGISTRY_PATH = Path(
    "configs/metropolis/mx_metropolitan_municipalities_2020.csv"
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "country_code",
    "metro_area_id",
    "metro_area_name",
    "city_id",
    "city_name",
    "municipality_cvegeo",
    "municipality_name",
    "state_name",
    "state_code",
    "is_core_municipality",
    "source_name",
    "source_year",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "coverage_status",
    "notes",
)

_NORMALIZED_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
_NON_EMPTY_COLUMNS = (
    "municipality_cvegeo",
    "city_id",
    "metro_area_id",
)


def _normalize_string_column(series: pd.Series) -> pd.Series:
    """Convert a registry column to stripped pandas strings."""
    return series.astype("string").fillna("").str.strip()


def normalize_metropolitan_municipalities(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize metropolitan municipality registry values."""
    normalized = frame.copy()

    for column in _NORMALIZED_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _normalize_string_column(normalized[column])

    if "country_code" in normalized.columns:
        normalized["country_code"] = normalized["country_code"].str.upper()

    if "is_core_municipality" in normalized.columns:
        normalized["is_core_municipality"] = normalized["is_core_municipality"].str.lower()

    return normalized


def read_metropolitan_municipalities(path: str | Path) -> pd.DataFrame:
    """Read a metropolitan municipality registry CSV."""
    registry_path = Path(path)

    if not registry_path.exists():
        raise FileNotFoundError(f"Metropolitan registry file not found: {registry_path}")

    frame = pd.read_csv(registry_path, dtype="string", keep_default_na=False)

    return normalize_metropolitan_municipalities(frame)


def _ensure_required_columns(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required metropolitan registry columns: {missing}")


def validate_metropolitan_municipalities(frame: pd.DataFrame) -> None:
    """Validate the metropolitan municipality registry schema and keys."""
    _ensure_required_columns(frame)
    normalized = normalize_metropolitan_municipalities(frame)

    for column in _NON_EMPTY_COLUMNS:
        empty_rows = normalized[column].eq("")

        if empty_rows.any():
            row_numbers = [str(index) for index in normalized.index[empty_rows].tolist()]
            rows = ", ".join(row_numbers)
            raise ValueError(f"Column '{column}' cannot be empty. Empty rows: {rows}")

    duplicated = normalized.duplicated(
        subset=["metro_area_id", "municipality_cvegeo"],
        keep=False,
    )

    if duplicated.any():
        duplicate_keys = (
            normalized.loc[duplicated, ["metro_area_id", "municipality_cvegeo"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        keys = ", ".join(
            f"{row['metro_area_id']} + {row['municipality_cvegeo']}"
            for row in duplicate_keys
        )
        raise ValueError(
            "Duplicate rows by metro_area_id + municipality_cvegeo: "
            f"{keys}"
        )


def get_metro_municipalities(
    frame: pd.DataFrame,
    metro_area_id: str,
) -> pd.DataFrame:
    """Return all municipalities registered for a metropolitan area."""
    _ensure_required_columns(frame)
    normalized = normalize_metropolitan_municipalities(frame)
    metro_area_key = str(metro_area_id).strip()

    return normalized.loc[
        normalized["metro_area_id"].eq(metro_area_key),
    ].reset_index(drop=True)


def get_city_metro_coverage(
    frame: pd.DataFrame,
    city_id: str,
) -> pd.DataFrame:
    """Return the full metropolitan municipality coverage for a city_id."""
    _ensure_required_columns(frame)
    normalized = normalize_metropolitan_municipalities(frame)
    city_key = str(city_id).strip()

    city_rows = normalized.loc[normalized["city_id"].eq(city_key)]
    metro_area_ids = city_rows["metro_area_id"].drop_duplicates()

    return normalized.loc[
        normalized["metro_area_id"].isin(metro_area_ids),
    ].reset_index(drop=True)


def _join_unique_values(values: pd.Series) -> str:
    unique_values = sorted({str(value) for value in values if str(value)})
    return "|".join(unique_values)


def list_metro_areas(frame: pd.DataFrame) -> pd.DataFrame:
    """List metropolitan areas and compact coverage counts."""
    validate_metropolitan_municipalities(frame)
    normalized = normalize_metropolitan_municipalities(frame)

    group_columns = ["country_code", "metro_area_id", "metro_area_name"]
    aggregations: dict[str, tuple[str, Any]] = {
        "municipality_count": ("municipality_cvegeo", "nunique"),
        "city_count": ("city_id", "nunique"),
    }

    if "coverage_status" in normalized.columns:
        aggregations["coverage_statuses"] = ("coverage_status", _join_unique_values)

    return (
        normalized.groupby(group_columns, as_index=False, sort=True)
        .agg(**aggregations)
        .sort_values("metro_area_id")
        .reset_index(drop=True)
    )


def summarize_metropolitan_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-serializable coverage counts for the registry."""
    validate_metropolitan_municipalities(frame)
    normalized = normalize_metropolitan_municipalities(frame)
    metro_areas = list_metro_areas(normalized)

    municipalities_by_metro_area = {
        str(row["metro_area_id"]): int(row["municipality_count"])
        for row in metro_areas.to_dict(orient="records")
    }
    cities_by_metro_area = {
        str(row["metro_area_id"]): int(row["city_count"])
        for row in metro_areas.to_dict(orient="records")
    }

    summary: dict[str, Any] = {
        "row_count": int(len(normalized)),
        "country_codes": sorted(normalized["country_code"].drop_duplicates().tolist()),
        "metro_area_count": int(normalized["metro_area_id"].nunique()),
        "municipality_count": int(normalized["municipality_cvegeo"].nunique()),
        "city_count": int(normalized["city_id"].nunique()),
        "municipalities_by_metro_area": municipalities_by_metro_area,
        "cities_by_metro_area": cities_by_metro_area,
    }

    if "coverage_status" in normalized.columns:
        coverage_statuses = (
            normalized.groupby("metro_area_id")["coverage_status"]
            .agg(_join_unique_values)
            .to_dict()
        )
        summary["coverage_statuses_by_metro_area"] = coverage_statuses

    return summary
