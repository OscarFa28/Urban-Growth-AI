import re
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
    "coverage_status",
    "source_name",
    "source_year",
)

VALID_COVERAGE_STATUSES: frozenset[str] = frozenset(
    {
        "official_2020",
        "manual_review_required",
        "standalone_manual_review_required",
    }
)

_BOOLEAN_VALUES = frozenset({"true", "false"})
_CVEGEO_PATTERN = re.compile(r"^\d{5}$")


def _normalize_string_column(series: pd.Series) -> pd.Series:
    """Convert a registry column to stripped pandas strings."""
    return series.astype("string").fillna("").str.strip()


def normalize_metropolitan_municipalities(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize metropolitan municipality registry values without mutating input."""
    normalized = frame.copy()

    for column in REQUIRED_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _normalize_string_column(normalized[column])

    if "country_code" in normalized.columns:
        normalized["country_code"] = normalized["country_code"].str.upper()

    if "is_core_municipality" in normalized.columns:
        normalized["is_core_municipality"] = normalized["is_core_municipality"].str.lower()

    if "coverage_status" in normalized.columns:
        normalized["coverage_status"] = normalized["coverage_status"].str.lower()

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


def _format_row_indexes(indexes: pd.Index) -> str:
    return ", ".join(str(index) for index in indexes.tolist())


def _validate_required_values(frame: pd.DataFrame) -> None:
    for column in REQUIRED_COLUMNS:
        empty_rows = frame[column].eq("")

        if empty_rows.any():
            rows = _format_row_indexes(frame.index[empty_rows])
            raise ValueError(f"Column '{column}' cannot be empty. Empty rows: {rows}")


def _validate_municipality_cvegeo(frame: pd.DataFrame) -> None:
    invalid_rows = ~frame["municipality_cvegeo"].str.match(_CVEGEO_PATTERN)

    if invalid_rows.any():
        rows = _format_row_indexes(frame.index[invalid_rows])
        raise ValueError(
            f"Column 'municipality_cvegeo' must be a 5-digit string. Invalid rows: {rows}"
        )


def _validate_coverage_status(frame: pd.DataFrame) -> None:
    invalid_rows = ~frame["coverage_status"].isin(VALID_COVERAGE_STATUSES)

    if invalid_rows.any():
        invalid_values = sorted(frame.loc[invalid_rows, "coverage_status"].unique())
        values = ", ".join(str(value) for value in invalid_values)
        allowed = ", ".join(sorted(VALID_COVERAGE_STATUSES))
        raise ValueError(f"Invalid coverage_status values: {values}. Allowed values: {allowed}")


def _validate_is_core_municipality(frame: pd.DataFrame) -> None:
    invalid_rows = ~frame["is_core_municipality"].isin(_BOOLEAN_VALUES)

    if invalid_rows.any():
        rows = _format_row_indexes(frame.index[invalid_rows])
        raise ValueError(
            f"Column 'is_core_municipality' must be 'true' or 'false'. Invalid rows: {rows}"
        )


def validate_metropolitan_municipalities(frame: pd.DataFrame) -> None:
    """Validate the metropolitan municipality registry schema and keys."""
    _ensure_required_columns(frame)
    normalized = normalize_metropolitan_municipalities(frame)

    _validate_required_values(normalized)
    _validate_municipality_cvegeo(normalized)
    _validate_coverage_status(normalized)
    _validate_is_core_municipality(normalized)

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
            f"{row['metro_area_id']} + {row['municipality_cvegeo']}" for row in duplicate_keys
        )
        raise ValueError(f"Duplicate rows by metro_area_id + municipality_cvegeo: {keys}")


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


def _manual_review_required(statuses: pd.Series) -> bool:
    return statuses.str.contains("manual_review_required", regex=False).any()


def list_metro_areas(frame: pd.DataFrame) -> pd.DataFrame:
    """List metropolitan areas and compact coverage counts."""
    validate_metropolitan_municipalities(frame)
    normalized = normalize_metropolitan_municipalities(frame)

    group_columns = ["country_code", "metro_area_id", "metro_area_name"]

    return (
        normalized.groupby(group_columns, as_index=False, sort=True)
        .agg(
            municipality_count=("municipality_cvegeo", "nunique"),
            city_count=("city_id", "nunique"),
            coverage_statuses=("coverage_status", _join_unique_values),
            requires_manual_review=("coverage_status", _manual_review_required),
        )
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
    coverage_statuses_by_metro_area = {
        str(row["metro_area_id"]): str(row["coverage_statuses"])
        for row in metro_areas.to_dict(orient="records")
    }
    coverage_statuses_by_city_id = (
        normalized.groupby("city_id")["coverage_status"].agg(_join_unique_values).to_dict()
    )
    cities_by_coverage_status = {
        str(status): sorted(group["city_id"].drop_duplicates().tolist())
        for status, group in normalized.groupby("coverage_status")
    }

    manual_review_city_ids = sorted(
        city_id
        for city_id, statuses in coverage_statuses_by_city_id.items()
        if "manual_review_required" in statuses
    )
    manual_review_metro_area_ids = sorted(
        metro_area_id
        for metro_area_id, statuses in coverage_statuses_by_metro_area.items()
        if "manual_review_required" in statuses
    )

    return {
        "row_count": int(len(normalized)),
        "country_codes": sorted(normalized["country_code"].drop_duplicates().tolist()),
        "metro_area_count": int(normalized["metro_area_id"].nunique()),
        "municipality_count": int(normalized["municipality_cvegeo"].nunique()),
        "city_count": int(normalized["city_id"].nunique()),
        "municipalities_by_metro_area": municipalities_by_metro_area,
        "cities_by_metro_area": cities_by_metro_area,
        "coverage_statuses_by_metro_area": coverage_statuses_by_metro_area,
        "coverage_statuses_by_city_id": coverage_statuses_by_city_id,
        "cities_by_coverage_status": cities_by_coverage_status,
        "manual_review_required_city_ids": manual_review_city_ids,
        "manual_review_required_metro_area_ids": manual_review_metro_area_ids,
    }
