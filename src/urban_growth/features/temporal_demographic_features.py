"""Temporal demographic feature engineering."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from urban_growth.features.demographic_features import (
    AVERAGE_FEATURES,
    COUNT_FEATURES,
    _add_derived_demographic_features,
    clean_inegi_numeric,
    read_csv_robust,
)

STATE_NAME_TO_CODE = {
    "aguascalientes": "01",
    "baja california": "02",
    "baja california sur": "03",
    "campeche": "04",
    "coahuila": "05",
    "coahuila de zaragoza": "05",
    "colima": "06",
    "chiapas": "07",
    "chihuahua": "08",
    "ciudad de mexico": "09",
    "ciudad de méxico": "09",
    "cdmx": "09",
    "durango": "10",
    "guanajuato": "11",
    "guerrero": "12",
    "hidalgo": "13",
    "jalisco": "14",
    "mexico": "15",
    "méxico": "15",
    "michoacan": "16",
    "michoacán": "16",
    "michoacan de ocampo": "16",
    "michoacán de ocampo": "16",
    "morelos": "17",
    "nayarit": "18",
    "nuevo leon": "19",
    "nuevo león": "19",
    "oaxaca": "20",
    "puebla": "21",
    "queretaro": "22",
    "querétaro": "22",
    "quintana roo": "23",
    "san luis potosi": "24",
    "san luis potosí": "24",
    "sinaloa": "25",
    "sonora": "26",
    "tabasco": "27",
    "tamaulipas": "28",
    "tlaxcala": "29",
    "veracruz": "30",
    "veracruz de ignacio de la llave": "30",
    "yucatan": "31",
    "yucatán": "31",
    "zacatecas": "32",
}


CITY_ID_TO_MUNICIPALITY_CVEGEO = {
    "mx_aguascalientes": "01001",
    "mx_arandas": "14008",
    "mx_cdmx": "09000",
    "mx_guadalajara": "14039",
    "mx_leon": "11020",
    "mx_monterrey": "19039",
    "mx_puerto_vallarta": "14067",
    "mx_queretaro": "22014",
    "mx_san_juan_de_los_lagos": "14073",
    "mx_san_luis_potosi": "24028",
    "mx_tijuana": "02004",
    "mx_zacatecas": "32056",
}


MUNICIPAL_ID_COLUMNS = ["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"]


def _normalize_state_name(value: object) -> str:
    return str(value).strip().lower()


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 1.0,
) -> pd.Series:
    result = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _available_source_columns(preview: pd.DataFrame) -> list[str]:
    requested = (
        set(MUNICIPAL_ID_COLUMNS)
        | set(COUNT_FEATURES)
        | set(AVERAGE_FEATURES)
        | {"NOM_ENT", "NOM_MUN"}
    )
    return [column for column in requested if column in preview.columns]


def _filter_municipal_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["LOC"].astype(str).str.strip().eq("0000")
        & df["AGEB"].astype(str).str.strip().eq("0000")
        & df["MZA"].astype(str).str.strip().eq("000")
    ].copy()


def _build_municipality_cvegeo(df: pd.DataFrame) -> pd.Series:
    return df["ENTIDAD"].astype(str).str.strip().str.zfill(2) + df["MUN"].astype(
        str
    ).str.strip().str.zfill(3)


def _clean_and_rename_municipal_data(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["municipality_cvegeo"] = _build_municipality_cvegeo(df)

    for source_column, output_column in COUNT_FEATURES.items():
        if source_column in df.columns:
            output[output_column] = clean_inegi_numeric(df[source_column])

    for source_column, output_column in AVERAGE_FEATURES.items():
        if source_column in df.columns:
            output[output_column] = clean_inegi_numeric(df[source_column])

    if output["municipality_cvegeo"].duplicated().any():
        duplicates = output.loc[
            output["municipality_cvegeo"].duplicated(),
            "municipality_cvegeo",
        ].head()
        msg = f"Duplicated municipality keys found: {duplicates.tolist()}"
        raise ValueError(msg)

    return output


def load_municipal_census_2010(census_2010_dir: Path) -> pd.DataFrame:
    """Load 2010 INEGI municipal totals from RESAGEBURB XLS files."""
    paths = sorted(census_2010_dir.glob("RESAGEBURB_*XLS10.xls"))

    if not paths:
        msg = f"No 2010 RESAGEBURB XLS files found in {census_2010_dir}"
        raise FileNotFoundError(msg)

    frames = []

    for path in paths:
        xls = pd.ExcelFile(path, engine="xlrd")

        for sheet_name in xls.sheet_names:
            preview = pd.read_excel(
                path,
                sheet_name=sheet_name,
                nrows=1,
                dtype=str,
                engine="xlrd",
            )
            available_columns = _available_source_columns(preview)

            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                dtype=str,
                engine="xlrd",
                usecols=available_columns,
            )

            df = _filter_municipal_total_rows(df)
            frames.append(df)

    census = pd.concat(frames, ignore_index=True)
    return _clean_and_rename_municipal_data(census)


def load_municipal_census_2020(census_2020_dir: Path) -> pd.DataFrame:
    """Load 2020 INEGI municipal totals from RESAGEBURB CSV files."""
    paths = sorted(census_2020_dir.glob("RESAGEBURB_*CSV20.csv"))

    if not paths:
        msg = f"No 2020 RESAGEBURB CSV files found in {census_2020_dir}"
        raise FileNotFoundError(msg)

    frames = []

    for path in paths:
        preview, _ = read_csv_robust(path, dtype=str, nrows=1)
        available_columns = _available_source_columns(preview)

        df, _ = read_csv_robust(path, dtype=str, usecols=available_columns)
        df = _filter_municipal_total_rows(df)
        frames.append(df)

    census = pd.concat(frames, ignore_index=True)
    return _clean_and_rename_municipal_data(census)


def build_municipal_temporal_adjustments(
    municipal_2010: pd.DataFrame,
    municipal_2020: pd.DataFrame,
    start_year: int,
    end_year: int,
    post_2020_method: str = "hold",
) -> pd.DataFrame:
    """Build municipal count ratios and average deltas by year."""
    if post_2020_method not in {"hold", "extrapolate"}:
        msg = "post_2020_method must be 'hold' or 'extrapolate'"
        raise ValueError(msg)

    merged = municipal_2010.merge(
        municipal_2020,
        on="municipality_cvegeo",
        how="inner",
        suffixes=("_2010", "_2020"),
    )

    count_columns = [
        output_column
        for output_column in COUNT_FEATURES.values()
        if f"{output_column}_2010" in merged.columns and f"{output_column}_2020" in merged.columns
    ]
    average_columns = [
        output_column
        for output_column in AVERAGE_FEATURES.values()
        if f"{output_column}_2010" in merged.columns and f"{output_column}_2020" in merged.columns
    ]

    records = []

    for year in range(start_year, end_year + 1):
        year_frame = merged[["municipality_cvegeo"]].copy()
        year_frame["year"] = year

        interpolation_factor = (year - 2010) / 10 if year <= 2020 else 1.0

        for column in count_columns:
            value_2010 = merged[f"{column}_2010"].astype(float)
            value_2020 = merged[f"{column}_2020"].astype(float)

            if year <= 2020 or post_2020_method == "hold":
                estimated_value = value_2010 + (value_2020 - value_2010) * interpolation_factor
            else:
                annual_growth = _safe_divide(value_2020, value_2010).pow(1 / 10)
                estimated_value = value_2020 * annual_growth.pow(year - 2020)

            year_frame[f"{column}_municipal_ratio_to_2020"] = _safe_divide(
                estimated_value,
                value_2020,
            )

        for column in average_columns:
            value_2010 = merged[f"{column}_2010"].astype(float)
            value_2020 = merged[f"{column}_2020"].astype(float)

            if year <= 2020 or post_2020_method == "hold":
                estimated_value = value_2010 + (value_2020 - value_2010) * interpolation_factor
            else:
                annual_delta = (value_2020 - value_2010) / 10
                estimated_value = value_2020 + annual_delta * (year - 2020)

            year_frame[f"{column}_municipal_delta_from_2020"] = estimated_value - value_2020

        records.append(year_frame)

    return pd.concat(records, ignore_index=True)


def build_cell_municipality_cvegeo(cells: pd.DataFrame) -> pd.Series:
    """Build municipality CVEGEO for cells."""
    if "municipality_cvegeo" in cells.columns:
        return cells["municipality_cvegeo"].astype(str).str.strip().str.zfill(5)

    if "city_id" in cells.columns:
        mapped = cells["city_id"].map(CITY_ID_TO_MUNICIPALITY_CVEGEO)

        if mapped.notna().all():
            return mapped.astype(str)

    if "municipality_id" not in cells.columns:
        msg = "Cells must include municipality_id, city_id, or municipality_cvegeo."
        raise ValueError(msg)

    def build_from_row(row: pd.Series) -> str:
        municipality_id = str(row["municipality_id"]).strip()

        if municipality_id in CITY_ID_TO_MUNICIPALITY_CVEGEO:
            return CITY_ID_TO_MUNICIPALITY_CVEGEO[municipality_id]

        if municipality_id.endswith(".0"):
            municipality_id = municipality_id[:-2]

        if municipality_id.isdigit() and len(municipality_id) >= 5:
            return municipality_id.zfill(5)[-5:]

        if not municipality_id.isdigit():
            msg = f"Invalid municipality_id: {municipality_id}"
            raise ValueError(msg)

        if "state" not in row:
            msg = "State column is required when municipality_id is not 5 digits."
            raise ValueError(msg)

        state_name = _normalize_state_name(row["state"])
        state_code = STATE_NAME_TO_CODE.get(state_name)

        if state_code is None:
            msg = f"Unknown state name for municipality mapping: {row['state']}"
            raise ValueError(msg)

        return state_code + municipality_id.zfill(3)

    return cells.apply(build_from_row, axis=1)


def build_temporal_demographic_features(
    cells_2020: gpd.GeoDataFrame,
    municipal_adjustments: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> gpd.GeoDataFrame:
    """Apply municipal temporal adjustments to 2020 cell demographic features."""
    cells = cells_2020.copy()
    cells["municipality_cvegeo"] = build_cell_municipality_cvegeo(cells)

    if "demographic_cell_area_m2" not in cells.columns:
        cells["demographic_cell_area_m2"] = cells.geometry.area

    yearly_frames = []

    for year in range(start_year, end_year + 1):
        yearly = cells.copy()
        yearly["year"] = year
        yearly_frames.append(yearly)

    temporal = gpd.GeoDataFrame(
        pd.concat(yearly_frames, ignore_index=True),
        geometry="geometry",
        crs=cells_2020.crs,
    )

    temporal = temporal.merge(
        municipal_adjustments,
        on=["municipality_cvegeo", "year"],
        how="left",
    )

    adjusted_count_columns = []

    for output_column in COUNT_FEATURES.values():
        ratio_column = f"{output_column}_municipal_ratio_to_2020"

        if output_column in temporal.columns and ratio_column in temporal.columns:
            temporal[output_column] = temporal[output_column].astype(float) * temporal[
                ratio_column
            ].fillna(1.0)
            adjusted_count_columns.append(output_column)

    adjusted_average_columns = []

    for output_column in AVERAGE_FEATURES.values():
        delta_column = f"{output_column}_municipal_delta_from_2020"

        if output_column in temporal.columns and delta_column in temporal.columns:
            temporal[output_column] = (
                temporal[output_column].astype(float) + temporal[delta_column].fillna(0.0)
            ).clip(lower=0)
            adjusted_average_columns.append(output_column)

    helper_columns = [
        column
        for column in temporal.columns
        if column.endswith("_municipal_ratio_to_2020")
        or column.endswith("_municipal_delta_from_2020")
    ]
    temporal = temporal.drop(columns=helper_columns)

    temporal["demographic_reference_year"] = 2020
    temporal["demographic_is_temporal_estimate"] = temporal["year"] != 2020
    temporal["demographic_temporal_method"] = (
        "municipal_linear_interpolation_2010_2020_hold_2020_after_2020"
    )
    temporal["demographic_adjusted_count_feature_count"] = len(adjusted_count_columns)
    temporal["demographic_adjusted_average_feature_count"] = len(adjusted_average_columns)

    temporal = _add_derived_demographic_features(temporal)

    return gpd.GeoDataFrame(temporal, geometry="geometry", crs=cells_2020.crs)
