"""INEGI demographic feature engineering."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

INEGI_ENCODINGS = ("utf-8-sig", "latin1", "cp1252")

INEGI_ID_COLUMNS = ["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"]

COUNT_FEATURES = {
    "POBTOT": "population_total",
    "POBFEM": "female_population",
    "POBMAS": "male_population",
    "P_0A2": "population_0_2",
    "P_3A5": "population_3_5",
    "P_6A11": "population_6_11",
    "P_12A14": "population_12_14",
    "P_15A17": "population_15_17",
    "P_18A24": "population_18_24",
    "P_12YMAS": "population_12_plus",
    "P_15YMAS": "population_15_plus",
    "P_18YMAS": "population_18_plus",
    "P_60YMAS": "population_60_plus",
    "POB0_14": "population_0_14",
    "POB15_64": "population_15_64",
    "POB65_MAS": "population_65_plus",
    "PEA": "economically_active_population",
    "POCUPADA": "employed_population",
    "PDESOCUP": "unemployed_population",
    "PSINDER": "population_without_health_services",
    "PDER_SS": "population_with_health_services",
    "VIVTOT": "housing_units_total",
    "TVIVHAB": "occupied_housing_units",
    "TVIVPAR": "private_housing_units_total",
    "VIVPAR_HAB": "occupied_private_housing_units",
    "OCUPVIVPAR": "private_housing_occupants",
    "VPH_C_ELEC": "housing_with_electricity",
    "VPH_AGUADV": "housing_with_piped_water",
    "VPH_DRENAJ": "housing_with_drainage",
    "VPH_INTER": "housing_with_internet",
    "VPH_AUTOM": "housing_with_car",
    "VPH_MOTO": "housing_with_motorcycle",
    "VPH_BICI": "housing_with_bicycle",
    "VPH_PC": "housing_with_computer",
    "VPH_CEL": "housing_with_cellphone",
    "VPH_TV": "housing_with_tv",
    "VPH_REFRI": "housing_with_refrigerator",
    "VPH_LAVAD": "housing_with_washing_machine",
    "VPH_SNBIEN": "housing_without_basic_goods",
}

AVERAGE_FEATURES = {
    "GRAPROES": "average_schooling_years",
    "PROM_HNV": "average_children_born",
    "PROM_OCUP": "source_average_occupants_per_dwelling",
}


def read_csv_robust(path: Path, **kwargs: object) -> tuple[pd.DataFrame, str]:
    """Read INEGI CSV files using a small encoding fallback chain."""
    for encoding in INEGI_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs), encoding
        except UnicodeDecodeError:
            continue

    msg = f"Could not decode CSV file: {path}"
    raise ValueError(msg)


def clean_inegi_numeric(series: pd.Series) -> pd.Series:
    """Convert INEGI numeric columns, treating reserved values as missing."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(
        {
            "*": np.nan,
            "N/D": np.nan,
            "ND": np.nan,
            "": np.nan,
        }
    )
    cleaned = cleaned.str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def build_inegi_ageb_cvegeo(df: pd.DataFrame) -> pd.Series:
    """Build INEGI AGEB CVEGEO from census columns."""
    return (
        df["ENTIDAD"].astype(str).str.strip().str.zfill(2)
        + df["MUN"].astype(str).str.strip().str.zfill(3)
        + df["LOC"].astype(str).str.strip().str.zfill(4)
        + df["AGEB"].astype(str).str.strip().str.zfill(4)
    )


def filter_ageb_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep AGEB total rows and drop state/municipality/locality/block rows."""
    return df[
        df["AGEB"].astype(str).str.strip().ne("0000") & df["MZA"].astype(str).str.strip().eq("000")
    ].copy()


def find_resageburb_csv_paths(resageburb_dir: Path) -> list[Path]:
    """Find 2020 RESAGEBURB CSV files."""
    paths = sorted(resageburb_dir.glob("RESAGEBURB_*CSV20.csv"))
    if not paths:
        msg = f"No RESAGEBURB CSV20 files found in {resageburb_dir}"
        raise FileNotFoundError(msg)
    return paths


def find_ageb_shapefiles(ageb_geometry_dir: Path) -> list[Path]:
    """Find state-level AGEB shapefiles and ignore the national 00a.shp file."""
    paths = []

    for path in sorted(ageb_geometry_dir.rglob("*a.shp")):
        stem = path.stem
        state_code = stem[:2]

        if len(stem) == 3 and stem.endswith("a") and state_code.isdigit() and state_code != "00":
            paths.append(path)

    if not paths:
        msg = f"No state AGEB shapefiles found in {ageb_geometry_dir}"
        raise FileNotFoundError(msg)

    return paths


def load_resageburb_2020(resageburb_dir: Path) -> pd.DataFrame:
    """Load and normalize INEGI 2020 AGEB-level census data."""
    csv_paths = find_resageburb_csv_paths(resageburb_dir)
    source_columns = sorted(
        set(INEGI_ID_COLUMNS)
        | set(COUNT_FEATURES)
        | set(AVERAGE_FEATURES)
        | {"NOM_ENT", "NOM_MUN", "NOM_LOC"}
    )

    frames = []

    for path in csv_paths:
        preview, _ = read_csv_robust(path, dtype=str, nrows=1)
        available_columns = [column for column in source_columns if column in preview.columns]

        missing_id_columns = [
            column for column in INEGI_ID_COLUMNS if column not in preview.columns
        ]
        if missing_id_columns:
            msg = f"{path.name} is missing required columns: {missing_id_columns}"
            raise ValueError(msg)

        df, _ = read_csv_robust(path, dtype=str, usecols=available_columns)
        df = filter_ageb_total_rows(df)
        df["CVEGEO"] = build_inegi_ageb_cvegeo(df)

        for source_column in set(COUNT_FEATURES) | set(AVERAGE_FEATURES):
            if source_column in df.columns:
                df[source_column] = clean_inegi_numeric(df[source_column])

        frames.append(df)

    census = pd.concat(frames, ignore_index=True)

    if census["CVEGEO"].duplicated().any():
        duplicates = census.loc[census["CVEGEO"].duplicated(), "CVEGEO"].head()
        msg = f"Duplicated AGEB CVEGEO keys found: {duplicates.tolist()}"
        raise ValueError(msg)

    return census


def load_ageb_geometries_2020(ageb_geometry_dir: Path) -> gpd.GeoDataFrame:
    """Load INEGI 2020 AGEB geometries from state shapefiles."""
    shapefiles = find_ageb_shapefiles(ageb_geometry_dir)

    frames = []
    target_crs = None

    for path in shapefiles:
        gdf = gpd.read_file(path)

        required_columns = {"CVEGEO", "geometry"}
        missing_columns = required_columns - set(gdf.columns)
        if missing_columns:
            msg = f"{path.name} is missing columns: {sorted(missing_columns)}"
            raise ValueError(msg)

        gdf["CVEGEO"] = gdf["CVEGEO"].astype(str).str.strip()

        if target_crs is None:
            target_crs = gdf.crs
        elif gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        keep_columns = [
            column
            for column in [
                "CVEGEO",
                "CVE_ENT",
                "CVE_MUN",
                "CVE_LOC",
                "CVE_AGEB",
                "geometry",
            ]
            if column in gdf.columns
        ]
        frames.append(gdf[keep_columns])

    ageb_geo = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=target_crs,
    )

    if ageb_geo["CVEGEO"].duplicated().any():
        duplicates = ageb_geo.loc[ageb_geo["CVEGEO"].duplicated(), "CVEGEO"].head()
        msg = f"Duplicated AGEB geometry keys found: {duplicates.tolist()}"
        raise ValueError(msg)

    return ageb_geo


def load_inegi_ageb_census_2020(
    resageburb_dir: Path,
    ageb_geometry_dir: Path,
) -> gpd.GeoDataFrame:
    """Load and join INEGI 2020 AGEB census data with AGEB geometry."""
    census = load_resageburb_2020(resageburb_dir)
    ageb_geo = load_ageb_geometries_2020(ageb_geometry_dir)

    census_keys = set(census["CVEGEO"])
    geometry_keys = set(ageb_geo["CVEGEO"])
    missing_geometry_count = len(census_keys - geometry_keys)

    if missing_geometry_count:
        print(
            "Warning: "
            f"{missing_geometry_count} AGEB census rows do not have geometry. "
            "They will be dropped."
        )

    joined = ageb_geo.merge(census, on="CVEGEO", how="inner", validate="one_to_one")
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=ageb_geo.crs)


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 0.0,
) -> pd.Series:
    result = numerator.astype(float) / denominator.replace(0, np.nan).astype(float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _ensure_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        df[column] = default
    return df[column]


def add_demographic_features(
    cells: gpd.GeoDataFrame,
    ageb_data: gpd.GeoDataFrame,
    cell_id_column: str = "cell_id",
) -> gpd.GeoDataFrame:
    """Allocate AGEB-level demographic data to grid cells using area overlap."""
    if cell_id_column not in cells.columns:
        msg = f"Cell id column not found: {cell_id_column}"
        raise ValueError(msg)

    if cells.crs is None:
        msg = "Cells GeoDataFrame must have a CRS."
        raise ValueError(msg)

    if ageb_data.crs is None:
        msg = "AGEB GeoDataFrame must have a CRS."
        raise ValueError(msg)

    result = cells.copy()
    original_crs = result.crs
    analysis_crs = ageb_data.crs

    cells_projected = result[[cell_id_column, "geometry"]].copy()
    cells_projected = cells_projected.to_crs(analysis_crs)
    cells_projected["demographic_cell_area_m2"] = cells_projected.geometry.area

    available_count_features = {
        source: output for source, output in COUNT_FEATURES.items() if source in ageb_data.columns
    }
    available_average_features = {
        source: output for source, output in AVERAGE_FEATURES.items() if source in ageb_data.columns
    }

    ageb_columns = (
        ["CVEGEO", "geometry"] + list(available_count_features) + list(available_average_features)
    )
    ageb_projected = ageb_data[ageb_columns].copy().to_crs(analysis_crs)
    ageb_projected["demographic_ageb_area_m2"] = ageb_projected.geometry.area
    ageb_projected = ageb_projected[ageb_projected["demographic_ageb_area_m2"] > 0].copy()

    intersections = gpd.overlay(
        cells_projected,
        ageb_projected,
        how="intersection",
        keep_geom_type=True,
    )

    if intersections.empty:
        output = result.copy()
        output["demographic_cell_area_m2"] = cells_projected["demographic_cell_area_m2"].to_numpy()
        output["demographic_source_area_m2"] = 0.0
        output["demographic_source_area_coverage_ratio"] = 0.0
        output["demographic_source_ageb_count"] = 0

        for output_column in COUNT_FEATURES.values():
            output[output_column] = 0.0

        for output_column in AVERAGE_FEATURES.values():
            output[output_column] = 0.0

        return _add_derived_demographic_features(output)

    intersections["demographic_intersection_area_m2"] = intersections.geometry.area
    intersections["demographic_ageb_area_weight"] = _safe_divide(
        intersections["demographic_intersection_area_m2"],
        intersections["demographic_ageb_area_m2"],
    )

    named_aggs = {
        "demographic_source_area_m2": (
            "demographic_intersection_area_m2",
            "sum",
        ),
        "demographic_source_ageb_count": ("CVEGEO", "nunique"),
    }

    for source_column, output_column in available_count_features.items():
        intersections[output_column] = (
            intersections[source_column].astype(float)
            * intersections["demographic_ageb_area_weight"]
        )
        named_aggs[output_column] = (output_column, "sum")

    average_weight_columns = []

    for source_column, output_column in available_average_features.items():
        weight_column = f"__{output_column}_weighted"
        average_weight_columns.append(weight_column)
        intersections[weight_column] = (
            intersections[source_column].astype(float)
            * intersections["demographic_intersection_area_m2"]
        )
        named_aggs[weight_column] = (weight_column, "sum")

    aggregated = intersections.groupby(cell_id_column).agg(**named_aggs).reset_index()

    cell_areas = cells_projected[[cell_id_column, "demographic_cell_area_m2"]].drop_duplicates(
        cell_id_column
    )

    aggregated = aggregated.merge(cell_areas, on=cell_id_column, how="left")

    for _source_column, output_column in available_average_features.items():
        weight_column = f"__{output_column}_weighted"
        aggregated[output_column] = _safe_divide(
            aggregated[weight_column],
            aggregated["demographic_source_area_m2"],
        )
        aggregated = aggregated.drop(columns=weight_column)

    output = result.merge(aggregated, on=cell_id_column, how="left")

    output["demographic_cell_area_m2"] = output["demographic_cell_area_m2"].fillna(0.0)
    output["demographic_source_area_m2"] = output["demographic_source_area_m2"].fillna(0.0)
    output["demographic_source_ageb_count"] = (
        output["demographic_source_ageb_count"].fillna(0).astype(int)
    )
    output["demographic_source_area_coverage_ratio"] = _safe_divide(
        output["demographic_source_area_m2"],
        output["demographic_cell_area_m2"],
    ).clip(0, 1)

    for output_column in COUNT_FEATURES.values():
        if output_column not in output.columns:
            output[output_column] = 0.0
        output[output_column] = output[output_column].fillna(0.0)

    for output_column in AVERAGE_FEATURES.values():
        if output_column not in output.columns:
            output[output_column] = 0.0
        output[output_column] = output[output_column].fillna(0.0)

    output = _add_derived_demographic_features(output)

    return gpd.GeoDataFrame(output, geometry="geometry", crs=original_crs)


def _add_derived_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    cell_area_km2 = _safe_divide(df["demographic_cell_area_m2"], pd.Series(1_000_000))

    population_total = _ensure_column(df, "population_total")
    occupied_housing = _ensure_column(df, "occupied_housing_units")
    housing_total = _ensure_column(df, "housing_units_total")

    df["population_density_per_km2"] = _safe_divide(population_total, cell_area_km2)
    df["housing_density_per_km2"] = _safe_divide(housing_total, cell_area_km2)
    df["occupied_housing_density_per_km2"] = _safe_divide(
        occupied_housing,
        cell_area_km2,
    )

    df["female_share"] = _safe_divide(
        _ensure_column(df, "female_population"),
        population_total,
    )
    df["male_share"] = _safe_divide(
        _ensure_column(df, "male_population"),
        population_total,
    )

    df["children_0_14_share"] = _safe_divide(
        _ensure_column(df, "population_0_14"),
        population_total,
    )
    df["working_age_share"] = _safe_divide(
        _ensure_column(df, "population_15_64"),
        population_total,
    )
    df["elderly_65_plus_share"] = _safe_divide(
        _ensure_column(df, "population_65_plus"),
        population_total,
    )
    df["elderly_60_plus_share"] = _safe_divide(
        _ensure_column(df, "population_60_plus"),
        population_total,
    )
    df["young_children_share"] = _safe_divide(
        _ensure_column(df, "population_0_2") + _ensure_column(df, "population_3_5"),
        population_total,
    )
    df["school_age_share"] = _safe_divide(
        _ensure_column(df, "population_6_11") + _ensure_column(df, "population_12_14"),
        population_total,
    )
    df["young_adult_share"] = _safe_divide(
        _ensure_column(df, "population_18_24"),
        population_total,
    )
    df["dependency_ratio"] = _safe_divide(
        _ensure_column(df, "population_0_14") + _ensure_column(df, "population_65_plus"),
        _ensure_column(df, "population_15_64"),
    )

    df["economically_active_rate"] = _safe_divide(
        _ensure_column(df, "economically_active_population"),
        _ensure_column(df, "population_12_plus"),
    )
    df["employment_rate"] = _safe_divide(
        _ensure_column(df, "employed_population"),
        _ensure_column(df, "economically_active_population"),
    )
    df["unemployment_rate"] = _safe_divide(
        _ensure_column(df, "unemployed_population"),
        _ensure_column(df, "economically_active_population"),
    )

    df["health_no_access_share"] = _safe_divide(
        _ensure_column(df, "population_without_health_services"),
        population_total,
    )
    df["health_access_share"] = _safe_divide(
        _ensure_column(df, "population_with_health_services"),
        population_total,
    )

    df["housing_occupancy_rate"] = _safe_divide(occupied_housing, housing_total)
    df["avg_occupants_per_housing_unit"] = _safe_divide(
        population_total,
        occupied_housing,
    )

    df["internet_access_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_internet"),
        occupied_housing,
    )
    df["car_ownership_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_car"),
        occupied_housing,
    )
    df["motorcycle_ownership_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_motorcycle"),
        occupied_housing,
    )
    df["bicycle_ownership_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_bicycle"),
        occupied_housing,
    )
    df["electricity_access_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_electricity"),
        occupied_housing,
    )
    df["piped_water_access_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_piped_water"),
        occupied_housing,
    )
    df["drainage_access_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_drainage"),
        occupied_housing,
    )
    df["computer_access_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_computer"),
        occupied_housing,
    )
    df["cellphone_access_rate"] = _safe_divide(
        _ensure_column(df, "housing_with_cellphone"),
        occupied_housing,
    )

    return df


def build_inegi_block_cvegeo(df: pd.DataFrame) -> pd.Series:
    """Build INEGI block/manzana CVEGEO from census columns."""
    return (
        df["ENTIDAD"].astype(str).str.strip().str.zfill(2)
        + df["MUN"].astype(str).str.strip().str.zfill(3)
        + df["LOC"].astype(str).str.strip().str.zfill(4)
        + df["AGEB"].astype(str).str.strip().str.zfill(4)
        + df["MZA"].astype(str).str.strip().str.zfill(3)
    )


def filter_block_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep block/manzana rows and drop state/municipality/locality/AGEB totals."""
    mza = df["MZA"].astype(str).str.strip()
    return df[
        df["AGEB"].astype(str).str.strip().ne("0000") & ~mza.isin(["0", "000", "0000", ""])
    ].copy()


def _normalize_state_codes(state_codes: list[str] | None) -> set[str] | None:
    if not state_codes:
        return None
    return {str(state_code).strip().zfill(2) for state_code in state_codes}


def _resageburb_state_code(path: Path) -> str:
    return path.stem.split("_")[1][:2]


def find_resageburb_csv_paths_by_state(
    resageburb_dir: Path,
    state_codes: list[str] | None = None,
) -> list[Path]:
    """Find 2020 RESAGEBURB CSV files, optionally filtered by state code."""
    normalized_state_codes = _normalize_state_codes(state_codes)
    paths = sorted(resageburb_dir.glob("RESAGEBURB_*CSV20.csv"))

    if normalized_state_codes:
        paths = [path for path in paths if _resageburb_state_code(path) in normalized_state_codes]

    if not paths:
        msg = f"No RESAGEBURB CSV20 files found in {resageburb_dir}"
        raise FileNotFoundError(msg)

    return paths


def find_block_shapefiles(
    block_geometry_dir: Path,
    state_codes: list[str] | None = None,
) -> list[Path]:
    """Find state-level block/manzana shapefiles and ignore 00m.shp."""
    normalized_state_codes = _normalize_state_codes(state_codes)
    paths = []

    for path in sorted(block_geometry_dir.rglob("*m.shp")):
        stem = path.stem
        source_state_code = stem[:2]

        is_state_block_layer = (
            len(stem) == 3
            and stem.endswith("m")
            and source_state_code.isdigit()
            and source_state_code != "00"
        )

        if not is_state_block_layer:
            continue

        if normalized_state_codes and source_state_code not in normalized_state_codes:
            continue

        paths.append(path)

    if not paths:
        msg = f"No state block shapefiles found in {block_geometry_dir}"
        raise FileNotFoundError(msg)

    return paths


def load_resageburb_blocks_2020(
    resageburb_dir: Path,
    state_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Load and normalize INEGI 2020 block/manzana-level census data."""
    csv_paths = find_resageburb_csv_paths_by_state(resageburb_dir, state_codes)

    source_columns = sorted(
        set(INEGI_ID_COLUMNS)
        | set(COUNT_FEATURES)
        | set(AVERAGE_FEATURES)
        | {"NOM_ENT", "NOM_MUN", "NOM_LOC"}
    )

    frames = []

    for path in csv_paths:
        preview, _ = read_csv_robust(path, dtype=str, nrows=1)
        available_columns = [column for column in source_columns if column in preview.columns]

        missing_id_columns = [
            column for column in INEGI_ID_COLUMNS if column not in preview.columns
        ]
        if missing_id_columns:
            msg = f"{path.name} is missing required columns: {missing_id_columns}"
            raise ValueError(msg)

        df, _ = read_csv_robust(path, dtype=str, usecols=available_columns)
        df = filter_block_rows(df)
        df["CVEGEO"] = build_inegi_block_cvegeo(df)

        for source_column in set(COUNT_FEATURES) | set(AVERAGE_FEATURES):
            if source_column in df.columns:
                df[source_column] = clean_inegi_numeric(df[source_column])

        frames.append(df)

    blocks = pd.concat(frames, ignore_index=True)

    if blocks["CVEGEO"].duplicated().any():
        duplicates = blocks.loc[blocks["CVEGEO"].duplicated(), "CVEGEO"].head()
        msg = f"Duplicated block CVEGEO keys found: {duplicates.tolist()}"
        raise ValueError(msg)

    return blocks


def load_block_geometries_2020(
    block_geometry_dir: Path,
    state_codes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load INEGI 2020 block/manzana geometries from state shapefiles."""
    shapefiles = find_block_shapefiles(block_geometry_dir, state_codes)

    frames = []
    target_crs = None

    for path in shapefiles:
        gdf = gpd.read_file(path)

        required_columns = {"CVEGEO", "geometry"}
        missing_columns = required_columns - set(gdf.columns)
        if missing_columns:
            msg = f"{path.name} is missing columns: {sorted(missing_columns)}"
            raise ValueError(msg)

        gdf["CVEGEO"] = gdf["CVEGEO"].astype(str).str.strip()

        if target_crs is None:
            target_crs = gdf.crs
        elif gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        keep_columns = [
            column
            for column in [
                "CVEGEO",
                "CVE_ENT",
                "CVE_MUN",
                "CVE_LOC",
                "CVE_AGEB",
                "CVE_MZA",
                "geometry",
            ]
            if column in gdf.columns
        ]

        frames.append(gdf[keep_columns])

    block_geo = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=target_crs,
    )

    if block_geo["CVEGEO"].duplicated().any():
        duplicates = block_geo.loc[block_geo["CVEGEO"].duplicated(), "CVEGEO"].head()
        msg = f"Duplicated block geometry keys found: {duplicates.tolist()}"
        raise ValueError(msg)

    return block_geo


def load_inegi_block_census_2020(
    resageburb_dir: Path,
    block_geometry_dir: Path,
    state_codes: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load and join INEGI 2020 block census data with block geometry."""
    census = load_resageburb_blocks_2020(resageburb_dir, state_codes)
    block_geo = load_block_geometries_2020(block_geometry_dir, state_codes)

    census_keys = set(census["CVEGEO"])
    geometry_keys = set(block_geo["CVEGEO"])
    missing_geometry_count = len(census_keys - geometry_keys)

    if missing_geometry_count:
        print(
            "Warning: "
            f"{missing_geometry_count} block census rows do not have geometry. "
            "They will be dropped."
        )

    joined = block_geo.merge(census, on="CVEGEO", how="inner", validate="one_to_one")
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=block_geo.crs)
