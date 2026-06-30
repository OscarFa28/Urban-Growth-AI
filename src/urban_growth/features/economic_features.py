"""Economic feature engineering from INEGI DENUE."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

DENUE_COLUMNS = [
    "id",
    "codigo_act",
    "nombre_act",
    "per_ocu",
    "tipoUniEco",
    "cve_ent",
    "cve_mun",
    "latitud",
    "longitud",
]

ECONOMIC_SECTORS = [
    "agriculture",
    "mining",
    "utilities",
    "construction",
    "manufacturing",
    "wholesale",
    "retail",
    "transportation",
    "information",
    "finance",
    "real_estate",
    "professional_services",
    "business_support",
    "education",
    "health",
    "recreation",
    "food_lodging",
    "other_services",
    "government",
    "unknown",
]

EMPLOYEE_SIZE_CLASSES = ["micro", "small", "medium", "large", "unknown"]


MEXICO_LCC_CRS = (
    "+proj=lcc +lat_0=12 +lon_0=-102 +lat_1=17.5 +lat_2=29.5 "
    "+x_0=2500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
)


def classify_denue_sector(activity_code: object) -> str:
    """Classify DENUE SCIAN activity code into broad economic sectors."""
    code = str(activity_code).strip()

    if len(code) < 2 or not code[:2].isdigit():
        return "unknown"

    prefix = int(code[:2])

    if prefix == 11:
        return "agriculture"
    if prefix == 21:
        return "mining"
    if prefix == 22:
        return "utilities"
    if prefix == 23:
        return "construction"
    if 31 <= prefix <= 33:
        return "manufacturing"
    if prefix == 43:
        return "wholesale"
    if prefix == 46:
        return "retail"
    if prefix in {48, 49}:
        return "transportation"
    if prefix == 51:
        return "information"
    if prefix == 52:
        return "finance"
    if prefix == 53:
        return "real_estate"
    if prefix == 54:
        return "professional_services"
    if prefix == 56:
        return "business_support"
    if prefix == 61:
        return "education"
    if prefix == 62:
        return "health"
    if prefix == 71:
        return "recreation"
    if prefix == 72:
        return "food_lodging"
    if prefix == 81:
        return "other_services"
    if prefix == 93:
        return "government"

    return "unknown"


def classify_employee_size(employee_range: object) -> str:
    """Classify DENUE employee range into broad business size classes."""
    value = str(employee_range).strip().lower()

    if value in {"0 a 5 personas", "6 a 10 personas"}:
        return "micro"
    if value in {"11 a 30 personas", "31 a 50 personas"}:
        return "small"
    if value in {"51 a 100 personas", "101 a 250 personas"}:
        return "medium"
    if "251" in value:
        return "large"

    return "unknown"


def _read_csv_preview(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1", dtype=str, nrows=1)


def find_denue_csv_files(denue_dir: Path) -> list[Path]:
    """Find DENUE CSV files inside downloaded folders."""
    paths = sorted(denue_dir.rglob("conjunto_de_datos/*.csv"))

    if not paths:
        msg = f"No DENUE CSV files found in {denue_dir}"
        raise FileNotFoundError(msg)

    return paths


def load_denue_points(
    denue_dir: Path,
    target_state_codes: set[str] | None = None,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    chunksize: int = 200_000,
) -> gpd.GeoDataFrame:
    """Load DENUE points filtered by state and bounding box."""
    frames = []

    for path in find_denue_csv_files(denue_dir):
        preview = _read_csv_preview(path)
        available_columns = [column for column in DENUE_COLUMNS if column in preview.columns]

        for chunk in pd.read_csv(
            path,
            encoding="latin1",
            dtype=str,
            usecols=available_columns,
            chunksize=chunksize,
        ):
            chunk["cve_ent"] = chunk["cve_ent"].astype(str).str.strip().str.zfill(2)

            if target_state_codes is not None:
                chunk = chunk[chunk["cve_ent"].isin(target_state_codes)].copy()

            if chunk.empty:
                continue

            chunk["latitud"] = pd.to_numeric(chunk["latitud"], errors="coerce")
            chunk["longitud"] = pd.to_numeric(chunk["longitud"], errors="coerce")
            chunk = chunk.dropna(subset=["latitud", "longitud"])

            if bbox_wgs84 is not None:
                minx, miny, maxx, maxy = bbox_wgs84
                chunk = chunk[
                    chunk["longitud"].between(minx, maxx) & chunk["latitud"].between(miny, maxy)
                ].copy()

            if chunk.empty:
                continue

            chunk["economic_sector"] = chunk["codigo_act"].map(classify_denue_sector)
            chunk["employee_size_class"] = chunk["per_ocu"].map(classify_employee_size)

            frames.append(chunk)

    if not frames:
        columns = DENUE_COLUMNS + ["economic_sector", "employee_size_class", "geometry"]
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:4326")

    df = pd.concat(frames, ignore_index=True)

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326",
    )


def _cell_area_km2(cells: gpd.GeoDataFrame) -> pd.Series:
    if "cell_area_km2" in cells.columns:
        return cells["cell_area_km2"].astype(float)

    if "area_m2" in cells.columns:
        return cells["area_m2"].astype(float) / 1_000_000

    return cells.geometry.area / 1_000_000


def _target_state_codes_from_cells(cells: pd.DataFrame) -> set[str]:
    if "municipality_cvegeo" in cells.columns:
        return set(
            cells["municipality_cvegeo"].dropna().astype(str).str.strip().str.zfill(5).str[:2]
        )

    return {
        "01",
        "02",
        "09",
        "11",
        "14",
        "19",
        "22",
        "24",
        "32",
    }


def build_economic_features(
    cells: gpd.GeoDataFrame,
    denue_points: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Aggregate DENUE economic features into grid cells."""
    if "cell_id" not in cells.columns:
        msg = "Cells must include cell_id."
        raise ValueError(msg)

    output = cells.copy()
    output["economic_business_count_total"] = 0

    for sector in ECONOMIC_SECTORS:
        output[f"economic_{sector}_business_count"] = 0

    for size_class in EMPLOYEE_SIZE_CLASSES:
        output[f"economic_{size_class}_business_count"] = 0

    output["economic_distance_to_nearest_business_m"] = np.nan
    output["economic_distance_to_nearest_business_km"] = np.nan
    output["economic_nearest_business_sector"] = pd.NA
    output["economic_nearest_business_activity_code"] = pd.NA
    output["economic_nearest_business_employee_size"] = pd.NA

    output["economic_cell_area_km2"] = _cell_area_km2(output)

    if denue_points.empty:
        output["economic_business_density_per_km2"] = 0.0
        return output

    if output.crs is None:
        msg = "Cells must have a CRS."
        raise ValueError(msg)

    metric_crs = MEXICO_LCC_CRS
    cells_metric = output[["cell_id", "geometry"]].to_crs(metric_crs)
    points_metric = denue_points.to_crs(metric_crs)

    joined = gpd.sjoin(
        points_metric[
            [
                "id",
                "codigo_act",
                "per_ocu",
                "economic_sector",
                "employee_size_class",
                "geometry",
            ]
        ],
        cells_metric,
        how="inner",
        predicate="within",
    )

    if not joined.empty:
        total_counts = joined.groupby("cell_id").size()
        output["economic_business_count_total"] = (
            output["cell_id"].map(total_counts).fillna(0).astype(int)
        )

        sector_counts = joined.groupby(["cell_id", "economic_sector"]).size().unstack(fill_value=0)

        for sector in ECONOMIC_SECTORS:
            column = f"economic_{sector}_business_count"
            if sector in sector_counts.columns:
                output[column] = output["cell_id"].map(sector_counts[sector]).fillna(0).astype(int)

        size_counts = (
            joined.groupby(["cell_id", "employee_size_class"]).size().unstack(fill_value=0)
        )

        for size_class in EMPLOYEE_SIZE_CLASSES:
            column = f"economic_{size_class}_business_count"
            if size_class in size_counts.columns:
                output[column] = (
                    output["cell_id"].map(size_counts[size_class]).fillna(0).astype(int)
                )

    output["economic_business_density_per_km2"] = (
        (output["economic_business_count_total"] / output["economic_cell_area_km2"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    for sector in ECONOMIC_SECTORS:
        count_column = f"economic_{sector}_business_count"
        density_column = f"economic_{sector}_business_density_per_km2"
        output[density_column] = (
            (output[count_column] / output["economic_cell_area_km2"])
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

    centroids = cells_metric[["cell_id"]].copy()
    centroids = gpd.GeoDataFrame(
        centroids,
        geometry=cells_metric.geometry.centroid,
        crs=metric_crs,
    )

    nearest = gpd.sjoin_nearest(
        centroids,
        points_metric[
            [
                "codigo_act",
                "economic_sector",
                "employee_size_class",
                "geometry",
            ]
        ],
        how="left",
        distance_col="economic_distance_to_nearest_business_m",
    )

    nearest = nearest.drop_duplicates("cell_id")

    output = output.merge(
        nearest[
            [
                "cell_id",
                "economic_distance_to_nearest_business_m",
                "economic_sector",
                "codigo_act",
                "employee_size_class",
            ]
        ],
        on="cell_id",
        how="left",
        suffixes=("", "_nearest"),
    )

    output["economic_distance_to_nearest_business_m"] = output[
        "economic_distance_to_nearest_business_m_nearest"
    ]
    output["economic_distance_to_nearest_business_km"] = (
        output["economic_distance_to_nearest_business_m"] / 1000
    )
    output["economic_nearest_business_sector"] = output["economic_sector"]
    output["economic_nearest_business_activity_code"] = output["codigo_act"]
    output["economic_nearest_business_employee_size"] = output["employee_size_class"]

    output = output.drop(
        columns=[
            "economic_distance_to_nearest_business_m_nearest",
            "economic_sector",
            "codigo_act",
            "employee_size_class",
        ],
        errors="ignore",
    )

    return gpd.GeoDataFrame(output, geometry="geometry", crs=cells.crs)


def build_denue_economic_features(
    cells: gpd.GeoDataFrame,
    denue_dir: Path,
    target_state_codes: set[str] | None = None,
    bbox_buffer_degrees: float = 0.25,
    chunksize: int = 200_000,
) -> gpd.GeoDataFrame:
    """Build economic features from raw DENUE CSV files."""
    cells_wgs84 = cells.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = cells_wgs84.total_bounds
    bbox = (
        float(minx - bbox_buffer_degrees),
        float(miny - bbox_buffer_degrees),
        float(maxx + bbox_buffer_degrees),
        float(maxy + bbox_buffer_degrees),
    )

    states = target_state_codes or _target_state_codes_from_cells(cells)

    denue_points = load_denue_points(
        denue_dir=denue_dir,
        target_state_codes=states,
        bbox_wgs84=bbox,
        chunksize=chunksize,
    )

    return build_economic_features(cells=cells, denue_points=denue_points)
