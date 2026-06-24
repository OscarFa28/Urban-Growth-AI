from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

WGS84_CRS = "EPSG:4326"
DEFAULT_GRID_SIZES_M = [200, 500, 1000, 2000]


def read_city_boundary(path: str | Path) -> gpd.GeoDataFrame:
    """Read a city boundary GeoJSON file."""
    boundary_path = Path(path)

    if not boundary_path.exists():
        raise FileNotFoundError(f"City boundary file not found: {boundary_path}")

    boundary = gpd.read_file(boundary_path)

    if boundary.empty:
        raise ValueError(f"City boundary file is empty: {boundary_path}")

    if boundary.crs is None:
        boundary = boundary.set_crs(WGS84_CRS)

    return boundary.to_crs(WGS84_CRS)


def get_boundary_city_id(boundary: gpd.GeoDataFrame) -> str:
    """Extract city_id from a city boundary GeoDataFrame."""
    if "city_id" not in boundary.columns:
        raise ValueError("Boundary must contain a 'city_id' column.")

    city_id = boundary["city_id"].iloc[0]

    if not isinstance(city_id, str) or not city_id:
        raise ValueError("Boundary 'city_id' must be a non-empty string.")

    return city_id


def get_metric_crs(boundary: gpd.GeoDataFrame) -> Any:
    """Estimate a local projected CRS in meters for accurate grid generation."""
    metric_crs = boundary.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError("Could not estimate a metric CRS for the boundary.")

    return metric_crs


def _get_polygonal_boundary(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only polygonal geometries from the boundary."""
    polygon_mask = boundary.geometry.apply(lambda geom: isinstance(geom, Polygon | MultiPolygon))
    polygonal_boundary = boundary.loc[polygon_mask].copy()

    if polygonal_boundary.empty:
        raise ValueError("Boundary does not contain polygonal geometries.")

    return polygonal_boundary


def _create_candidate_cells(
    bounds: tuple[float, float, float, float],
    cell_size_m: int,
    city_id: str,
) -> gpd.GeoDataFrame:
    """Create candidate square cells covering the boundary bounds."""
    minx, miny, maxx, maxy = bounds

    x_coords = np.arange(minx, maxx + cell_size_m, cell_size_m)
    y_coords = np.arange(miny, maxy + cell_size_m, cell_size_m)

    rows = []

    for row_idx, y in enumerate(y_coords[:-1]):
        for col_idx, x in enumerate(x_coords[:-1]):
            rows.append(
                {
                    "city_id": city_id,
                    "grid_size_m": cell_size_m,
                    "row": row_idx,
                    "col": col_idx,
                    "cell_id": f"{city_id}_{cell_size_m}m_r{row_idx:05d}_c{col_idx:05d}",
                    "geometry": box(x, y, x + cell_size_m, y + cell_size_m),
                }
            )

    return gpd.GeoDataFrame(rows)


def generate_grid(
    boundary: gpd.GeoDataFrame,
    cell_size_m: int,
    city_id: str | None = None,
    clip_to_boundary: bool = True,
) -> gpd.GeoDataFrame:
    """Generate a grid for a city boundary using a metric CRS."""
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be greater than zero.")

    boundary = _get_polygonal_boundary(boundary)

    if city_id is None:
        city_id = get_boundary_city_id(boundary)

    metric_crs = get_metric_crs(boundary)
    boundary_metric = boundary.to_crs(metric_crs)

    city_geometry = boundary_metric.geometry.union_all()
    candidate_grid = _create_candidate_cells(
        bounds=city_geometry.bounds,
        cell_size_m=cell_size_m,
        city_id=city_id,
    )
    candidate_grid = candidate_grid.set_crs(metric_crs)

    intersects_mask = candidate_grid.intersects(city_geometry)
    grid = candidate_grid.loc[intersects_mask].copy()

    if clip_to_boundary:
        grid["geometry"] = grid.geometry.intersection(city_geometry)

    grid["nominal_area_m2"] = float(cell_size_m * cell_size_m)
    grid["area_m2"] = grid.geometry.area
    grid["coverage_ratio"] = grid["area_m2"] / grid["nominal_area_m2"]

    grid = grid.loc[grid["area_m2"] > 0].copy()
    grid = grid.reset_index(drop=True)
    grid["cell_index"] = grid.index + 1

    return grid.to_crs(WGS84_CRS)[
        [
            "city_id",
            "grid_size_m",
            "cell_id",
            "cell_index",
            "row",
            "col",
            "nominal_area_m2",
            "area_m2",
            "coverage_ratio",
            "geometry",
        ]
    ]


def generate_multi_resolution_grids(
    boundary: gpd.GeoDataFrame,
    grid_sizes_m: list[int] | None = None,
    city_id: str | None = None,
    clip_to_boundary: bool = True,
) -> dict[int, gpd.GeoDataFrame]:
    """Generate multiple grid resolutions for a city boundary."""
    if grid_sizes_m is None:
        grid_sizes_m = DEFAULT_GRID_SIZES_M

    return {
        grid_size_m: generate_grid(
            boundary=boundary,
            cell_size_m=grid_size_m,
            city_id=city_id,
            clip_to_boundary=clip_to_boundary,
        )
        for grid_size_m in grid_sizes_m
    }


def grid_output_path(
    city_id: str,
    country_code: str,
    grid_size_m: int,
    output_dir: str | Path = "data/interim/grids",
) -> Path:
    """Build the standard output path for a city grid."""
    return (
        Path(output_dir)
        / country_code.lower()
        / f"{grid_size_m}m"
        / f"{city_id}_grid_{grid_size_m}m.geojson"
    )


def save_grid(grid: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Save a generated grid as GeoJSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid.to_file(path, driver="GeoJSON")

    return path


def boundary_input_path(
    city_id: str,
    country_code: str,
    boundary_dir: str | Path = "data/external/boundaries",
) -> Path:
    """Build the standard input path for a city boundary."""
    return Path(boundary_dir) / country_code.lower() / f"{city_id}.geojson"
