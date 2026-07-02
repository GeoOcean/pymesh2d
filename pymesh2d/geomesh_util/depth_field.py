import numpy as np
import pyproj
import rasterio
from rasterio.transform import rowcol
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.spatial import cKDTree

def depth_field_from_dat(x, y, z, input_crs, output_crs, interp_method="linear"):
    """
    Create a callable depth field from a .dat file containing x y z points.
    No projection handling — assumes all coordinates are in the same system.

    Parameters
    ----------
    x, y, z : ndarray
        1D arrays of point coordinates and depth values.
    input_crs : str or pyproj.CRS
        CRS of the input coordinates (e.g. UTM zone). Default 'EPSG:32630'.
    output_crs : str or pyproj.CRS
        CRS in which the depth field will be queried (e.g. 'EPSG:32630' for UTM zone 30N).
    interp_method : {'linear', 'nearest'}, optional
        Interpolation method to use (default 'linear')
    delimiter : str, optional
        Delimiter used in the .dat file (default: auto-detected by numpy)

    Returns
    -------
    depth_field : function
        Callable: depth_field(xy) -> interpolated depth values (m)
        where xy is an array of shape (N, 2) with [x, y] coordinates.
    """

    # --- Clean invalid values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]

    # --- Create interpolator
    if interp_method == "linear":
        interp = LinearNDInterpolator(list(zip(x, y)), z, fill_value=np.nan)
    elif interp_method == "nearest":
        interp = NearestNDInterpolator(list(zip(x, y)), z)
    else:
        raise ValueError("interp_method must be 'linear' or 'nearest'")

    input_crs = pyproj.CRS.from_user_input(input_crs)
    output_crs = pyproj.CRS.from_user_input(output_crs)
    transformer = pyproj.Transformer.from_crs(output_crs, input_crs, always_xy=True)

    # --- Closure function
    def depth_field(xy):
        """
        Returns interpolated depth (z) for given (x, y) coordinates.
        xy : (N, 2) array
        """
        xs, ys = xy[:, 0], xy[:, 1]
        xs, ys = transformer.transform(xs, ys)
        depth = -interp(xs, ys)
        depth[np.isnan(depth)] = 0.0
        return np.asarray(depth, dtype=float)

    return depth_field


def depth_field_from_tif(tiff_path, output_crs, method="nearest"):
    """
    Create a callable depth field from a GeoTIFF bathymetry file.

    Parameters
    ----------
    tiff_path : str
        Path to the bathymetry GeoTIFF file.
    output_crs : str or pyproj.CRS
        CRS of the coordinates passed to the returned depth field (e.g. UTM).
    method : {'nearest', 'linear'}, optional
        'nearest': nearest pixel (default, fast).
        'linear': bilinear interpolation (smoother, better for hfun).

    Returns
    -------
    depth_field : callable
        depth_field(xy) -> depth (m), xy shape (N, 2) in output_crs.
    """

    from scipy.interpolate import RegularGridInterpolator

    dataset = rasterio.open(tiff_path)
    band = dataset.read(1)
    nodata = dataset.nodata
    transform = dataset.transform
    raster_crs = dataset.crs

    output_crs = pyproj.CRS.from_user_input(output_crs)
    raster_crs = pyproj.CRS.from_user_input(raster_crs) if raster_crs else output_crs
    if raster_crs != output_crs:
        transformer = pyproj.Transformer.from_crs(
            output_crs, raster_crs, always_xy=True
        )
    else:
        transformer = None

    # Depth = -elevation; mask nodata for interpolator
    depth_grid = -np.asarray(band, dtype=np.float64)
    if nodata is not None:
        depth_grid = np.where(band == nodata, np.nan, depth_grid)

    if method == "linear":
        # Build interpolator in pixel (row, col) space
        inv_transform = ~transform
        rows = np.arange(band.shape[0])
        cols = np.arange(band.shape[1])
        interp = RegularGridInterpolator(
            (rows, cols),
            depth_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    def depth_field(xy):
        xs, ys = xy[:, 0], xy[:, 1]
        if transformer is not None:
            xs, ys = transformer.transform(xs, ys)
        xs, ys = np.asarray(xs), np.asarray(ys)

        if method == "nearest":
            rows, cols = rowcol(transform, xs, ys)
            rows = np.clip(rows, 0, band.shape[0] - 1)
            cols = np.clip(cols, 0, band.shape[1] - 1)
            depth = depth_grid[rows, cols]
            return np.asarray(depth, dtype=np.float64)
        else:
            # method == "linear": continuous (col, row) from inverse transform
            col_row = np.column_stack(inv_transform * (xs, ys))
            # RegularGridInterpolator expects (row, col) for array [rows, cols]
            row_col = col_row[:, [1, 0]]
            depth = interp(row_col)
            return np.asarray(depth, dtype=np.float64)

    return depth_field


def depth_field_from_xr(ds, input_crs, output_crs, var_name="elevation"):
    """
    Create a callable depth field from an xarray.Dataset (bathymetry grid),
    reprojecting coordinates from dataset CRS to the desired output CRS.
    
    Uses KDTree for accurate nearest-neighbor search instead of searchsorted,
    ensuring consistent results with direct raster access.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing bathymetry (e.g. GEBCO subset) with coordinates (lat, lon).
    input_crs : str or pyproj.CRS
        CRS of the dataset coordinates (e.g. 'EPSG:4326' for lat/lon).
    output_crs : str or pyproj.CRS
        CRS in which the depth field will be queried (e.g. 'EPSG:32630' for UTM zone 30N).
    var_name : str, optional
        Name of the variable in the dataset containing elevation data (default 'elevation').

    Returns
    -------
    depth_field : function
        Callable: depth_field(xy) -> depth values (m)
        where xy is an array of shape (N, 2) with [x, y] coordinates in `output_crs`.
    """

    # -----------------------extract lon/lat grid and data
    lon = ds["lon"].values
    lat = ds["lat"].values
    z = np.asarray(ds[var_name].values)

    if z.ndim == 2:
        Lon, Lat = np.meshgrid(lon, lat, indexing='xy')
        grid_points = np.column_stack([Lon.ravel(), Lat.ravel()])
        grid_values = z.ravel()
    else:
        raise ValueError(f"z must be 2D, got shape {z.shape}")

    tree = cKDTree(grid_points)

    # -----------------------prepare transformers
    input_crs = pyproj.CRS.from_user_input(input_crs)
    output_crs = pyproj.CRS.from_user_input(output_crs)
    to_ds = pyproj.Transformer.from_crs(output_crs, input_crs, always_xy=True)

    # -----------------------closure function
    def depth_field(xy):
        """
        Returns interpolated depth (nearest neighbor) at given coordinates.
        xy : (N, 2) array in output_crs (e.g., UTM)
        """
        xs, ys = xy[:, 0], xy[:, 1]

        # -----------------------reproject query points to dataset CRS
        lon_q, lat_q = to_ds.transform(xs, ys)
        query_points = np.column_stack([lon_q, lat_q])

        # -----------------------find nearest neighbors using KDTree
        _, indices = tree.query(query_points, k=1)

        # -----------------------sample depth (depth = -elevation)
        depth = -grid_values[indices]
        
        depth = np.asarray(depth, dtype=np.float64)

        return depth

    return depth_field