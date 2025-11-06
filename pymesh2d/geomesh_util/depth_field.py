from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from rasterio.transform import rowcol
import numpy as np
import pyproj
import rasterio


def make_depth_field_from_dat(dat_path, interp_method="linear", delimiter=None):
    """
    Create a callable depth field from a .dat file containing x y z points.
    No projection handling — assumes all coordinates are in the same system.

    Parameters
    ----------
    dat_path : str
        Path to the .dat file containing three columns: x y z
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

    # --- Load file
    try:
        data = np.loadtxt(dat_path, delimiter=delimiter)
    except Exception as e:
        raise ValueError(f"Error reading file '{dat_path}': {e}")

    if data.shape[1] < 3:
        raise ValueError("The .dat file must contain at least three columns: x y z")

    x, y, z = data[:, 0], data[:, 1], data[:, 2]

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

    # --- Closure function
    def depth_field(xy):
        """
        Returns interpolated depth (z) for given (x, y) coordinates.
        xy : (N, 2) array
        """
        xs, ys = xy[:, 0], xy[:, 1]
        depth = interp(xs, ys)
        depth[np.isnan(depth)] = 0.0
        return np.asarray(depth, dtype=float)

    return depth_field


def make_depth_field_from_tif(tiff_path, input_crs):
    """
    Create a callable depth field from a GeoTIFF bathymetry file,
    automatically reprojecting UTM coordinates to the raster CRS.

    Parameters
    ----------
    tiff_path : str
        Path to the bathymetry GeoTIFF file.
    input_crs : str or pyproj.CRS, optional
        CRS of the input coordinates (e.g. UTM zone). Default 'EPSG:32630'.

    Returns
    -------
    depth_field : function
        Callable: depth_field(xy) -> depth values (m)
        where xy is an array of shape (N, 2) with [x, y] coordinates in `input_crs`.
    """

    # --- Open raster and get metadata
    dataset = rasterio.open(tiff_path)
    band = dataset.read(1)
    nodata = dataset.nodata
    transform = dataset.transform
    raster_crs = dataset.crs

    # --- Prepare coordinate transformer (UTM -> raster CRS)
    input_crs = pyproj.CRS.from_user_input(input_crs)
    if raster_crs != input_crs:
        transformer = pyproj.Transformer.from_crs(
            input_crs, raster_crs, always_xy=True
        ).transform
    else:
        transformer = None

    # --- Closure function
    def depth_field(xy):
        """
        Returns interpolated depth (nearest) at given coordinates.
        xy : (N, 2) array in input_crs (e.g., UTM)
        """
        xs, ys = xy[:, 0], xy[:, 1]

        # Reproject to raster CRS if needed
        if transformer is not None:
            xs, ys = transformer(xs, ys)

        # Convert to pixel indices
        rows, cols = rowcol(transform, xs, ys)

        # Clip to valid raster bounds
        rows = np.clip(rows, 0, band.shape[0] - 1)
        cols = np.clip(cols, 0, band.shape[1] - 1)

        # Sample depths
        depth = -band[rows, cols]
        depth = np.where(depth == nodata, np.nan, depth)
        return depth

    return depth_field