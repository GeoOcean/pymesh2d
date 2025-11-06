import numpy as np
import pyproj
import rasterio

from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator

def interpolate_from_xyz(dat_path, vert, method="rbf", delimiter=None, rbf_function="multiquadric", epsilon=None):
    """
    Interpolation of values from scattered (x, y, z, value) points at arbitrary 3D nodes.

    Parameters
    ----------
    dat_path : str
        Path to a .dat file containing at least 4 columns: x, y, z, value.
    vert : (N, 3) ndarray
        Target coordinates (x, y, z) where interpolation is evaluated.
    method : {'linear', 'nearest', 'rbf'}, optional
        Interpolation method:
        - 'linear' : piecewise linear interpolation using tetrahedra (scipy.LinearNDInterpolator)
        - 'nearest': nearest-neighbor interpolation using KDTree
        - 'rbf'    : smooth radial basis interpolation (scipy.RBFInterpolator)
    delimiter : str, optional
        Column delimiter in the file (default auto).
    rbf_function : str, optional
        RBF kernel function ('multiquadric', 'inverse', 'gaussian', 'thin_plate_spline', etc.)
        Used only if method='rbf'.
    epsilon : float, optional
        Shape parameter for RBF interpolation. Auto-estimated if None.

    Returns
    -------
    values_interp : (N,) ndarray
        Interpolated values at the target 3D points.
    """

    # --- Load data
    data = np.loadtxt(dat_path, delimiter=delimiter)
    if data.shape[1] < 4:
        raise ValueError("The .dat file must contain at least four columns: x y z value")

    x, y, z, val = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # --- Remove invalid points
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(val)
    x, y, z, val = x[mask], y[mask], z[mask], val[mask]
    points = np.column_stack((x, y, z))

    # --- Interpolation method
    if method == "linear":
        interp = LinearNDInterpolator(points, val, fill_value=np.nan)
        values_interp = interp(vert[:, 0], vert[:, 1], vert[:, 2])

    elif method == "nearest":
        tree = cKDTree(points)
        _, idx = tree.query(vert)
        values_interp = val[idx]

    elif method == "rbf":
        interp = RBFInterpolator(points, val, kernel=rbf_function, epsilon=epsilon)
        values_interp = interp(vert)

    else:
        raise ValueError("method must be 'linear', 'nearest', or 'rbf'")

    # --- Handle NaNs
    values_interp = np.asarray(values_interp, dtype=float)
    values_interp[np.isnan(values_interp)] = 0

    return values_interp



def interpolate_from_tiff(
    tiff_path,
    vert,
    input_crs=None,
    order=3,
    mode="constant",
    cval=np.nan
):
    """
    Fast interpolation of GeoTIFF values at mesh nodes (bicubic/bilinear).

    Parameters
    ----------
    tiff_path : str
        Path to GeoTIFF file.
    vert : (N, 2) array
        Node coordinates (x, y) in input CRS.
    input_crs : str or pyproj.CRS, optional
        CRS of input mesh. If None, assumes same as raster.
    order : int
        Interpolation order (0=nearest, 1=bilinear, 3=bicubic).
    mode : str
        Boundary handling mode for map_coordinates.
    cval : float
        Constant value outside domain if mode="constant".

    Returns
    -------
    z : (N,) array
        Interpolated raster values at mesh nodes.
    """
    with rasterio.open(tiff_path) as src:
        band = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata
        raster_crs = src.crs

        if input_crs is not None and pyproj.CRS.from_user_input(input_crs) != raster_crs:
            transformer = pyproj.Transformer.from_crs(
                input_crs, raster_crs, always_xy=True
            ).transform
            xs, ys = transformer(vert[:, 0], vert[:, 1])
        else:
            xs, ys = vert[:, 0], vert[:, 1]

        inv_transform = ~transform
        cols, rows = inv_transform * (xs, ys)

        z = -map_coordinates(band, [rows, cols], order=order, mode=mode, cval=cval)

        if nodata is not None:
            z[z == nodata] = cval

        return z

