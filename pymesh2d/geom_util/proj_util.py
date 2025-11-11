import numpy as np
import pyproj
from shapely.ops import transform


def get_utm_crs_from_crs(crs):
    """
    Get the appropriate UTM CRS for a given geographic CRS.

    Parameters
    ----------
    crs : pyproj.CRS
        Input coordinate reference system (geographic).

    Returns
    -------
    pyproj.CRS
        Corresponding UTM coordinate reference system.
    """

    if crs.is_projected:
        return crs
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = transformer.transform(0, 0)
    utm_zone = int((lon0 + 180) / 6) + 1
    epsg_code = 326 if lat0 >= 0 else 327
    return pyproj.CRS.from_epsg(epsg_code * 100 + utm_zone)


def reproject_node(node, crs_from, crs_to):
    """
    Reproject 2D geometry from one CRS to another.

    Parameters
    ----------
    node : ndarray of shape (N, 2)
        Array of vertex coordinates to be reprojected.
    crs_from : pyproj.CRS
        Source coordinate reference system.
    crs_to : pyproj.CRS
        Target coordinate reference system.

    Returns
    -------
    ndarray of shape (N, 2)
        Reprojected vertex coordinates.
    """

    node = np.asarray(node)
    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    x2, y2 = transformer.transform(node[:, 0], node[:, 1])
    return np.column_stack((x2, y2))


def reproject_geometry(geom, crs_from, crs_to):
    """
    Reproject a geometry or coordinate array from one CRS to another.

    Parameters
    ----------
    geom : ndarray of shape (N, 2) or shapely geometry
        Coordinates or geometry to be reprojected.
    crs_from : pyproj.CRS or str
        Source coordinate reference system.
    crs_to : pyproj.CRS or str
        Target coordinate reference system.

    Returns
    -------
    same type as input
        Reprojected geometry or coordinate array.
    """

    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    return transform(transformer.transform, geom)
