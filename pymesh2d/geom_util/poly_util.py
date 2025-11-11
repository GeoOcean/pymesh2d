import numpy as np
from shapely.geometry import Polygon


def resample_polygon(polygon, spacing: float):
    """
    Resample a shapely Polygon (or MultiPolygon) at uniform spacing along
    its exterior and interior boundaries.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or MultiPolygon
        Input polygon geometry (must be closed).
    spacing : float
        Desired distance between consecutive points along the boundaries.

    Returns
    -------
    Polygon
        Resampled polygon with the same topology (holes preserved).
    """

    # -----------------------handle MultiPolygon input
    if polygon.geom_type == "MultiPolygon":
        # keep largest polygon only
        polygon = max(polygon.geoms, key=lambda p: p.area)

    # -----------------------resample exterior ring
    exterior = np.asarray(polygon.exterior.coords)
    exterior_resampled = _resample_closed_line(exterior, spacing)

    # -----------------------resample interior rings (holes)
    interiors_resampled = []
    for hole in polygon.interiors:
        coords = np.asarray(hole.coords)
        hole_resampled = _resample_closed_line(coords, spacing)
        interiors_resampled.append(hole_resampled)

    # -----------------------build new polygon
    poly_new = Polygon(exterior_resampled, interiors_resampled)

    return poly_new


# -----------------------helper: resample a closed line
def _resample_closed_line(line, spacing):
    # ensure ring is closed
    if not np.allclose(line[0], line[-1]):
        line = np.vstack([line, line[0]])

    # cumulative distances
    seglen = np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1]))
    dist = np.insert(np.cumsum(seglen), 0, 0)
    total = dist[-1]
    if total == 0:
        return line

    # uniformly spaced distances
    new_d = np.arange(0, total, spacing)
    x_new = np.interp(new_d, dist, line[:, 0])
    y_new = np.interp(new_d, dist, line[:, 1])

    # close explicitly
    resampled = np.column_stack((x_new, y_new))
    if not np.allclose(resampled[0], resampled[-1]):
        resampled = np.vstack([resampled, resampled[0]])

    return resampled


def buffer_area_for_polygon(polygon: Polygon, area_factor: float) -> Polygon:
    """
    Buffer the polygon by a factor of its area divided by its length.
    This is a heuristic to ensure that the buffer is proportional to the size of the polygon.

    Parameters
    ----------
    polygon : Polygon
        The polygon to be buffered.
    mas : float
        The buffer factor.

    Returns
    -------
    Polygon
        The buffered polygon.
    """

    return polygon.buffer(area_factor * polygon.area / polygon.length)


def polygon_to_node_edge(poly):
    """
    Extract node and edge arrays (PSLG format) from a Shapely Polygon or MultiPolygon.
    Ensures all contours are closed and verifies even connectivity.

    Parameters
    ----------
    poly : shapely.geometry.Polygon or MultiPolygon
        Input polygon geometry.

    Returns
    -------
    node : ndarray (N, 2)
        Node coordinates (x, y)
    edge : ndarray (E, 2)
        Edge connectivity (0-based indices)

    Raises
    ------
    ValueError
        If the resulting edge structure is not properly closed.
    """
    # -----------------------handle MultiPolygon recursively
    if poly.geom_type == "MultiPolygon":
        nodes_all, edges_all = [], []
        offset = 0
        for p in poly.geoms:
            node, edge = polygon_to_node_edge(p)
            edges_all.append(edge + offset)
            nodes_all.append(node)
            offset += len(node)
        return np.vstack(nodes_all), np.vstack(edges_all)

    # -----------------------extract exterior coordinates
    ext = np.array(poly.exterior.coords)
    node = [ext[:-1]]  # remove duplicate closing point
    edge = [np.column_stack([np.arange(len(ext) - 1), np.arange(1, len(ext))])]
    edge[-1][-1, 1] = 0  # close loop explicitly

    # -----------------------extract holes (if any)
    for hole in poly.interiors:
        pts = np.array(hole.coords)
        n0 = len(np.vstack(node))
        node.append(pts[:-1])  # skip duplicate closure
        e = np.column_stack(
            [np.arange(n0, n0 + len(pts) - 1), np.arange(n0 + 1, n0 + len(pts))]
        )
        e[-1, 1] = n0
        edge.append(e)

    # -----------------------combine all
    node = np.vstack(node)
    edge = np.vstack(edge).astype(int)

    # -----------------------verify closure condition
    nnod = node.shape[0]
    nadj = np.bincount(edge.ravel(), minlength=nnod)
    if np.any(nadj % 2 != 0):
        raise ValueError(
            "Invalid topology: some nodes are not closed (odd connectivity)."
        )

    return node, edge
