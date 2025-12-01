import numpy as np
from shapely.geometry import LinearRing, Point, Polygon

from .grd_util import build_loops


def read_poly_from_dat(dat_path, delimiter=None):
    """
    Equivalent Python implementation of the MATLAB contour reader:
    Reads polygon(s) from a .dat file where 'NaN NaN' separates contours,
    automatically closes each contour, and builds the global node/edge arrays.

    Parameters
    ----------
    dat_path : str
        Path to the .dat file.
    delimiter : str, optional
        Delimiter for numpy.loadtxt (default: auto-detect).

    Returns
    -------
    node : ndarray of shape (N, 2)
        Node coordinates (x, y)
    edge : ndarray of shape (M, 2)
        Edge connectivity (zero-based indices)
    """

    # --- Load file
    p0 = np.loadtxt(dat_path, delimiter=delimiter)
    if p0.shape[1] < 2:
        raise ValueError("The .dat file must contain at least two columns: x y")

    # --- Find NaN separators
    isnan = np.isnan(p0[:, 0])
    s = np.where(isnan)[0]
    s = np.concatenate(([0], s, [len(p0)]))

    node = []
    edge = []
    cont = 0

    # --- Loop over polygons
    for i in range(len(s) - 1):
        p = p0[s[i] : s[i + 1], :]
        p = p[~np.isnan(p[:, 0])]  # remove NaN rows
        if len(p) == 0:
            continue

        n = len(p)
        # Close the polygon by connecting last point to first
        c = np.column_stack([np.arange(0, n), np.arange(1, n + 1)])
        c[-1, 1] = 0  # last edge closes to first

        # Apply offset to edge indices
        c = c + cont

        # Append
        node.append(p)
        edge.append(c)

        cont += n  # offset for next polygon

    # --- Concatenate all nodes and edges
    node = np.vstack(node)
    edge = np.vstack(edge).astype(int)

    return node, edge


def clean_polygon(
    poly: Polygon,
    min_perimeter: float = 100.0,
    min_vertices: int = 8,
    min_area: float = 5000.0,
) -> Polygon | None:
    """
    Clean a Polygon by removing small holes and discarding the polygon
    if it does not meet perimeter, vertex count, or area thresholds.

    Parameters
    ----------
    poly : Polygon
        Input polygon to be cleaned.
    min_perimeter : float, optional
        Minimum perimeter threshold for the polygon and its holes (default 100.0).
    min_vertices : int, optional
        Minimum number of vertices threshold for the polygon and its holes (default 8).
    min_area : float, optional
        Minimum area threshold for the polygon (default 5000.0).

    Returns
    -------
    Polygon or None
        Cleaned Polygon if valid, otherwise None.
    """

    # -----------------------check polygon validity
    if poly.is_empty:
        return None

    # -----------------------filter small polygon
    if (
        len(poly.exterior.coords) < min_vertices
        or poly.exterior.length < min_perimeter
        or poly.area < min_area
    ):
        return None

    # -----------------------filter small holes
    valid_holes = []
    for interior in poly.interiors:
        ring = LinearRing(interior.coords)
        if len(interior.coords) >= min_vertices and ring.length >= min_perimeter:
            valid_holes.append(interior)

    # -----------------------return cleaned polygon
    cleaned = Polygon(poly.exterior, valid_holes)
    return cleaned if cleaned.is_valid else cleaned.buffer(0)


def identify_boundary(vert, tria, z, zlim=0.0, Manual_open_boundary=None):
    """
    Identify open and land (including islands) boundaries from a triangulated mesh.
    Manual_open_boundary: shapely Polygon (optional)
    If provided, any edge whose midpoint lies inside this polygon will be classified as open.

    Parameters
    ----------
    vert : (N, 2) array
        Node coordinates (x, y).
    tria : (M, 3) array
        Triangle connectivity (node indices).
    z : (N,) array
        Elevation values at nodes.
    zlim : float
        Elevation threshold to classify open boundaries.
    Manual_open_boundary : shapely Polygon, optional
        Polygon defining manual open boundary areas.

    Returns
    -------
    edge_tag : (K, 3) array
        Edge connectivity with tags (node1, node2, tag), where tag=1 for open, tag=2 for land.
    edge_open : (L, 2) array
        Open boundary edges (node1, node2).
    edge_land : (P, 2) array
        Land boundary edges (node1, node2).
    """

    # --- Build edge list
    edges = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])
    edges = np.sort(edges, axis=1)

    edges_sorted, counts = np.unique(edges, axis=0, return_counts=True)
    edge_free = edges_sorted[counts == 1]
    if edge_free.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 2))

    loops = build_loops(edge_free)

    # --- Identify outer and inner loops
    polygons = []
    for loop in loops:
        xy = vert[loop]
        ring = LinearRing(xy)
        polygons.append((loop, ring.area))

    polygons.sort(key=lambda x: x[1], reverse=True)
    outer_loop = polygons[0][0]
    inner_loops = [p[0] for p in polygons[1:]]

    edge_open = []
    edge_land = []

    # --- Classify edges
    for loop in [outer_loop] + inner_loops:
        for i in range(len(loop) - 1):
            a, b = loop[i], loop[i + 1]
            zmean = 0.5 * (z[a] + z[b])

            # Calculate midpoint
            mid = (vert[a] + vert[b]) / 2.0

            # --- Test if inside manual open boundary
            in_manual_open = False
            if Manual_open_boundary is not None:
                in_manual_open = Manual_open_boundary.contains(Point(mid))

            # --- Final classification
            if (loop is outer_loop and zmean > zlim) or in_manual_open:
                edge_open.append([a, b])
            else:
                edge_land.append([a, b])

    edge_open = np.array(edge_open, dtype=int)
    edge_land = np.array(edge_land, dtype=int)

    # --- Assemble edge tags
    tag_open = np.ones((edge_open.shape[0], 1), dtype=int)
    tag_land = np.full((edge_land.shape[0], 1), 2, dtype=int)

    edge_tag_parts = []
    if edge_open.shape[0] > 0:
        edge_tag_parts.append(np.hstack([edge_open, tag_open]))
    if edge_land.shape[0] > 0:
        edge_tag_parts.append(np.hstack([edge_land, tag_land]))

    edge_tag = np.vstack(edge_tag_parts) if edge_tag_parts else np.empty((0, 3))

    return edge_tag, edge_open, edge_land