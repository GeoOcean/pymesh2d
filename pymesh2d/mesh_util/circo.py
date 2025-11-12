import numpy as np

from ..mesh_ball.tribal2 import tribal2
from ..mesh_cost.triarea import triarea
from .tricon import tricon


def circumcenters(vert, tria, edge, tria_6col):
    """
    Compute circumcenters with logic matching MeshKernel's ComputeFaceCircumcenters.

    This function computes circumcenters for triangles in a triangulation, with
    special handling for triangles with boundary edges. The logic matches the
    behavior in MeshKernel's ComputeFaceCircumcenters function.

    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        XY coordinates of the vertices in the triangulation.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices (used if tria_6col is 3-column format).
    EDGE : ndarray of shape (E, 5)
        Edge connectivity array from tricon: [V1, V2, T1, T2, CE].
    TRIA_6COL : ndarray of shape (T, 6)
        Triangle-to-edge mapping [V1, V2, V3, E1, E2, E3] from tricon.

    Returns
    -------
    BB : ndarray of shape (T, 3)
        Circumballs for each triangle, where each row is `[XC, YC, RC²]`
        — the center coordinates and squared radius.

    Notes
    -----
    - If a triangle has all edges on boundary (numberOfInteriorEdges == 0):
      the center of mass is used instead of the circumcenter.
    - For other triangles: the circumcenter is computed normally, then projected
      onto the triangle boundary if it lies outside the triangle.
    - This matches the behavior in MeshKernel/src/MeshFaceCenters.cpp.

    References
    ----------
    MeshKernel implementation: MeshKernel/src/MeshFaceCenters.cpp
    """
    # ---------------------------------------------- basic checks
    if not (
        isinstance(vert, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(edge, np.ndarray)
        and isinstance(tria_6col, np.ndarray)
    ):
        raise TypeError("circumcenters:incorrectInputClass")

    if vert.ndim != 2 or edge.ndim != 2 or tria_6col.ndim != 2:
        raise ValueError("circumcenters:incorrectDimensions")

    # ---------------------------------------------- identify boundary edges
    # edge[:, 3] == -1 means no T2, so it's a boundary edge
    boundary_edges = edge[:, 3] == -1

    # ---------------------------------------------- get edge indices for each triangle
    # columns 3-5 contain edge indices
    edge_indices = tria_6col[:, 3:6]

    # ---------------------------------------------- count interior edges per triangle
    nboundary_edges = np.sum(boundary_edges[edge_indices], axis=1)
    numberOfInteriorEdges = 3 - nboundary_edges  # 3 edges per triangle

    # ---------------------------------------------- initialize output array
    # same format as tribal2: [cx, cy, r^2]
    bb = np.zeros((tria_6col.shape[0], 3))

    # ---------------------------------------------- calculate all circumcenters
    tria_3col = tria if tria.shape[1] == 3 else tria_6col[:, :3]
    bb_all = tribal2(vert, tria_3col)

    # ---------------------------------------------- process each triangle
    for t_idx in range(tria_6col.shape[0]):
        tri_verts = (
            tria_6col[t_idx, :3] if tria_6col.shape[1] >= 6 else tria[t_idx, :]
        )

        # If triangle has no interior edges (all edges on boundary), use center of mass
        if numberOfInteriorEdges[t_idx] == 0:
            center_of_mass = np.mean(vert[tri_verts], axis=0)
            bb[t_idx, 0:2] = center_of_mass
            bb[t_idx, 2] = 0.0
        else:
            # Triangle has at least one interior edge: compute circumcenter normally
            result = bb_all[t_idx, :].copy()

            # Check if circumcenter is inside the triangle
            is_inside = _is_point_in_triangle(
                result[0:2],
                vert[tri_verts[0]],
                vert[tri_verts[1]],
                vert[tri_verts[2]],
            )

            if not is_inside:
                # Circumcenter is outside: project onto intersection with triangle
                center_of_mass = np.mean(vert[tri_verts], axis=0)

                # Find intersection between segment (center_of_mass, result) and triangle edges
                intersection_found = False
                for n in range(3):
                    next_n = (n + 1) % 3
                    v0 = vert[tri_verts[n]]
                    v1 = vert[tri_verts[next_n]]

                    intersection = _segment_intersection(
                        center_of_mass, result[0:2], v0, v1
                    )
                    if intersection is not None:
                        result[0:2] = intersection
                        intersection_found = True
                        break

                # If no intersection found (shouldn't happen), use center of mass
                if not intersection_found:
                    result[0:2] = center_of_mass

            bb[t_idx, :] = result

    return bb


def _is_point_in_triangle(p, v0, v1, v2):
    """
    Check if point p is inside triangle (v0, v1, v2) using barycentric coordinates.

    Parameters
    ----------
    P : ndarray of shape (2,)
        Point coordinates to test.
    V0, V1, V2 : ndarray of shape (2,)
        Triangle vertex coordinates.

    Returns
    -------
    bool
        True if point is inside the triangle, False otherwise.
    """
    # ---------------------------------------------- compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    # ---------------------------------------------- compute dot products
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)

    # ---------------------------------------------- compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # ---------------------------------------------- check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def _segment_intersection(p1, p2, p3, p4):
    """
    Find intersection point of two line segments (p1, p2) and (p3, p4).

    Parameters
    ----------
    P1, P2 : ndarray of shape (2,)
        Endpoints of the first line segment.
    P3, P4 : ndarray of shape (2,)
        Endpoints of the second line segment.

    Returns
    -------
    ndarray of shape (2,) or None
        Intersection point if segments intersect, None otherwise.
    """
    # ---------------------------------------------- line segment intersection
    # using parametric equations
    d1 = p2 - p1
    d2 = p4 - p3

    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None  # Parallel lines

    diff = p3 - p1
    t1 = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    t2 = (diff[0] * d1[1] - diff[1] * d1[0]) / denom

    # ---------------------------------------------- check if intersection is within both segments
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return p1 + t1 * d1

    return None


def small_flow_links(vert, tria, edge, removesmalllinkstrsh=0.1, conn=None):
    """
    Identify and report small flow links between adjacent triangles.

    This function implements the logic from Delft3D-FM's flow_geominit.f90 to
    identify flow links that are too short. A flow link is the connection
    between two adjacent triangles, measured as the distance between their
    circumcenters.

    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        XY coordinates of the vertices in the triangulation.
    TRIA : ndarray of shape (T, 3) or (T, 6)
        Array of triangles (vertex indices).
    EDGE : ndarray of shape (E, 5)
        Edge connectivity array from tricon: [V1, V2, T1, T2, CE].
    REMOVESMALLLINKSTRSH : float, optional
        Threshold for removing small links (default 0.1).
        Matches the Removesmalllinkstrsh parameter in Delft3D-FM.
    CONN : ndarray, optional
        Constrained edges (used for tricon if tria is 3-column format).

    Returns
    -------
    NLINKTOOSMALL : int
        Number of small flow links identified.
    SMALL_LINK_INDICES : ndarray
        Array of edge indices that are too small.

    Notes
    -----
    - The threshold formula matches Delft3D-FM:
      dxlim = 0.9 * removesmalllinkstrsh * 0.5 * (sqrt(ba(n1)) + sqrt(ba(n2)))
    - A link is considered too small if dxlink < dxlim, where dxlink is the
      distance between circumcenters of adjacent triangles.
    - All edges with two faces are checked, regardless of boundary status.

    References
    ----------
    Delft3D-FM: flow_geominit.f90 (lines 399-429)
    MeshKernel: Mesh2D::GetEdgesCrossingSmallFlowEdges
    """
    # ---------------------------------------------- basic checks
    if tria.size == 0 or edge.size == 0:
        return 0, np.array([], dtype=int)

    # ---------------------------------------------- get tria in 6-column format
    if tria.shape[1] == 3:
        _, tria_6col = tricon(tria, conn if conn is not None else np.empty((0, 2), dtype=int))
    else:
        tria_6col = tria

    # ---------------------------------------------- calculate circumcenters
    bb = circumcenters(vert, tria, edge, tria_6col)
    xz = bb[:, 0:2]
    tria_3col = tria if tria.shape[1] == 3 else tria_6col[:, :3]
    ba = np.abs(triarea(vert, tria_3col))

    # ---------------------------------------------- find edges connecting two triangles
    # Delft3D checks all 2D links (KN(3, L) == 2), which includes constrained edges
    # MeshKernel also checks all edges with two faces
    internal_mask = (edge[:, 2] >= 0) & (edge[:, 3] >= 0)

    if np.sum(internal_mask) == 0:
        return 0, np.array([], dtype=int)

    t1_indices = edge[internal_mask, 2].astype(int)
    t2_indices = edge[internal_mask, 3].astype(int)
    internal_edge_indices = np.where(internal_mask)[0]

    # ---------------------------------------------- filter valid indices
    # MeshKernel checks ALL edges with two faces, regardless of boundary status or area
    valid_mask = (
        (t1_indices >= 0) & (t1_indices < len(ba)) &
        (t2_indices >= 0) & (t2_indices < len(ba))
    )

    if not np.any(valid_mask):
        return 0, np.array([], dtype=int)

    t1_indices = t1_indices[valid_mask]
    t2_indices = t2_indices[valid_mask]
    internal_edge_indices = internal_edge_indices[valid_mask]

    # ---------------------------------------------- calculate thresholds and distances
    # Match Delft3D-FM formula: dxlim = 0.9 * removesmalllinkstrsh * 0.5 * (sqrt(ba(n1)) + sqrt(ba(n2)))
    sqrt_ba = np.sqrt(ba[t1_indices]) + np.sqrt(ba[t2_indices])
    dxlim = 0.9 * removesmalllinkstrsh * 0.5 * sqrt_ba
    dxlink = np.linalg.norm(xz[t2_indices] - xz[t1_indices], axis=1)

    # ---------------------------------------------- identify too small links
    # Match Delft3D-FM: if (dxlink < dxlim) - no additional conditions
    too_small = dxlink < dxlim
    nlinktoosmall = np.sum(too_small)

    if nlinktoosmall > 0:
        small_link_indices = internal_edge_indices[too_small]
        print(f"\n{nlinktoosmall:5d} small flow links discarded")
    else:
        small_link_indices = np.array([], dtype=int)

    return nlinktoosmall, small_link_indices


def small_flow_centers(vert, tria, edge, removesmalllinkstrsh=0.1, conn=None):
    """
    Get circumcenters of triangles involved in small flow links.

    This function identifies small flow links and returns the circumcenters
    of all triangles involved in these problematic links.

    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        XY coordinates of the vertices in the triangulation.
    TRIA : ndarray of shape (T, 3) or (T, 6)
        Array of triangles (vertex indices).
    EDGE : ndarray of shape (E, 5)
        Edge connectivity array from tricon: [V1, V2, T1, T2, CE].
    REMOVESMALLLINKSTRSH : float, optional
        Threshold for removing small links (default 0.1).
    CONN : ndarray, optional
        Constrained edges (used for tricon if tria is 3-column format).

    Returns
    -------
    CIRCUMCENTERS : ndarray of shape (N, 2)
        XY coordinates of circumcenters from triangles involved in small flow links.
        Returns empty array if no small links are found.
    NLINKTOOSMALL : int
        Number of small flow links identified.

    Notes
    -----
    - Returns circumcenters for all triangles involved in small flow links,
      including those with boundary edges (which use center of mass or projected circumcenters).
    - Useful for visualization and debugging of mesh quality issues.
    """
    # ---------------------------------------------- basic checks
    if tria.size == 0 or edge.size == 0:
        return np.array([]).reshape(0, 2), 0

    # ---------------------------------------------- identify small flow links
    nlinktoosmall, small_link_indices = small_flow_links(
        vert, tria, edge, removesmalllinkstrsh, conn
    )
    if nlinktoosmall == 0:
        return np.array([]).reshape(0, 2), 0

    # ---------------------------------------------- get tria in 6-column format
    if tria.shape[1] == 3:
        _, tria_6col = tricon(tria, conn if conn is not None else np.empty((0, 2), dtype=int))
    else:
        tria_6col = tria

    # ---------------------------------------------- calculate circumcenters
    bb = circumcenters(vert, tria, edge, tria_6col)
    xz = bb[:, 0:2]

    # ---------------------------------------------- get problematic triangles
    problem_edges = edge[small_link_indices]
    all_problem_triangles = np.unique(
        np.concatenate([
            problem_edges[:, 2].astype(int),
            problem_edges[:, 3].astype(int),
        ])
    )

    if len(all_problem_triangles) == 0:
        return np.array([]).reshape(0, 2), nlinktoosmall

    # ---------------------------------------------- return circumcenters
    # no exclusion of boundary triangles
    return xz[all_problem_triangles], nlinktoosmall
