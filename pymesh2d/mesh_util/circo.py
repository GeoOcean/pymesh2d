import warnings

import numpy as np

from ..mesh_ball.tribal2 import tribal2
from ..mesh_cost.triarea import triarea
from ..mesh_util.deltri import deltri
from .tricon import tricon

# -------------------------------- suppress runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


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
    # -------------------------------- basic checks
    if not (
        isinstance(vert, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(edge, np.ndarray)
        and isinstance(tria_6col, np.ndarray)
    ):
        raise TypeError("circumcenters:incorrectInputClass")

    if vert.ndim != 2 or edge.ndim != 2 or tria_6col.ndim != 2:
        raise ValueError("circumcenters:incorrectDimensions")

    # -------------------------------- identify boundary edges
    # edge[:, 3] == -1 means no T2, so it's a boundary edge
    boundary_edges = edge[:, 3] == -1

    # -------------------------------- get edge indices for each triangle
    # columns 3-5 contain edge indices
    edge_indices = tria_6col[:, 3:6]

    # -------------------------------- count interior edges per triangle
    nboundary_edges = np.sum(boundary_edges[edge_indices], axis=1)
    numberOfInteriorEdges = 3 - nboundary_edges  # 3 edges per triangle

    # -------------------------------- initialize output array
    # same format as tribal2: [cx, cy, r^2]
    bb = np.zeros((tria_6col.shape[0], 3))

    # -------------------------------- calculate all circumcenters
    tria_3col = tria if tria.shape[1] == 3 else tria_6col[:, :3]
    bb_all = tribal2(vert, tria_3col)

    # -------------------------------- process each triangle
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
    # -------------------------------- compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    # -------------------------------- compute dot products
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)

    # -------------------------------- compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # -------------------------------- check if point is in triangle
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
    # -------------------------------- line segment intersection
    # using parametric equations
    d1 = p2 - p1
    d2 = p4 - p3

    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None  # Parallel lines

    diff = p3 - p1
    t1 = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    t2 = (diff[0] * d1[1] - diff[1] * d1[0]) / denom

    # -------------------------------- check if intersection is within both segments
    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return p1 + t1 * d1

    return None


def small_flow_links(vert, tria, edge, removesmalllinkstrsh, conn=None, tria_6col=None, return_circumcenters=False):
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
    TRIA_6COL : ndarray, optional
        Precomputed 6-column triangle connectivity from tricon. If provided,
        avoids a second call to tricon when tria has 3 columns.
    RETURN_CIRCUMCENTERS : bool, optional
        If True, return circumcenters positions (V, 2) as third output for reuse.

    Returns
    -------
    NLINKTOOSMALL : int
        Number of small flow links identified.
    SMALL_LINK_INDICES : ndarray
        Array of edge indices that are too small.
    XZ : ndarray, optional
        Circumcenter positions (T, 2), only if return_circumcenters=True.

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
    # -------------------------------- basic checks
    if tria.size == 0 or edge.size == 0:
        if return_circumcenters:
            return 0, np.array([], dtype=int), np.empty((0, 2))
        return 0, np.array([], dtype=int)

    # -------------------------------- get tria in 6-column format (reuse if provided)
    if tria_6col is not None:
        pass  # use caller's tria_6col
    elif tria.shape[1] == 3:
        _, tria_6col = tricon(tria, conn if conn is not None else np.empty((0, 2), dtype=int))
    else:
        tria_6col = tria

    # -------------------------------- calculate circumcenters
    bb = circumcenters(vert, tria, edge, tria_6col)
    xz = bb[:, 0:2]
    tria_3col = tria if tria.shape[1] == 3 else tria_6col[:, :3]
    ba = np.abs(triarea(vert, tria_3col))

    # -------------------------------- find edges connecting two triangles
    # Delft3D checks all 2D links (KN(3, L) == 2), which includes constrained edges
    # MeshKernel also checks all edges with two faces
    internal_mask = (edge[:, 2] >= 0) & (edge[:, 3] >= 0)

    if np.sum(internal_mask) == 0:
        if return_circumcenters:
            return 0, np.array([], dtype=int), xz
        return 0, np.array([], dtype=int)

    t1_indices = edge[internal_mask, 2].astype(int)
    t2_indices = edge[internal_mask, 3].astype(int)
    internal_edge_indices = np.where(internal_mask)[0]

    # -------------------------------- filter valid indices
    # MeshKernel checks ALL edges with two faces, regardless of boundary status or area
    valid_mask = (
        (t1_indices >= 0) & (t1_indices < len(ba)) &
        (t2_indices >= 0) & (t2_indices < len(ba))
    )

    if not np.any(valid_mask):
        if return_circumcenters:
            return 0, np.array([], dtype=int), xz
        return 0, np.array([], dtype=int)

    t1_indices = t1_indices[valid_mask]
    t2_indices = t2_indices[valid_mask]
    internal_edge_indices = internal_edge_indices[valid_mask]

    # -------------------------------- calculate thresholds and distances
    # Match Delft3D-FM formula: dxlim = 0.9 * removesmalllinkstrsh * 0.5 * (sqrt(ba(n1)) + sqrt(ba(n2)))
    sqrt_ba = np.sqrt(ba[t1_indices]) + np.sqrt(ba[t2_indices])
    dxlim = 0.9 * removesmalllinkstrsh * 0.5 * sqrt_ba
    dxlink = np.linalg.norm(xz[t2_indices] - xz[t1_indices], axis=1)

    # -------------------------------- identify too small links
    # Match Delft3D-FM: if (dxlink < dxlim) - no additional conditions
    too_small = dxlink < dxlim
    nlinktoosmall = np.sum(too_small)

    if nlinktoosmall > 0:
        small_link_indices = internal_edge_indices[too_small]
    else:
        small_link_indices = np.array([], dtype=int)

    if return_circumcenters:
        return nlinktoosmall, small_link_indices, xz
    return nlinktoosmall, small_link_indices


def small_flow_centers(vert, tria, edge, removesmalllinkstrsh, conn=None):
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
    # -------------------------------- basic checks
    if tria.size == 0 or edge.size == 0:
        return np.array([]).reshape(0, 2), 0

    # -------------------------------- identify small flow links
    nlinktoosmall, small_link_indices = small_flow_links(
        vert, tria, edge, removesmalllinkstrsh, conn
    )
    if nlinktoosmall == 0:
        return np.array([]).reshape(0, 2), 0

    # -------------------------------- get tria in 6-column format
    if tria.shape[1] == 3:
        _, tria_6col = tricon(tria, conn if conn is not None else np.empty((0, 2), dtype=int))
    else:
        tria_6col = tria

    # -------------------------------- calculate circumcenters
    bb = circumcenters(vert, tria, edge, tria_6col)
    xz = bb[:, 0:2]

    # -------------------------------- get problematic triangles
    problem_edges = edge[small_link_indices]
    all_problem_triangles = np.unique(
        np.concatenate([
            problem_edges[:, 2].astype(int),
            problem_edges[:, 3].astype(int),
        ])
    )

    if len(all_problem_triangles) == 0:
        return np.array([]).reshape(0, 2), nlinktoosmall

    # -------------------------------- return circumcenters
    # no exclusion of boundary triangles
    return xz[all_problem_triangles], nlinktoosmall


def _is_edge_constrained(conn, va, vb):
    """Return True if edge (va, vb) or (vb, va) is in conn."""
    if conn is None or conn.size == 0:
        return False
    c = conn[:, :2]
    return np.any(((c[:, 0] == va) & (c[:, 1] == vb)) | ((c[:, 0] == vb) & (c[:, 1] == va)))


def _signed_area_one(vert, a, b, c):
    """Signed area of triangle (a, b, c); positive = CCW."""
    ev12 = vert[b] - vert[a]
    ev13 = vert[c] - vert[a]
    return 0.5 * (ev12[0] * ev13[1] - ev12[1] * ev13[0])


def try_flip_small_flow_edges(vert, tria, conn, edge, small_link_indices):
    """
    Try to fix small flow links by flipping the shared edge (swap diagonal of the quad).
    Merges the two triangles into a different pair without adding/removing vertices,
    often increasing the flow link length. Only flips if the quad is strictly convex
    and the shared edge is not constrained.

    Returns
    -------
    n_flipped : int
        Number of edges that were flipped (tria is modified in place).
    """
    if small_link_indices.size == 0:
        return 0
    problem_edges = edge[small_link_indices]
    t1_indices = problem_edges[:, 2].astype(int)
    t2_indices = problem_edges[:, 3].astype(int)
    edge_vertices = problem_edges[:, 0:2].astype(int)
    n_vert = vert.shape[0]
    n_tria = tria.shape[0]
    n_flipped = 0
    for idx in range(len(small_link_indices)):
        t1_idx, t2_idx = t1_indices[idx], t2_indices[idx]
        if t1_idx < 0 or t1_idx >= n_tria or t2_idx < 0 or t2_idx >= n_tria:
            continue
        shared_v1, shared_v2 = edge_vertices[idx, 0], edge_vertices[idx, 1]
        if _is_edge_constrained(conn, shared_v1, shared_v2):
            continue
        tri1_verts = tria[t1_idx, :]
        tri2_verts = tria[t2_idx, :]
        opp1 = None
        for v in tri1_verts:
            if v != shared_v1 and v != shared_v2:
                opp1 = v
                break
        opp2 = None
        for v in tri2_verts:
            if v != shared_v1 and v != shared_v2:
                opp2 = v
                break
        if opp1 is None or opp2 is None or opp1 == opp2:
            continue
        # Convex quad: shared_v1 and shared_v2 on opposite sides of line (opp1, opp2)
        sa1 = _signed_area_one(vert, opp1, opp2, shared_v1)
        sa2 = _signed_area_one(vert, opp1, opp2, shared_v2)
        if sa1 * sa2 >= 0:
            continue
        # Both new triangles must have positive area (CCW)
        if sa1 <= 0:
            continue
        # New triangles: (opp1, opp2, shared_v1) and (opp2, opp1, shared_v2)
        tria[t1_idx, :] = [opp1, opp2, shared_v1]
        tria[t2_idx, :] = [opp2, opp1, shared_v2]
        n_flipped += 1
        # One flip per call so connectivity stays consistent; next iteration will re-run tricon
        break
    return n_flipped


def correct_small_flow_links(vert, tria, edge, small_link_indices, circumcenters_pos, 
                              removesmalllinkstrsh, conn=None, aggressive_mode=False):
    """
    Compute vertex displacements and constrained triangle splits to correct small flow links.
    
    This function analyzes problematic flow links and determines:
    1. Vertex displacements needed to separate circumcenters
    2. Constrained triangles that need node insertion
    
    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        XY coordinates of the vertices in the triangulation.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices.
    EDGE : ndarray of shape (E, 5)
        Edge connectivity array from tricon: [V1, V2, T1, T2, CE].
    SMALL_LINK_INDICES : ndarray
        Array of edge indices that are too small.
    CIRCUMCENTERS_POS : ndarray of shape (T, 2)
        XY coordinates of circumcenters for each triangle.
    REMOVESMALLLINKSTRSH : float, optional
        Threshold for removing small links (default 0.1).
    CONN : ndarray, optional
        Constrained edges (vertices that cannot be moved).
    AGGRESSIVE_MODE : int, optional
        Aggressive mode level:
        - 0: Normal mode (default)
        - 1: Aggressive mode (5x displacement) when few links remain
        - 2: Very aggressive mode (10x displacement) for single remaining link
    
    Returns
    -------
    VERTEX_DISPLACEMENTS : ndarray of shape (V, 2)
        Array of displacement vectors for each vertex (zero if no displacement needed).
    VERTEX_COUNTS : ndarray of shape (V,)
        Array of contribution counts for each vertex (zero if no contribution).
    CONSTRAINED_TRIANGLES_TO_SPLIT : set
        Set of triangle indices that need node insertion.
    NEW_VERTICES : list
        List of new vertex coordinates to insert (barycenters of constrained triangles).
    
    Notes
    -----
    - For triangles with all constrained vertices, a new node is inserted at the barycenter.
    - For triangles with free vertices, displacements are computed to separate circumcenters.
    - Displacement magnitude is more aggressive when only one vertex is free.
    """
    from ..mesh_cost.triarea import triarea
    
    # -------------------------------- initialize outputs
    n_vert = vert.shape[0]
    vertex_displacements = np.zeros((n_vert, 2), dtype=np.float64)
    vertex_counts = np.zeros(n_vert, dtype=np.int32)
    constrained_triangles_to_split = set()
    
    # -------------------------------- get constrained vertices
    if conn is not None and conn.size > 0:
        conn_array = np.zeros(n_vert, dtype=bool)
        conn_array[conn.flatten()] = True
    else:
        conn_array = np.zeros(n_vert, dtype=bool)
    
    # -------------------------------- compute triangle areas
    tri_areas = np.abs(triarea(vert, tria))
    
    # -------------------------------- process each problematic link
    # CRITICAL: Verify edge indices are valid FIRST
    if len(small_link_indices) == 0:
        return vertex_displacements, vertex_counts, constrained_triangles_to_split, []
    
    valid_edge_mask = (
        (small_link_indices >= 0) & (small_link_indices < len(edge))
    )
    if not np.any(valid_edge_mask):
        # No valid edge indices - skip processing
        return vertex_displacements, vertex_counts, constrained_triangles_to_split, []
    
    # Filter to only valid edge indices
    valid_small_link_indices = small_link_indices[valid_edge_mask]
    problem_edges = edge[valid_small_link_indices]
    t1_indices = problem_edges[:, 2].astype(int)
    t2_indices = problem_edges[:, 3].astype(int)
    edge_vertices = problem_edges[:, 0:2].astype(int)
    
    valid_mask = (
        (t1_indices >= 0) & (t1_indices < len(tria)) &
        (t2_indices >= 0) & (t2_indices < len(tria)) &
        (edge_vertices[:, 0] >= 0) & (edge_vertices[:, 0] < len(vert)) &
        (edge_vertices[:, 1] >= 0) & (edge_vertices[:, 1] < len(vert))
    )
    
    if not np.any(valid_mask):
        # No valid triangles to process
        return vertex_displacements, vertex_counts, constrained_triangles_to_split, []
    
    for idx in np.where(valid_mask)[0]:
        t1_idx, t2_idx = t1_indices[idx], t2_indices[idx]
        shared_v1, shared_v2 = edge_vertices[idx, 0], edge_vertices[idx, 1]
        
        # CRITICAL: Verify triangle indices are still valid
        if t1_idx >= len(tria) or t2_idx >= len(tria) or t1_idx < 0 or t2_idx < 0:
            continue
        
        # CRITICAL: Verify circumcenters array matches triangle count
        if t1_idx >= circumcenters_pos.shape[0] or t2_idx >= circumcenters_pos.shape[0]:
            continue
        
        # Identify opposite vertices (vertices not on the shared edge)
        tri1_verts, tri2_verts = tria[t1_idx, :], tria[t2_idx, :]
        
        # CRITICAL: Verify shared vertices are actually in both triangles
        if shared_v1 not in tri1_verts or shared_v2 not in tri1_verts:
            continue
        if shared_v1 not in tri2_verts or shared_v2 not in tri2_verts:
            continue
        
        # Find opposite vertex in triangle 1 (vertex not on the shared edge)
        opp1 = None
        for v in tri1_verts:
            if v != shared_v1 and v != shared_v2:
                opp1 = v
                break
        
        # Find opposite vertex in triangle 2 (vertex not on the shared edge)
        opp2 = None
        for v in tri2_verts:
            if v != shared_v1 and v != shared_v2:
                opp2 = v
                break
        
        if opp1 is None or opp2 is None:
            continue
        
        # CRITICAL: Verify vertex indices are valid
        if opp1 >= len(vert) or opp2 >= len(vert) or opp1 < 0 or opp2 < 0:
            continue
        
        # Get circumcenters and compute link distance
        cc1, cc2 = circumcenters_pos[t1_idx], circumcenters_pos[t2_idx]
        cc_diff = cc2 - cc1
        link_distance = max(np.linalg.norm(cc_diff), 1e-12)
        
        # Compute threshold
        area1, area2 = max(tri_areas[t1_idx], 1e-12), max(tri_areas[t2_idx], 1e-12)
        sqrt_ba = np.sqrt(area1) + np.sqrt(area2)
        if sqrt_ba < 1e-12:
            continue
        
        dxlim = 0.9 * removesmalllinkstrsh * 0.5 * sqrt_ba
        if link_distance >= dxlim:
            continue
        
        # Compute how much distance we need to add
        needed_distance = max(dxlim - link_distance, 0.0)
        
        # -------------------------------- check if vertices are free
        opp1_free = not conn_array[opp1]
        opp2_free = not conn_array[opp2]
        
        if not opp1_free and not opp2_free:
            # -------------------------------- all opposite vertices constrained
            # Mark smaller triangle for node insertion (simpler, more reliable)
            if area1 < area2:
                constrained_triangles_to_split.add(t1_idx)
            else:
                constrained_triangles_to_split.add(t2_idx)
            continue
        
        # -------------------------------- IMPROVED STRATEGY: Targeted displacement
        # Focus on moving ONLY the vertices directly involved in the problematic link
        # Strategy: Move opposite vertices away from each other along the circumcenter separation line
        
        # Compute separation direction (from cc1 to cc2)
        if link_distance < 1e-12:
            # Circumcenters coincide - use perpendicular to shared edge
            edge_vec = vert[shared_v2] - vert[shared_v1]
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-12:
                continue
            cc_dir = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        else:
            cc_dir = cc_diff / link_distance
        
        # Compute displacement magnitude - conservative and targeted
        if aggressive_mode:
            # More aggressive for few/single links, but controlled to avoid affecting other triangles
            aggressive_factor = 6.0 if aggressive_mode == 2 else 3.0  # Reduced from 8.0/4.0
        else:
            aggressive_factor = 1.0
        
        # Base magnitude: enough to achieve the needed separation
        # Limit to reasonable fraction of triangle size to avoid affecting neighbors
        max_displacement = min(np.sqrt(area1), np.sqrt(area2)) * 0.5
        base_magnitude = min(needed_distance * aggressive_factor, max_displacement)
        
        # Apply displacements ONLY to opposite vertices (not edge vertices)
        # This keeps modifications localized to the problematic triangles
        if opp1_free and opp2_free:
            # Both free: move them apart symmetrically along separation line
            displacement_magnitude = base_magnitude
            disp_vec1 = -cc_dir * displacement_magnitude * 0.5  # Move opp1 away from cc2
            disp_vec2 = cc_dir * displacement_magnitude * 0.5   # Move opp2 away from cc1
        elif opp1_free:
            # Only opp1 free: move it more aggressively
            displacement_magnitude = base_magnitude * 1.5
            disp_vec1 = -cc_dir * displacement_magnitude
            disp_vec2 = np.zeros(2)
        elif opp2_free:
            # Only opp2 free: move it more aggressively
            displacement_magnitude = base_magnitude * 1.5
            disp_vec1 = np.zeros(2)
            disp_vec2 = cc_dir * displacement_magnitude
        else:
            # Should not happen (already handled above)
            continue
        
        # -------------------------------- accumulate displacements
        # CRITICAL: Verify vertex indices are within bounds before accessing
        if opp1_free and 0 <= opp1 < len(vertex_displacements):
            vertex_displacements[opp1] += disp_vec1
            vertex_counts[opp1] += 1
        
        if opp2_free and 0 <= opp2 < len(vertex_displacements):
            vertex_displacements[opp2] += disp_vec2
            vertex_counts[opp2] += 1
    
    # -------------------------------- compute new vertices for constrained triangles
    new_vertices = []
    for tri_idx in constrained_triangles_to_split:
        # CRITICAL: Verify triangle index is valid
        if 0 <= tri_idx < len(tria):
            tri_verts = tria[tri_idx, :]
            # CRITICAL: Verify all vertex indices are valid
            if np.all((0 <= tri_verts) & (tri_verts < len(vert))):
                center = np.mean(vert[tri_verts, :], axis=0)
                if np.all(np.isfinite(center)):
                    new_vertices.append(center)
    
    return vertex_displacements, vertex_counts, constrained_triangles_to_split, new_vertices


def fix_small_flow_links(vert, conn, tria, tnum, node, PSLG, part, opts):
    """
    Fix all small flow links in the mesh through iterative correction.
    
    This function iteratively identifies and corrects small flow links by:
    1. Moving free vertices to separate circumcenters
    2. Inserting nodes in constrained triangles
    3. Deleting problematic triangles if stagnation occurs
    
    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        XY coordinates of the vertices.
    CONN : ndarray of shape (E, 2)
        Constrained edges.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices.
    TNUM : ndarray of shape (T,)
        Array of part indices.
    NODE : ndarray
        Node coordinates (for deltri).
    PSLG : ndarray
        Planar Straight Line Graph (for deltri).
    PART : dict
        Partition information (for deltri).
    OPTS : dict
        Options dictionary with 'removesmalllinkstrsh' and 'disp' keys.
    
    Returns
    -------
    VERT : ndarray
        Modified vertex coordinates.
    CONN : ndarray
        Modified constrained edges.
    TRIA : ndarray
        Modified triangles.
    TNUM : ndarray
        Modified part indices.
    """
    from ..mesh_cost.triarea import triarea
    from .tricon import tricon
    
    max_fix_iter = opts.get("max_fix_iter", 100)
    removesmalllinkstrsh = opts.get("removesmalllinkstrsh")
    fix_iter = 0
    stagnation_count = 0
    
    # Adaptive strategy: be more aggressive when there are few small links remaining
    initial_nlinks = None
    
    # Optimize: reduce max iterations when few links remain
    adaptive_max_iter = max_fix_iter
    
    for fix_iter in range(adaptive_max_iter):
        # -------------------------------- rebuild connectivity (single tricon)
        # CRITICAL: Always rebuild connectivity to ensure indices are current
        edge_fix, tria_6col_fix = tricon(tria, conn)

        # -------------------------------- check for small flow links (reuse tria_6col, get circumcenters)
        # CRITICAL: Always recalculate to ensure circumcenters match current mesh
        nlinktoosmall, small_link_indices, circumcenters_pos = small_flow_links(
            vert, tria, edge_fix, removesmalllinkstrsh, conn,
            tria_6col=tria_6col_fix, return_circumcenters=True
        )

        if nlinktoosmall == 0:
            break
        
        # Track initial number for adaptive strategy
        if initial_nlinks is None:
            initial_nlinks = nlinktoosmall
            # Reduce max iterations if starting with few links
            if initial_nlinks <= 5:
                adaptive_max_iter = min(max_fix_iter, 30)
        
        # Adaptive strategy: more aggressive when few links remain
        few_links_remaining = nlinktoosmall <= max(3, initial_nlinks * 0.1)
        single_link_remaining = nlinktoosmall == 1  # Very aggressive for last link
        
        # Don't exit early if single link remains - keep trying
        if few_links_remaining and not single_link_remaining and fix_iter > 10 and stagnation_count >= 2:
            # Try one more aggressive pass, then exit (but not for single link)
            if fix_iter < adaptive_max_iter - 1:
                continue
            else:
                break
        
        # -------------------------------- Verify indices are valid before processing
        if len(small_link_indices) > 0:
            valid_edge_mask = (
                (small_link_indices >= 0) & 
                (small_link_indices < len(edge_fix))
            )
            if not np.all(valid_edge_mask):
                # Some edge indices are invalid - skip this iteration
                stagnation_count += 1
                continue

        # -------------------------------- try to merge by edge flip (unify pair without creating small elements)
        n_flipped = try_flip_small_flow_edges(vert, tria, conn, edge_fix, small_link_indices)
        if n_flipped > 0:
            stagnation_count = 0
            continue

        # -------------------------------- delete problematic triangles if stagnation
        # IMPROVED STRATEGY: Only delete triangles that are actually problematic
        # More aggressive deletion when few links remain, very aggressive for single link
        if single_link_remaining:
            stagnation_threshold = 1  # Delete immediately if stagnation with single link
            early_delete_threshold = 1  # Delete after 1 iteration if single link
        elif few_links_remaining:
            stagnation_threshold = 2
            early_delete_threshold = 3
        else:
            stagnation_threshold = 3
            early_delete_threshold = 5
        
        if stagnation_count >= stagnation_threshold or (nlinktoosmall <= 3 and fix_iter > early_delete_threshold):
            problem_edges = edge_fix[small_link_indices]
            t1_indices = problem_edges[:, 2].astype(int)
            t2_indices = problem_edges[:, 3].astype(int)
            
            # Verify edge indices are valid
            valid_edge_idx = (
                (small_link_indices >= 0) & (small_link_indices < len(edge_fix))
            )
            if not np.any(valid_edge_idx):
                stagnation_count += 1
                continue
            
            problem_edges_valid = problem_edges[valid_edge_idx]
            t1_indices = problem_edges_valid[:, 2].astype(int)
            t2_indices = problem_edges_valid[:, 3].astype(int)
            
            problem_triangles = set()
            valid_mask = (
                (t1_indices >= 0) & (t1_indices < len(tria)) &
                (t2_indices >= 0) & (t2_indices < len(tria))
            )
            
            if not np.any(valid_mask):
                stagnation_count += 1
                continue
            
            # Compute triangle areas and link distances to identify truly problematic triangles
            tri_areas_simple = np.abs(triarea(vert, tria))
            
            # Recompute link distances for problematic edges to verify they're still small
            for idx in np.where(valid_mask)[0]:
                t1_idx, t2_idx = t1_indices[idx], t2_indices[idx]
                
                # Verify triangle indices
                if t1_idx >= len(circumcenters_pos) or t2_idx >= len(circumcenters_pos):
                    continue
                
                # Check if link is still too small
                cc1, cc2 = circumcenters_pos[t1_idx], circumcenters_pos[t2_idx]
                link_dist = np.linalg.norm(cc2 - cc1)
                area1, area2 = tri_areas_simple[t1_idx], tri_areas_simple[t2_idx]
                sqrt_ba = np.sqrt(max(area1, 1e-12)) + np.sqrt(max(area2, 1e-12))
                dxlim = 0.9 * removesmalllinkstrsh * 0.5 * sqrt_ba
                
                # Only mark as problematic if link is still too small
                if link_dist < dxlim:
                    # For single link remaining, be more aggressive - delete both triangles if needed
                    if single_link_remaining:
                        # Delete the smaller triangle, or both if areas are similar
                        if area1 < area2 * 0.8:  # t1 significantly smaller
                            problem_triangles.add(t1_idx)
                        elif area2 < area1 * 0.8:  # t2 significantly smaller
                            problem_triangles.add(t2_idx)
                        else:
                            # Similar areas - delete both to force retriangulation
                            problem_triangles.add(t1_idx)
                            problem_triangles.add(t2_idx)
                    else:
                        # Normal strategy: prefer deleting the smaller triangle
                        if area1 < area2:
                            problem_triangles.add(t1_idx)
                        elif area2 < area1:
                            problem_triangles.add(t2_idx)
                        else:
                            # Equal areas - delete the one with smaller link distance contribution
                            problem_triangles.add(t1_idx if link_dist < dxlim * 0.5 else t2_idx)
            
            if len(problem_triangles) > 0:
                problem_triangles = np.array(list(problem_triangles))
                # Only delete if we have a reasonable number (not too many at once)
                if len(problem_triangles) <= max(5, nlinktoosmall * 2):
                    keep_triangles = np.ones(len(tria), dtype=bool)
                    keep_triangles[problem_triangles] = False
                    tria = tria[keep_triangles, :]
                    if len(tnum) == len(keep_triangles):
                        tnum = tnum[keep_triangles]
                    vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
                    tria = tria[:, 0:3]
                    stagnation_count = 0
                    continue
                else:
                    # Too many triangles to delete - be more selective
                    # Delete only the smallest ones
                    problem_areas = tri_areas_simple[problem_triangles]
                    sorted_idx = np.argsort(problem_areas)
                    n_to_delete = min(len(problem_triangles), max(3, nlinktoosmall))
                    triangles_to_delete = problem_triangles[sorted_idx[:n_to_delete]]
                    
                    keep_triangles = np.ones(len(tria), dtype=bool)
                    keep_triangles[triangles_to_delete] = False
                    tria = tria[keep_triangles, :]
                    if len(tnum) == len(keep_triangles):
                        tnum = tnum[keep_triangles]
                    vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
                    tria = tria[:, 0:3]
                    stagnation_count = 0
                    continue

        # -------------------------------- circumcenters already from small_flow_links
        if not np.all(np.isfinite(circumcenters_pos).all(axis=1)):
            stagnation_count += 1
            continue

        # -------------------------------- compute displacements and triangles to split
        # CRITICAL: Ensure circumcenters_pos matches current tria size
        if circumcenters_pos.shape[0] != len(tria):
            # Circumcenters don't match current triangle count - recalculate
            _, _, circumcenters_pos = small_flow_links(
                vert, tria, edge_fix, removesmalllinkstrsh, conn,
                tria_6col=tria_6col_fix, return_circumcenters=True
            )
        
        # Pass aggressive mode: 2 for single link, 1 for few links, 0 otherwise
        aggressive_mode_value = 2 if single_link_remaining else (1 if few_links_remaining else 0)
        vertex_displacements, vertex_counts, constrained_triangles_to_split, new_vertices = \
            correct_small_flow_links(
                vert, tria, edge_fix, small_link_indices, circumcenters_pos,
                removesmalllinkstrsh, conn, aggressive_mode=aggressive_mode_value
            )
        
        # -------------------------------- insert nodes at center of fully constrained triangles
        if len(new_vertices) > 0:
            vert = np.vstack([vert, np.array(new_vertices)])
            vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
            tria = tria[:, 0:3]
            stagnation_count = 0
            continue
        
        # -------------------------------- apply displacements
        # IMPROVED STRATEGY: More conservative and targeted displacement application
        has_displacement = vertex_counts > 0
        if np.any(has_displacement):
            # -------------------------------- create conn_array for this iteration
            n_vert = len(vert)
            conn_array = np.zeros(n_vert, dtype=bool)
            if conn.size > 0:
                conn_indices = conn.flatten()
                valid_conn = (conn_indices >= 0) & (conn_indices < n_vert)
                conn_array[conn_indices[valid_conn]] = True
            
            # -------------------------------- filter unconstrained vertices
            # Only apply to vertices that have displacements AND are free
            free_mask = has_displacement[:n_vert] & ~conn_array
            v_indices = np.where(free_mask)[0]
            
            if len(v_indices) > 0:
                # -------------------------------- IMPROVED: Adaptive relaxation factor
                # More aggressive when few links remain, very aggressive for single link
                if single_link_remaining:
                    base_relaxation = 2.0 + 0.3 * fix_iter
                    relaxation = min(base_relaxation, 5.0)  # Very high cap for single link
                elif few_links_remaining:
                    base_relaxation = 1.0 + 0.2 * fix_iter
                    relaxation = min(base_relaxation, 3.0)  # Higher cap for few links
                else:
                    base_relaxation = 0.5 + 0.1 * fix_iter
                    relaxation = min(base_relaxation, 2.0)  # Cap at 2.0 for stability
                
                # -------------------------------- apply displacements (vectorized)
                # Average multiple contributions if vertex is involved in multiple small links
                counts = vertex_counts[v_indices]
                counts = np.maximum(counts, 1)
                
                # Normalize displacements by count and apply relaxation
                normalized_displacements = vertex_displacements[v_indices] / counts[:, None]
                
                # Limit displacement magnitude to avoid creating new problems
                # Use a conservative cap based on displacement magnitudes
                disp_magnitudes = np.linalg.norm(normalized_displacements, axis=1)
                
                # Conservative cap: use median of non-zero displacements as reference
                # This prevents outliers from causing excessive modifications
                if np.any(disp_magnitudes > 0):
                    median_disp = np.median(disp_magnitudes[disp_magnitudes > 0])
                    # Cap at 2x median to allow some variation but prevent extreme values
                    max_allowed_disp = max(median_disp * 2.0, 1e-6)
                else:
                    max_allowed_disp = 1e-6
                
                # Clip displacements that are too large relative to typical displacement
                scale_factor = np.minimum(1.0, max_allowed_disp / (disp_magnitudes + 1e-12))
                normalized_displacements = normalized_displacements * scale_factor[:, None]
                
                # Apply with relaxation
                vert[v_indices] += normalized_displacements * relaxation
            
            vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
            tria = tria[:, 0:3]
            stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count >= 2:
                continue
    
    # -------------------------------- final verification
    edge_final, _ = tricon(tria, conn)
    nlinktoosmall_final, _ = small_flow_links(
        vert, tria, edge_final, removesmalllinkstrsh, conn
    )
    
    return vert, conn, tria, tnum
