import numpy as np
from shapely.geometry import Polygon, LineString

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

def simplify_polygon_by_angle(
    polygon: Polygon,
    min_angle_deg: float = 3.0,
) -> Polygon:
    """
    Simplify a polygon by removing points whose interior angle is below
    min_angle_deg (very sharp turns).
    Applies to both the exterior ring and all holes (interior rings).

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to simplify.
    min_angle_deg : float, default=3.0
        Minimum angle in degrees. Points with a smaller interior angle are removed.

    Returns
    -------
    Polygon
        Simplified polygon (exterior and interiors treated identically).
    """

    def _calculate_interior_angle(p1, p2, p3):
        """Interior angle at point p2 between (p1-p2) and (p2-p3), in degrees (0 to 180)."""
        v1 = np.asarray(p1) - np.asarray(p2)
        v2 = np.asarray(p3) - np.asarray(p2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 180.0
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _simplify_ring_by_angle(coords, min_angle_deg):
        """Simplify a ring (exterior or interior) by removing only the small angles."""
        coords = np.asarray(coords)
        if len(coords) < 3:
            return coords
        if len(coords) > 0 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) < 3:
            return coords

        result = list(coords)
        max_iterations = len(coords) * 10

        for _ in range(max_iterations):
            if len(result) < 3:
                break
            n_before = len(result)
            n = len(result)
            i = 0
            while i < n:
                prev_idx = (i - 1) % n
                curr_idx = i
                next_idx = (i + 1) % n
                p1 = result[prev_idx]
                p2 = result[curr_idx]
                p3 = result[next_idx]
                interior_angle = _calculate_interior_angle(p1, p2, p3)
                if interior_angle < min_angle_deg:
                    result.pop(curr_idx)
                    n -= 1
                else:
                    i += 1
            if len(result) == n_before:
                break

        if len(result) < 3:
            return coords[:3] if len(coords) >= 3 else coords
        return np.array(result)

    if polygon.is_empty:
        return polygon
    if polygon.type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    # Exterior ring
    exterior_coords = np.array(polygon.exterior.coords[:-1])
    exterior_simplified = _simplify_ring_by_angle(exterior_coords, min_angle_deg)

    # Holes: same treatment as the exterior ring
    interiors_simplified = []
    for interior in polygon.interiors:
        interior_coords = np.array(interior.coords[:-1])
        ring_simplified = _simplify_ring_by_angle(interior_coords, min_angle_deg)
        if len(ring_simplified) >= 3:
            interiors_simplified.append(ring_simplified)

    return Polygon(exterior_simplified, interiors_simplified)

def _resample_ring_hfun(ring_coords, hfun, harg=()):
    """
    Helper function to resample a single ring (exterior or interior) using hfun.
    
    Parameters
    ----------
    ring_coords : ndarray of shape (N, 2)
        Coordinates of the ring vertices.
    hfun : float or callable
        Mesh-size function.
    harg : tuple, optional
        Extra arguments passed to hfun when callable.
    
    Returns
    -------
    ndarray of shape (M, 2)
        Resampled ring coordinates (closed, first point repeated at end).
    """
    polygon = np.asarray(ring_coords, dtype=float)
    n = polygon.shape[0]
    if n < 2:
        if n == 0:
            # Empty ring: return empty array (will be skipped in resample_polygon_hfun)
            return np.array([]).reshape(0, 2)
        if n == 1:
            # Single point: create a minimal valid ring (4 points)
            p = polygon[0]
            eps_ring = max(np.linalg.norm(p) * 1e-6, 1e-6)
            return np.array([
                p,
                p + np.array([eps_ring, 0]),
                p + np.array([eps_ring, eps_ring]),
                p,  # closed
            ])
        # Two points: add a third point to form a valid ring
        p1, p2 = polygon[0], polygon[1]
        mid = (p1 + p2) / 2.0
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = perp / perp_norm * np.linalg.norm(p2 - p1) * 0.1
        else:
            perp = np.array([1e-6, 0])
        p3 = mid + perp
        return np.array([p1, p2, p3, p1])  # closed

    # Build segment list and cumulative arc length for closed contour
    segs = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    nseg = len(segs)
    seg_len = np.array([np.linalg.norm(segs[i][1] - segs[i][0]) for i in range(nseg)])
    eps = np.finfo(float).eps
    seg_len = np.maximum(seg_len, eps)
    s_cum = np.concatenate([[0], np.cumsum(seg_len)])
    s_total = s_cum[-1]
    if s_total <= 0:
        # Degenerate ring: ensure at least 3 distinct points
        if n < 3:
            # Create minimal valid ring from available points
            if n == 1:
                p = polygon[0]
                eps_ring = max(np.linalg.norm(p) * 1e-6, 1e-6)
                return np.array([
                    p,
                    p + np.array([eps_ring, 0]),
                    p + np.array([eps_ring, eps_ring]),
                    p,
                ])
            elif n == 2:
                p1, p2 = polygon[0], polygon[1]
                mid = (p1 + p2) / 2.0
                perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 0:
                    perp = perp / perp_norm * np.linalg.norm(p2 - p1) * 0.1
                else:
                    perp = np.array([1e-6, 0])
                p3 = mid + perp
                return np.array([p1, p2, p3, p1])
        return np.vstack([polygon[:3], polygon[0:1]])  # At least 3 points + closure

    def s_to_xy(s):
        """Map arc length s in [0, s_total) to (x, y) on the contour."""
        s = np.clip(float(s), 0.0, s_total - 1e-12)
        i = int(np.searchsorted(s_cum[1:], s, side="right"))
        if i >= nseg:
            i = nseg - 1
        t = (s - s_cum[i]) / seg_len[i]
        a, b = np.asarray(segs[i][0]), np.asarray(segs[i][1])
        return (1 - t) * a + t * b

    def eval_h_raw(pts):
        """Evaluate hfun at pts; may contain NaN."""
        pts = np.atleast_2d(pts)
        if np.isscalar(hfun) or isinstance(hfun, (int, float, np.number)):
            return np.full(pts.shape[0], float(hfun))
        return np.asarray(hfun(pts, *harg)).ravel()[: pts.shape[0]]

    # Evaluate h at polygon vertices and replace NaN with nearest-vertex value
    h_verts = eval_h_raw(polygon).astype(float)
    nan_mask = np.isnan(h_verts)
    if np.any(nan_mask):
        tree = cKDTree(polygon)
        ok = np.where(~nan_mask)[0]
        if len(ok) == 0:
            h_verts[:] = 1.0
        else:
            for i in np.where(nan_mask)[0]:
                _, j = tree.query(polygon[i], k=1)
                j = j if np.isscalar(j) else j[0]
                if nan_mask[j]:
                    h_verts[i] = (
                        np.nanmean(h_verts[~nan_mask])
                        if np.any(~nan_mask)
                        else 1.0
                    )
                else:
                    h_verts[i] = h_verts[j]
    tree_poly = cKDTree(polygon)

    def eval_h(pts):
        """Evaluate h at pts; NaN replaced by h at nearest polygon vertex."""
        pts = np.atleast_2d(pts)
        h = eval_h_raw(pts).astype(float)
        nan_pts = np.isnan(h)
        if np.any(nan_pts):
            idx_nan = np.where(nan_pts)[0]
            _, nearest = tree_poly.query(pts[idx_nan], k=1)
            if np.ndim(nearest) == 0:
                nearest = np.array([nearest])
            h[idx_nan] = h_verts[nearest]
        return h

    # March along contour with step size h(s)
    out = [np.asarray(polygon[0], dtype=float)]
    s_current = 0.0
    h_min = np.nanmin(eval_h(polygon))
    if not np.isfinite(h_min) or h_min <= 0:
        h_min = max(s_total * 0.01, eps)
    max_pts = int(np.ceil(s_total / h_min)) + 20
    max_pts = max(max_pts, 4)

    for _ in range(max_pts):
        pt = s_to_xy(s_current)
        h_val = float(eval_h(pt.reshape(1, -1))[0])
        if not np.isfinite(h_val) or h_val <= 0:
            h_val = h_min
        step = max(h_val, s_total * 1e-10)
        s_next = s_current + step
        if s_next >= s_total:
            break
        next_pt = s_to_xy(s_next)
        out.append(next_pt)
        s_current = s_next

    node = np.array(out)
    
    # Ensure at least 3 distinct points (4 total with closure) for Shapely LinearRing
    if len(node) < 3:
        # If we have less than 3 points, create a minimal valid ring
        if len(node) == 1:
            # Single point: create a small triangle/square
            p = node[0]
            eps_ring = max(np.linalg.norm(p) * 1e-6, 1e-6)
            node = np.array([
                p,
                p + np.array([eps_ring, 0]),
                p + np.array([eps_ring, eps_ring]),
            ])
        elif len(node) == 2:
            # Two points: add a third point to form a triangle
            p1, p2 = node[0], node[1]
            mid = (p1 + p2) / 2.0
            perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 0:
                perp = perp / perp_norm * np.linalg.norm(p2 - p1) * 0.1
            else:
                perp = np.array([1e-6, 0])
            p3 = mid + perp
            node = np.array([p1, p2, p3])
    
    # Close ring: first point repeated at end
    return np.vstack([node, node[0:1]])


def _make_hfun_evaluator(reference_pts, hfun, harg=()):
    """
    Build hfun(pts) -> (M,) with NaN filled from the nearest reference vertex.
    """

    def eval_h_raw(pts):
        pts = np.atleast_2d(pts)
        if np.isscalar(hfun) or isinstance(hfun, (int, float, np.number)):
            return np.full(pts.shape[0], float(hfun))
        return np.asarray(hfun(pts, *harg)).ravel()[: pts.shape[0]]

    reference_pts = np.asarray(reference_pts, dtype=float)
    h_verts = eval_h_raw(reference_pts).astype(float)
    nan_mask = np.isnan(h_verts)
    if np.any(nan_mask):
        tree_ref = cKDTree(reference_pts)
        ok = np.where(~nan_mask)[0]
        if len(ok) == 0:
            h_verts[:] = 1.0
        else:
            for i in np.where(nan_mask)[0]:
                _, j = tree_ref.query(reference_pts[i], k=1)
                j = j if np.isscalar(j) else j[0]
                if nan_mask[j]:
                    h_verts[i] = (
                        np.nanmean(h_verts[~nan_mask])
                        if np.any(~nan_mask)
                        else 1.0
                    )
                else:
                    h_verts[i] = h_verts[j]
    tree_ref = cKDTree(reference_pts)

    def eval_h(pts):
        pts = np.atleast_2d(pts)
        h = eval_h_raw(pts).astype(float)
        nan_pts = np.isnan(h)
        if np.any(nan_pts):
            idx_nan = np.where(nan_pts)[0]
            _, nearest = tree_ref.query(pts[idx_nan], k=1)
            if np.ndim(nearest) == 0:
                nearest = np.array([nearest])
            h[idx_nan] = h_verts[nearest]
        return h

    return eval_h


def _prune_ring_by_hfun(ring_coords, hfun, harg=(), min_fraction=1.0):
    """
    Remove ring vertices whose spacing along the contour is below hfun(p).

    A vertex is removed when either adjacent edge length is shorter than
    ``min_fraction * h(p)`` at that vertex. The pass is repeated until stable.
    """
    ring = np.asarray(ring_coords, dtype=float)
    if len(ring) < 4:
        return ring

    if np.allclose(ring[0], ring[-1]):
        pts = ring[:-1].copy()
    else:
        pts = ring.copy()

    if len(pts) < 3:
        return ring

    eval_h = _make_hfun_evaluator(pts, hfun, harg)
    eps = np.finfo(float).eps
    max_iterations = max(len(pts) * 2, 1)

    for _ in range(max_iterations):
        n = len(pts)
        if n <= 3:
            break

        to_remove = np.zeros(n, dtype=bool)
        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            d_prev = np.linalg.norm(pts[i] - pts[prev_i])
            d_next = np.linalg.norm(pts[next_i] - pts[i])
            h_req = max(min_fraction * float(eval_h(pts[i : i + 1])[0]), eps)
            if d_prev < h_req or d_next < h_req:
                to_remove[i] = True

        if not np.any(to_remove):
            break

        if np.count_nonzero(~to_remove) < 3:
            scores = []
            for i in range(n):
                prev_i = (i - 1) % n
                next_i = (i + 1) % n
                d_prev = np.linalg.norm(pts[i] - pts[prev_i])
                d_next = np.linalg.norm(pts[next_i] - pts[i])
                h_req = max(min_fraction * float(eval_h(pts[i : i + 1])[0]), eps)
                scores.append((i, min(d_prev, d_next) / h_req))
            worst = min(scores, key=lambda item: item[1])[0]
            to_remove = np.zeros(n, dtype=bool)
            to_remove[worst] = True

        pts = pts[~to_remove]

    if len(pts) < 3:
        return ring

    return np.vstack([pts, pts[0:1]])


def _signed_area(ring):
    """Signed area (positive = CCW). ring: (N, 2), closed (last point = first)."""
    r = np.asarray(ring)
    if len(r) < 4:
        return 0.0
    r = r[:-1] if np.allclose(r[0], r[-1]) else r
    x, y = r[:, 0], r[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _ensure_ring_orientation(ring_coords, want_ccw):
    """In-place: reverse ring if needed so that CCW == want_ccw."""
    r = np.asarray(ring_coords)
    if len(r) < 4:
        return r
    if np.allclose(r[0], r[-1]):
        r = r[:-1]
    area = _signed_area(np.vstack([r, r[0:1]]))
    is_ccw = area > 0
    if is_ccw != want_ccw:
        r = r[::-1]
    return np.vstack([r, r[0:1]])


def resample_polygon_hfun(
    polygon,
    hfun,
    harg=(),
    verify_spacing=True,
    verify_min_fraction=1.0,
):
    """
    Resample a closed polygon so that consecutive vertices are spaced by
    approximately h(p) along the contour, where h is given by the mesh-size
    function (same contract as in pymesh2d). The input vertex density has
    minimal influence on the result. NaN values from hfun are replaced by the
    h-value at the nearest polygon vertex. Holes (interiors) are preserved.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or ndarray of shape (N, 2)
        Polygon to resample. Either a Shapely Polygon (exterior and interiors
        are resampled) or an array of vertices (x, y) in order along the boundary.
    hfun : float or callable
        Mesh-size function. If callable, must have signature hfun(pts, *harg)
        with pts of shape (M, 2), returning mesh-size values (M,) or scalar.
        If float, a constant spacing is used.
    harg : tuple, optional
        Extra arguments passed to hfun when callable.
    verify_spacing : bool, optional
        If True (default), run a final pass on each resampled ring and remove
        vertices whose adjacent edge length is shorter than ``h(p)``.
    verify_min_fraction : float, optional
        Required minimum edge length as a fraction of ``h(p)`` during the final
        verification pass. Default is 1.0 (strictly enforce hfun spacing).

    Returns
    -------
    shapely.geometry.Polygon
        Resampled polygon with exterior and holes (interiors) preserved.
        Invalid geometries are fixed with buffer(0).

    Raises
    ------
    ValueError
        If polygon is not a Shapely Polygon or an (N, 2) array.
    ImportError
        If scipy is not available (required for nearest-neighbor NaN fill).
    """
    if cKDTree is None:
        raise ImportError("resample_polygon_hfun requires scipy (scipy.spatial.cKDTree)")

    # Check if input is a Shapely Polygon with holes
    has_interiors = False
    interiors = []
    if hasattr(polygon, "exterior") and hasattr(polygon.exterior, "coords"):
        # Extract exterior
        coords = np.array(polygon.exterior.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        exterior_coords = np.asarray(coords, dtype=float)
        
        # Extract interiors (holes) if present
        if hasattr(polygon, "interiors") and len(polygon.interiors) > 0:
            has_interiors = True
            for interior in polygon.interiors:
                interior_coords = np.array(interior.coords)
                if len(interior_coords) > 1 and np.allclose(interior_coords[0], interior_coords[-1]):
                    interior_coords = interior_coords[:-1]
                interiors.append(np.asarray(interior_coords, dtype=float))
        
        polygon = exterior_coords
    else:
        polygon = np.asarray(polygon, dtype=float)

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must be a Shapely Polygon or an (N, 2) array")
    
    # Resample exterior
    exterior_ring = _resample_ring_hfun(polygon, hfun, harg)
    if verify_spacing:
        exterior_ring = _prune_ring_by_hfun(
            exterior_ring, hfun, harg, min_fraction=verify_min_fraction
        )
    exterior_ring = _ensure_ring_orientation(exterior_ring, want_ccw=True)
    
    # Ensure exterior has at least 4 points (required for LinearRing)
    if len(exterior_ring) < 4:
        # If exterior is invalid, return empty polygon
        return Polygon()
    
    # Resample interiors (holes) if present
    resampled_interiors = []
    if has_interiors:
        for interior_coords in interiors:
            resampled_interior = _resample_ring_hfun(interior_coords, hfun, harg)
            if verify_spacing:
                resampled_interior = _prune_ring_by_hfun(
                    resampled_interior,
                    hfun,
                    harg,
                    min_fraction=verify_min_fraction,
                )
            # Filter out invalid interiors (need at least 4 points for LinearRing)
            if len(resampled_interior) >= 4:
                resampled_interiors.append(resampled_interior)
            # Skip interiors with < 4 points (too small or degenerate)
    
    # Create Polygon with exterior and holes
    if len(resampled_interiors) > 0:
        poly_resampled = Polygon(exterior_ring, resampled_interiors)
    else:
        poly_resampled = Polygon(exterior_ring)
    
    if not poly_resampled.is_valid:
        poly_resampled = poly_resampled.buffer(0)
    return poly_resampled


def _resample_ring_by_spacing(xy: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a ring (closed polygon) so points are spaced at least `spacing`
    apart along the contour. Based on cumulative distance + linear interpolation.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return xy

    # Close the ring for length/interpolation purposes
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    n = len(xy)

    # Cumulative distance along the contour (includes the closing segment)
    d = np.cumsum(
        np.r_[0, np.sqrt(((np.diff(xy, axis=0)) ** 2).sum(axis=1))]
    )
    total_length = d[-1]
    if total_length <= 0:
        return xy[:1]

    # Number of points to get segments >= spacing
    n_pts = max(4, int(np.floor(total_length / spacing)))
    n_pts = min(n_pts, max(4, n - 1))

    # Regularly spaced curvilinear abscissas (without duplicating the closing point)
    d_sampled = np.linspace(0, total_length, n_pts, endpoint=False)

    # Interpolate x and y
    x_new = np.interp(d_sampled, d, xy[:, 0])
    y_new = np.interp(d_sampled, d, xy[:, 1])
    xy_interp = np.column_stack([x_new, y_new])

    # Closed ring for Shapely (first point repeated at the end)
    return np.vstack([xy_interp, xy_interp[0:1]])


def resample_polygon(
    polygon: Polygon,
    spacing: float,
) -> Polygon:
    """
    Resample the polygon: points spaced at least `spacing` apart along the
    contour (exterior and interiors). Simple cumulative-distance +
    interpolation method.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if polygon.is_empty:
        return polygon
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    # Exterior
    exterior_coords = np.asarray(polygon.exterior.coords)
    exterior_ring = _resample_ring_by_spacing(exterior_coords, spacing)
    if len(exterior_ring) < 4:
        return polygon

    # Interiors
    interiors_rings = []
    for interior in polygon.interiors:
        ring = _resample_ring_by_spacing(np.asarray(interior.coords), spacing)
        if len(ring) >= 4:
            interiors_rings.append(ring)

    poly_new = Polygon(exterior_ring, interiors_rings)
    if not poly_new.is_valid:
        poly_new = poly_new.buffer(0)
    return poly_new


def buffer_area(polygon: Polygon, area_factor: float) -> Polygon:
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
