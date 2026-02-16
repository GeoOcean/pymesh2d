import numpy as np
from shapely.geometry import Polygon, LineString

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def resample_polygon_hfun(polygon, hfun, harg=()):
    """
    Resample a closed polygon so that consecutive vertices are spaced by
    approximately h(p) along the contour, where h is given by the mesh-size
    function (same contract as in pymesh2d). The input vertex density has
    minimal influence on the result. NaN values from hfun are replaced by the
    h-value at the nearest polygon vertex.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or ndarray of shape (N, 2)
        Polygon to resample. Either a Shapely Polygon (exterior ring is used)
        or an array of vertices (x, y) in order along the boundary.
    hfun : float or callable
        Mesh-size function. If callable, must have signature hfun(pts, *harg)
        with pts of shape (M, 2), returning mesh-size values (M,) or scalar.
        If float, a constant spacing is used.
    harg : tuple, optional
        Extra arguments passed to hfun when callable.

    Returns
    -------
    shapely.geometry.Polygon
        Resampled polygon (exterior only; no holes). Invalid geometries are
        fixed with buffer(0).

    Raises
    ------
    ValueError
        If polygon is not a Shapely Polygon or an (N, 2) array.
    ImportError
        If scipy is not available (required for nearest-neighbor NaN fill).
    """
    if cKDTree is None:
        raise ImportError("resample_polygon_hfun requires scipy (scipy.spatial.cKDTree)")

    # Extract (N, 2) vertex array from Shapely Polygon or array input
    if hasattr(polygon, "exterior") and hasattr(polygon.exterior, "coords"):
        coords = np.array(polygon.exterior.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        polygon = np.asarray(coords, dtype=float)
    else:
        polygon = np.asarray(polygon, dtype=float)

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must be a Shapely Polygon or an (N, 2) array")
    n = polygon.shape[0]
    if n < 2:
        if n == 0:
            return Polygon()
        if n == 1:
            return Polygon([polygon[0], polygon[0], polygon[0], polygon[0]])
        # Two points: close ring for Shapely
        return Polygon(np.vstack([polygon, polygon[0:1]]))

    # Build segment list and cumulative arc length for closed contour
    segs = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    nseg = len(segs)
    seg_len = np.array([np.linalg.norm(segs[i][1] - segs[i][0]) for i in range(nseg)])
    eps = np.finfo(float).eps
    seg_len = np.maximum(seg_len, eps)
    s_cum = np.concatenate([[0], np.cumsum(seg_len)])
    s_total = s_cum[-1]
    if s_total <= 0:
        return Polygon(polygon[:1])

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
    # Close ring for Shapely: first point repeated at end
    exterior_ring = np.vstack([node, node[0:1]])
    poly_resampled = Polygon(exterior_ring)
    if not poly_resampled.is_valid:
        poly_resampled = poly_resampled.buffer(0)
    return poly_resampled


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

    def resample_line(coords, spacing):
        coords = np.asarray(coords)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])  # close ring if open
        dists = np.cumsum(np.r_[0, np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(1))])
        if dists[-1] == 0:
            return coords
        new_d = np.arange(0, dists[-1], spacing)
        x = np.interp(new_d, dists, coords[:, 0])
        y = np.interp(new_d, dists, coords[:, 1])
        return np.c_[x, y]

    # ---- Exterior ----
    exterior = np.asarray(polygon.exterior.coords)
    exterior_resampled = resample_line(exterior, spacing)

    if len(exterior_resampled) < 4:
        raise ValueError("Exterior ring too short to form a polygon")

    # ---- Interiors ----
    interiors_resampled = []
    for interior in polygon.interiors:
        ring = np.asarray(interior.coords)
        ring_resampled = resample_line(ring, spacing)
        if len(ring_resampled) >= 4:
            interiors_resampled.append(ring_resampled)

    # ---- Construct polygon safely ----
    poly_new = Polygon(exterior_resampled, interiors_resampled)

    # ---- Fix geometry if invalid (self-intersection, etc.) ----
    if not poly_new.is_valid:
        poly_new = poly_new.buffer(0)

    return poly_new

def resample_polygon_iterate(polygon: Polygon, spacing: float) -> Polygon:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)
    
    def resample_ring_optimal(ring_coords, min_spacing):
        coords = np.asarray(ring_coords, dtype=np.float64)
        
        if len(coords) > 1 and np.allclose(coords[0], coords[-1], rtol=1e-10):
            coords = coords[:-1]
        
        if len(coords) < 2:
            return coords
        
        line = LineString(coords)
        total_length = line.length
        
        # if total_length < min_spacing:
        #     if len(coords) >= 4:
        #         return coords
        #     else:
        #         return np.vstack([coords, coords[0]])
        
        max_segments = int(np.floor(total_length / min_spacing))
        
        if max_segments < 1:
            max_segments = 1
        
        actual_spacing = total_length / max_segments
        
        n_points = max_segments + 1
        distances = np.linspace(0, total_length, n_points, endpoint=True)
        
        new_coords = np.array([line.interpolate(d).coords[0] for d in distances])
        
        if not np.allclose(new_coords[0], new_coords[-1], rtol=1e-10):
            new_coords[-1] = new_coords[0]
        
        diffs = np.diff(new_coords, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        if len(segment_lengths) > 0:
            min_seg_length = np.min(segment_lengths)
            
            if min_seg_length < min_spacing * 0.999:
                max_segments = int(np.floor(total_length / min_spacing)) - 1
                if max_segments < 1:
                    max_segments = 1
                actual_spacing = total_length / max_segments
                distances = np.linspace(0, total_length, max_segments + 1, endpoint=True)
                new_coords = np.array([line.interpolate(d).coords[0] for d in distances])
                if not np.allclose(new_coords[0], new_coords[-1], rtol=1e-10):
                    new_coords[-1] = new_coords[0]
        
        return new_coords
    
    exterior_coords = np.asarray(polygon.exterior.coords)
    exterior_resampled = resample_ring_optimal(exterior_coords, spacing)
    
    if len(exterior_resampled) < 4:
        raise ValueError("Exterior ring too short after resampling")
    
    interiors_resampled = []
    for interior in polygon.interiors:
        ring_coords = np.asarray(interior.coords)
        ring_resampled = resample_ring_optimal(ring_coords, spacing)
        
        if len(ring_resampled) >= 4:
            interiors_resampled.append(ring_resampled)
    
    poly_new = Polygon(exterior_resampled, interiors_resampled)
    
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
