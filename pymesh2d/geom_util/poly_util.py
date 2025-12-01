import numpy as np
from shapely.geometry import Polygon, LineString


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
