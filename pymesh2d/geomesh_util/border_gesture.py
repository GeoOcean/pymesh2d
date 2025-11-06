import numpy as np
import rasterio
import pyproj
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from shapely import ops
from matplotlib.path import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple

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

def get_multipolygon_from_raster(raster_path: str, zmax: Optional[float] = None, band: int = 1) -> Tuple[MultiPolygon, rasterio.crs.CRS]:
    """Extrait un MultiPolygon depuis un raster GeoTIFF par seuillage."""
    with rasterio.open(raster_path) as src:
        z = src.read(band, masked=True)
        bounds = src.bounds
        x = np.linspace(bounds.left, bounds.right, src.width)
        y = np.linspace(bounds.top, bounds.bottom, src.height)
        crs = src.crs

    new_mask = np.full(z.shape, 1)
    if zmax is not None:
        new_mask[z > zmax] = -1

    fig, ax = plt.subplots()
    ax.contourf(x, y, new_mask, levels=[0, 1])
    rings = [
        LinearRing(ring)
        for coll in ax.collections
        for path in coll.get_paths()
        for ring in path.to_polygons(closed_only=True)
        if ring.shape[0] > 3
    ]
    plt.close(fig)  # évite la fuite mémoire matplotlib

    if not rings:
        return MultiPolygon([]), crs

    polys = []
    while rings:
        idx = np.argmax([Polygon(r).area for r in rings])
        outer = rings.pop(idx)
        path = Path(np.asarray(outer.coords), closed=True)
        inners = [rings.pop(i) for i in reversed(range(len(rings))) if path.contains_point(np.asarray(rings[i].coords)[0])]
        polys.append(Polygon(outer, inners))

    return MultiPolygon(polys), crs

def get_utm_crs_from_crs(crs):
    """Retourne un CRS UTM adapté au CRS géographique donné."""
    if crs.is_projected:
        return crs
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = transformer.transform(0, 0)
    utm_zone = int((lon0 + 180) / 6) + 1
    epsg_code = 326 if lat0 >= 0 else 327
    return pyproj.CRS.from_epsg(epsg_code * 100 + utm_zone)


def reproject_geometry(geom, crs_from, crs_to):
    """Reprojette une géométrie shapely d’un CRS à un autre."""
    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True).transform
    return ops.transform(transformer, geom)

def resample_line_by_distance(x, y, spacing: float):
    coords = np.column_stack((x, y))
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    dist = np.insert(np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))), 0, 0)
    if dist[-1] == 0:
        return coords[:, 0], coords[:, 1]
    new_d = np.linspace(0, dist[-1], max(4, int(dist[-1] / spacing)))
    return np.interp(new_d, dist, coords[:, 0]), np.interp(new_d, dist, coords[:, 1])


def resample_polygon_by_distance(poly: Polygon, spacing: float) -> Optional[Polygon]:
    x, y = poly.exterior.xy
    x_new, y_new = resample_line_by_distance(x, y, spacing)
    if len(x_new) < 4:
        return None
    outer = LinearRing(np.column_stack([x_new, y_new]))
    inners = []
    for interior in poly.interiors:
        xi, yi = interior.xy
        xi_new, yi_new = resample_line_by_distance(xi, yi, spacing)
        if len(xi_new) >= 4:
            inners.append(LinearRing(np.column_stack([xi_new, yi_new])))
    return Polygon(outer, inners)


def resample_multipolygon_by_distance(mpoly, spacing: float) -> MultiPolygon:
    if isinstance(mpoly, Polygon):
        mpoly = MultiPolygon([mpoly])
    resampled = [p for poly in mpoly.geoms if (p := resample_polygon_by_distance(poly, spacing)) is not None]
    return MultiPolygon(resampled).buffer(0) if resampled else MultiPolygon([])

def clean_multipolygon(mpoly: MultiPolygon, min_perimeter: float = 100.0, min_vertices: int = 8, min_area: float = 5000.0):
    """Supprime les petits polygones et trous non pertinents (y compris les îles)."""
    def valid(poly):
        if (len(poly.exterior.coords) < min_vertices or
            poly.exterior.length < min_perimeter or
            poly.area < min_area):
            return None
        inners = [
            LinearRing(i.coords)
            for i in poly.interiors
            if len(i.coords) >= min_vertices
            and LinearRing(i.coords).length >= min_perimeter
        ]
        return Polygon(poly.exterior, inners)

    if isinstance(mpoly, Polygon):
        p = valid(mpoly)
        return MultiPolygon([p]) if p else MultiPolygon([])

    return MultiPolygon([p for poly in mpoly.geoms if (p := valid(poly))])


def process_raster_contours(raster_path=None, geom=None, crs=None, zmax=0.0, spacing=100.0):
    """Pipeline complet raster ou géométrie directe."""
    if geom is None and raster_path is None:
        raise ValueError("Fournir `raster_path` ou `geom`.")
    if raster_path:
        mpoly, crs = get_multipolygon_from_raster(raster_path, zmax=zmax)
    else:
        mpoly = geom
    utm_crs = get_utm_crs_from_crs(crs)
    mpoly_utm = reproject_geometry(mpoly, crs, utm_crs)
    mpoly_resampled_utm = resample_multipolygon_by_distance(mpoly_utm, spacing)
    mpoly_resampled_utm = clean_multipolygon(mpoly_resampled_utm, min_perimeter=spacing)
    mpoly_resampled = reproject_geometry(mpoly_resampled_utm, utm_crs, crs)
    return mpoly_resampled, crs

def extract_nodes_edges_indexed(geom):
    """Extrait les nœuds et arêtes (indices 0-based) d’un Polygon/MultiPolygon, sans fermer les anneaux."""
    import numpy as np
    from shapely.geometry import Polygon, MultiPolygon

    coords_list = []

    def process_polygon(poly):
        # extérieur
        x, y = poly.exterior.xy
        coords = np.column_stack((x, y))
        coords_list.append(coords)

        # intérieurs (trous)
        for interior in poly.interiors:
            xi, yi = interior.xy
            icoords = np.column_stack((xi, yi))
            coords_list.append(icoords)

    if isinstance(geom, Polygon):
        process_polygon(geom)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            process_polygon(poly)
    else:
        raise TypeError("geom doit être Polygon ou MultiPolygon")

    # --- concaténer toutes les coordonnées
    all_coords = np.vstack(coords_list)

    # --- nœuds uniques
    nodes, inv = np.unique(all_coords, axis=0, return_inverse=True)

    # --- construire les arêtes (indices 0-based, non fermées)
    edges = []
    offset = 0
    for coords in coords_list:
        n = len(coords)
        idx_seq = inv[offset:offset + n]
        # ne ferme pas le polygone
        ring_edges = np.column_stack([idx_seq[:-1], idx_seq[1:]])
        edges.append(ring_edges)
        offset += n

    edges = np.vstack(edges)

    return nodes, edges

def reproject_vertices(vert, crs_from, crs_to):
    """
    Reproject an array of vertex coordinates from one CRS to another.

    Parameters
    ----------
    vert : (N, 2) array
        Vertex coordinates [[x, y], ...].
    crs_from : pyproj.CRS or str
        Source coordinate reference system (e.g. 'EPSG:32630').
    crs_to : pyproj.CRS or str
        Target coordinate reference system (e.g. 'EPSG:4326').

    Returns
    -------
    vert_out : (N, 2) array
        Reprojected coordinates [[x, y], ...].
    """
    if not isinstance(vert, np.ndarray):
        vert = np.asarray(vert)

    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    x, y = transformer.transform(vert[:, 0], vert[:, 1])
    return np.column_stack((x, y))