import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


def getiso(xpos, ypos, zdat, ilev, filt=0.0):
    """
    getiso extract an iso-contour from a structured 2D dataset.

    Parameters
    ----------
    xpos, ypos : ndarray (N, M)
        Grid coordinates (must be same shape as zdat).
    zdat : ndarray (N, M)
        Scalar field values.
    ilev : float
        Isocontour level.
    filt : float, optional
        Minimum length scale filter (default 0).

    Returns
    -------
    node : ndarray (K, 2)
        Coordinates of contour vertices.
    edge : ndarray (E, 2)
        PSLG edges between contour vertices.

    References
    ----------
    Translation of the MESH2D function `getiso`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ----------------------------------- compute the isocontour
    fig, ax = plt.subplots()
    cs = ax.contour(xpos, ypos, zdat, levels=[ilev])

    try:
        # ------------------------------------ Matplotlib <3.9
        collections = cs.collections
    except AttributeError:
        # ------------------------------------ Matplotlib >=3.9
        class DummyCollection:
            def __init__(self, segs):
                self.paths = [type("PathLike", (), {"vertices": seg})() for seg in segs]

            def get_paths(self):
                return self.paths

        collections = [DummyCollection(segs) for segs in cs.allsegs if segs]

    plt.close(fig)

    node = []
    edge = []
    # ------------------------------------ "walk" contour segment
    for collection in collections:
        for path in collection.get_paths():
            ppts = path.vertices
            numc = ppts.shape[0]

            pmin = ppts.min(axis=0)
            pmax = ppts.max(axis=0)
            pdel = pmax - pmin

            if np.min(pdel) >= filt:
                if np.allclose(ppts[0], ppts[-1]):
                    # -------------------------------- closed - back to start
                    enew = np.vstack(
                        [
                            np.column_stack(
                                [np.arange(0, numc - 1), np.arange(1, numc)]
                            ),
                            [numc - 1, 0],
                        ]
                    )
                else:
                    # -------------------------------- open - dangling endpts
                    enew = np.column_stack([np.arange(0, numc - 1), np.arange(1, numc)])

                offset = len(node)
                enew = enew + offset

                node.extend(ppts.tolist())
                edge.extend(enew.tolist())

    node = np.array(node)
    edge = np.array(edge, dtype=int)

    return node, edge


def getiso_polygone(x, y, z, zmax=None) -> Polygon:
    """
    Extract a MultiPolygon from a 2D scalar field by thresholding (similar to getiso logic).

    Parameters
    ----------
    x, y : ndarray
        1D or 2D arrays defining the grid coordinates (must match z).
    z : ndarray
        2D scalar field.
    zmax : float, optional
        Threshold value. Polygons will enclose regions where z <= zmax.
        If None, the 0-level is used.

    Returns
    -------
    multipolygon : shapely.geometry.MultiPolygon
        Extracted polygons with holes (if any).
    """

    # -----------------------ensure arrays are 2D and consistent
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y

    if X.shape != Y.shape or X.shape != z.shape:
        raise ValueError("Inconsistent array shapes between x, y, z.")

    # -----------------------select contour level
    new_mask = np.full(z.shape, 1)
    if zmax is not None:
        new_mask[z > zmax] = -1

    # -----------------------compute contour lines
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, new_mask, levels=[0, 1])

    try:
        # ------------------------------------ Matplotlib <3.9
        collections = cs.collections
    except AttributeError:
        # ------------------------------------ Matplotlib >=3.9
        class DummyCollection:
            def __init__(self, segs):
                self.paths = [type("PathLike", (), {"vertices": seg})() for seg in segs]

            def get_paths(self):
                return self.paths

        collections = [DummyCollection(segs) for segs in cs.allsegs if segs]

    plt.close(fig)

    polygons = []

    # -----------------------extract closed paths as polygons
    for collection in collections:
        for path in collection.get_paths():
            pts = path.vertices
            if pts.shape[0] < 4:
                continue
            if np.allclose(pts[0], pts[-1]):
                poly = Polygon(pts)
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    polygons.append(poly)

    # -----------------------no valid polygons
    if not polygons:
        return None

    # -----------------------sort polygons by area (largest first)
    polygons.sort(key=lambda p: p.area, reverse=True)

    # -----------------------build hierarchy (holes)
    final_polys = []
    while polygons:
        outer = polygons.pop(0)
        holes = [p.exterior.coords for p in polygons if p.within(outer)]
        polygons = [p for p in polygons if not p.within(outer)]
        final_polys.append(Polygon(outer.exterior.coords, holes))

    largest_poly = max(final_polys, key=lambda p: p.area)

    return largest_poly
