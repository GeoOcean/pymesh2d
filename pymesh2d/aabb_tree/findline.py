import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findline(pa, pb, pp, tree=None, options=None):
    """
    Perform "point-on-line" spatial queries in d-dimensional space.

    This function identifies which d-dimensional line segments intersect
    a given set of query points. Lines are specified as pairs of endpoints
    `[PA, PB]`, where both are arrays of coordinates defining each segment.

    For each query point in `PI`, the routine returns the subset of lines
    that intersect or contain the point. This is particularly useful in
    mesh-generation or geometry-processing tasks to locate nearby or
    enclosing edges.

    Parameters
    ----------
    PA : ndarray of shape (NL, ND)
        Coordinates of the first endpoints of the line segments.
    PB : ndarray of shape (NL, ND)
        Coordinates of the second endpoints of the line segments.
    PI : ndarray of shape (NP, ND)
        Query points for which intersecting line segments are sought.
    TR : object, optional
        Precomputed AABB tree structure used to accelerate the search.
        If provided, it will be reused for efficiency, assuming that
        the set of line segments has not changed.
    OP : dict, optional
        Additional parameters controlling the creation or behavior of the
        underlying AABB tree.

    Returns
    -------
    LP : ndarray of shape (NP, 2)
        Index pairs defining, for each query point, the start and end
        positions in `LI` listing the intersecting line segments.
        Points with no intersection have `LP[i, 0] == 0`.
    LI : ndarray of shape (K,)
        Flattened list of line indices that intersect the query points.
        For query `i`, intersecting lines are `LI[LP[i,0]:LP[i,1]]`.
    TR : object, optional
        AABB tree structure used internally for the spatial queries.
        It can be reused in subsequent calls for improved performance.

    Notes
    -----
    If the same line collection `[PA, PB]` is queried multiple times,
    reusing the returned `TR` structure can significantly reduce computation
    time. The line endpoints **must remain unchanged** between queries.

    References
    ----------
    Translation of the MESH2D function `FINDLINE`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    Additional reference:
    D. Engwirda, *"Locally-optimal Delaunay-refinement & optimisation-based mesh generation"*,
    Ph.D. Thesis, School of Mathematics and Statistics, University of Sydney, 2014.
    http://hdl.handle.net/2123/13148
    """

    # --------------------- Basic checks
    if not (
        isinstance(pa, np.ndarray)
        and isinstance(pb, np.ndarray)
        and isinstance(pp, np.ndarray)
    ):
        raise TypeError("All inputs (pa, pb, pp) must be numpy arrays")

    if pa.ndim != 2 or pb.ndim != 2 or pp.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")

    if (
        pa.shape[0] != pb.shape[0]
        or pa.shape[1] != pb.shape[1]
        or pp.shape[1] != pa.shape[1]
    ):
        raise ValueError("Inconsistent array dimensions between pa, pb, and pp")

    # --------------------- Quick return for empty inputs
    if pa.size == 0 or pb.size == 0:
        return np.zeros((0, 2), dtype=int), np.array([], dtype=int), tree

    # --------------------- compute aabb-tree for d-line
    if tree is None:
        n_dim = pp.shape[1]
        n_lines = pa.shape[0]
        ab = np.zeros((n_lines, n_dim * 2))

        # compute aabb-tree
        for ax in range(n_dim):
            ab[:, ax] = np.minimum(pa[:, ax], pb[:, ax])
            ab[:, n_dim + ax] = np.maximum(pa[:, ax], pb[:, ax])

        tree = maketree(ab, options)

    # ---------------------  compute tree-to-vert mapping
    tm, _ = mapvert(tree, pp)

    # --------------------- compute intersection rel-tol
    p0 = np.min(np.vstack([pa, pb]), axis=0)
    p1 = np.max(np.vstack([pa, pb]), axis=0)
    zt = np.max(p1 - p0) * np.finfo(float).eps ** 0.8

    # --------------------- compute vert-to-line queries
    li, ip, lj = queryset(tree, tm, linekernel, pp, pa, pb, zt)
    # --------------------- re-index onto full obj. list
    lp = np.zeros((pp.shape[0], 2), dtype=int)
    lp[:, 1] = -1  # default for no intersection

    if li.size == 0:
        return lp, lj, tree

    lp[li, :] = ip

    return lp, lj, tree


def linekernel(pk, lk, pp, pa, pb, zt):
    """
    LINEKERNEL - d-dimensional point/line intersection kernel.
    Parameters
    ----------
    pk : (M,) array
        Indices of query points.
    lk : (N,) array
        Indices of lines.
    pp : (P,ND) array
        Query points.
    pa : (L,ND) array
        Line start points.
    pb : (L,ND) array
        Line end points.
    zt : float
        Relative tolerance for intersection.
    Returns
    -------
    ip : array
        Query point indices where intersection occurs.
    il : array
        Line indices intersecting.
    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Translation of the MESH2D function `LINEKERNEL`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    mp = len(pk)
    ml = len(lk)

    # -------------------------- push line/vert onto n*m tile
    pk = np.tile(pk, ml)
    lk = np.repeat(lk, mp)

    # --------------------------  compute O(n*m) intersections
    mm = 0.5 * (pa[lk, :] + pb[lk, :])
    DD = 0.5 * (pb[lk, :] - pa[lk, :])

    mpv = mm - pp[pk, :]

    tt = -np.sum(mpv * DD, axis=1) / np.sum(DD * DD, axis=1)
    tt = np.clip(tt, -1.0, 1.0)

    _n_dim = pp.shape[1]
    qq = mm + (tt[:, None] * DD)

    on = np.sum((pp[pk, :] - qq) ** 2, axis=1) <= zt**2

    ip = pk[on]
    il = lk[on]

    return ip, il
