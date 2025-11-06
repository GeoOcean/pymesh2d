import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findline(pa, pb, pp, tree=None, options=None):
    """
    FINDLINE - 'point-on-line' queries in d-dimensional space.

    Parameters
    ----------
    pa : np.ndarray
        (n_lines, n_dim) array of line start points.
    pb : np.ndarray
        (n_lines, n_dim) array of line end points.
    pp : np.ndarray
        (n_points, n_dim) array of query points.
    tree : dict, optional
        Precomputed AABB tree from `maketree`.
    options : dict, optional
        Tree construction options (passed to maketree).

    Returns
    -------
    lp : np.ndarray
        (n_points, 2) array of line index ranges per point (0-based).
    lj : np.ndarray
        List of intersecting line indices.
    tree : dict
        The AABB tree used in the computation.
    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Darren Engwirda, Locally-optimal Delaunay-refinement
    and optimisation-based mesh generation, Ph.D. Thesis,
    School of Mathematics and Statistics, The University
    of Sydney, September 2014.
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

    # --------------------- Build AABB tree if needed
    if tree is None:
        n_dim = pp.shape[1]
        n_lines = pa.shape[0]
        ab = np.zeros((n_lines, n_dim * 2))

        # Compute min/max bounds for each dimension
        for ax in range(n_dim):
            ab[:, ax] = np.minimum(pa[:, ax], pb[:, ax])
            ab[:, n_dim + ax] = np.maximum(pa[:, ax], pb[:, ax])

        tree = maketree(ab, options)

    # --------------------- Map tree to query vertices
    tm = mapvert(tree, pp)

    # --------------------- Compute relative tolerance
    p0 = np.min(np.vstack([pa, pb]), axis=0)
    p1 = np.max(np.vstack([pa, pb]), axis=0)
    zt = np.max(p1 - p0) * np.finfo(float).eps ** 0.8

    # --------------------- Query lines per vertex
    li, ip, lj = queryset(tree, tm, linekernel, pp, pa, pb, zt)

    # --------------------- Build range array (0-based)
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
    Darren Engwirda, Locally-optimal Delaunay-refinement
    and optimisation-based mesh generation, Ph.D. Thesis,
    School of Mathematics and Statistics, The University
    of Sydney, September 2014.
    """

    mp = len(pk)
    ml = len(lk)

    # Tile points and lines into n*m combinations
    pk = np.tile(pk, ml)
    lk = np.repeat(lk, mp)

    # Midpoints and half-length vectors
    mm = 0.5 * (pa[lk, :] + pb[lk, :])
    DD = 0.5 * (pb[lk, :] - pa[lk, :])

    mpv = mm - pp[pk, :]

    tt = -np.sum(mpv * DD, axis=1) / np.sum(DD * DD, axis=1)
    tt = np.clip(tt, -1.0, 1.0)

    n_dim = pp.shape[1]
    qq = mm + (tt[:, None] * DD)

    on = np.sum((pp[pk, :] - qq) ** 2, axis=1) <= zt**2

    ip = pk[on]
    il = lk[on]

    return ip, il
