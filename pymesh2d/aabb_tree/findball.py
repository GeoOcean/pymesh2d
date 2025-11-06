import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findball(bb, pp, tr=None, op=None):
    """
    FINDBALL: spatial queries for collections of d-balls.

    Parameters
    ----------
    bb : (B, ND+1) array
        Ball definitions: first ND columns are centers, last column is squared radii.
    pp : (P, ND) array
        Query points.
    tr : dict, optional
        Precomputed AABB tree (see maketree).
    op : dict, optional
        Options for tree creation.

    Returns
    -------
    bp : (P,2) array
        For each query point, indices into bj such that
        balls for query i are bj[bp[i,0]:bp[i,1]].
    bj : (M,) array
        Flattened list of intersecting ball indices.
    tr : dict
        AABB tree (for reuse in subsequent calls).

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

    bp, bj = np.array([]), np.array([])

    # ------------------------------ basic checks
    if bb is None or pp is None:
        raise ValueError("findball:incorrectNumInputs (need at least bb, pp)")

    bb = np.asarray(bb, dtype=float)
    pp = np.asarray(pp, dtype=float)

    if bb.ndim != 2 or bb.shape[1] < 3:
        raise ValueError("findball:incorrectDimensions (bb must be (B,ND+1))")
    if pp.ndim != 2 or bb.shape[1] != pp.shape[1] + 1:
        raise ValueError("findball:incorrectDimensions (pp must be (P,ND))")

    if tr is not None and not isinstance(tr, dict):
        raise TypeError("findball:incorrectInputClass (tr must be struct/dict)")
    if op is not None and not isinstance(op, dict):
        raise TypeError("findball:incorrectInputClass (op must be struct/dict)")

    if bb.size == 0:
        return bp, bj, tr

    # ------------------------------ build tree if not given
    if tr is None:
        nd = pp.shape[1]
        rs = np.sqrt(bb[:, nd])[:, None]  # radii
        rs = np.tile(rs, (1, nd))
        ab = np.hstack([bb[:, :nd] - rs, bb[:, :nd] + rs])  # aabb
        tr = maketree(ab, op)

    # ------------------------------ map query vertices
    tm, _ = mapvert(tr, pp)

    # ------------------------------ run query
    bi, ip, bj = queryset(tr, tm, ballkern, pp, bb)

    # ------------------------------ reindex onto full list
    bp = np.zeros((pp.shape[0], 2), dtype=int)
    bp[:, 1] = -1
    if bi.size > 0:
        bp[bi, :] = ip

    return bp, bj, tr


def ballkern(pk, bk, pp, bb):
    """
    BALLKERN: d-dimensional ball-vertex intersection kernel.

    Parameters
    ----------
    pk : (M,) array
        Indices of query points.
    bk : (N,) array
        Indices of balls.
    pp : (P,ND) array
        Query points.
    bb : (B,ND+1) array
        Ball centers + squared radii.

    Returns
    -------
    ip : array
        Query point indices where intersection occurs.
    ib : array
        Ball indices intersecting.

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
    mb = len(bk)
    nd = pp.shape[1]

    # Tile points and balls
    bk_tiled = np.tile(bk, mp)
    pk_tiled = np.repeat(pk, mb)

    # Compute squared distances
    diff = pp[pk_tiled, :] - bb[bk_tiled, :nd]
    dd = np.sum(diff**2, axis=1)

    # Check intersection
    inside = dd <= bb[bk_tiled, nd]

    ip = pk_tiled[inside]
    ib = bk_tiled[inside]

    return ip, ib
