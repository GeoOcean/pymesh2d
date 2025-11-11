import numpy as np

from .linenear import linenear
from .maketree import maketree
from .maprect import maprect
from .queryset import queryset


def lineline(pa, pb, pc, pd, tree=None, options=None):
    """
    Compute intersections between line segments in d-dimensional space.

    This function identifies intersections between two sets of line segments
    in arbitrary dimensions. Each line set is defined by pairs of endpoints
    `[PA, PB]` and `[PC, PD]`, where all coordinate arrays have the same
    number of dimensions.

    For each query line in `[PC, PD]`, the function returns the subset of
    lines from `[PA, PB]` that intersect it. This is useful for detecting
    geometric intersections or enforcing topological constraints in meshing
    and geometry-processing applications.

    Parameters
    ----------
    PA : ndarray of shape (NL, ND)
        Coordinates of the first endpoints of the target line segments.
    PB : ndarray of shape (NL, ND)
        Coordinates of the second endpoints of the target line segments.
    PC : ndarray of shape (NQ, ND)
        Coordinates of the first endpoints of the query line segments.
    PD : ndarray of shape (NQ, ND)
        Coordinates of the second endpoints of the query line segments.
    TR : object, optional
        Precomputed AABB tree structure used to accelerate the query.
        If provided, it will be reused for efficiency, assuming `[PA, PB]`
        has not changed.
    OP : dict, optional
        Additional parameters controlling the creation or behavior of the
        underlying AABB tree.

    Returns
    -------
    LP : ndarray of shape (NQ, 2)
        Index pairs defining, for each query line, the start and end
        positions in `LI` listing the intersecting line segments.
        Lines with no intersections have `LP[i, 0] == 0`.
    LI : ndarray of shape (K,)
        Flattened list of line indices intersecting the query lines.
        For query `i`, intersecting lines are `LI[LP[i,0]:LP[i,1]]`.
    TR : object, optional
        AABB tree structure used internally to compute the query.
        It can be reused in subsequent calls for improved performance.

    Notes
    -----
    If the same line collection `[PA, PB]` is queried multiple times,
    reusing the returned `TR` structure can significantly improve
    performance. However, the geometry of the lines must remain unchanged
    between queries.

    References
    ----------
    Translation of the MESH2D function `LINELINE`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    Additional reference:
    D. Engwirda, *"Locally-optimal Delaunay-refinement & optimisation-based mesh generation"*,
    Ph.D. Thesis, School of Mathematics and Statistics, University of Sydney, 2014.
    http://hdl.handle.net/2123/13148
    """

    # --------------------------------------------- basic checks
    if not all(isinstance(x, np.ndarray) for x in [pa, pb, pc, pd]):
        raise TypeError("pa, pb, pc, and pd must all be numpy arrays.")

    if any(x.ndim != 2 for x in [pa, pb, pc, pd]):
        raise ValueError("All inputs must be 2D arrays.")

    if pa.shape[1] != pb.shape[1] or pc.shape[1] != pd.shape[1]:
        raise ValueError("Line endpoints must have consistent dimensions.")

    if pa.shape[1] < 2:
        raise ValueError("Dimension must be >= 2.")

    nd = pa.shape[1]
    nl = pa.shape[0]
    ml = pc.shape[0]

    if nl == 0 or ml == 0:
        return np.zeros((0, 2), dtype=int), np.array([], dtype=int), tree

    # ------------------------------ compute aabb-tree for d-line
    if tree is None:
        ab = np.zeros((nl, nd * 2))
        for ax in range(nd):  #  compute aabb's
            ab[:, ax] = np.minimum(pa[:, ax], pb[:, ax])
            ab[:, nd + ax] = np.maximum(pa[:, ax], pb[:, ax])
        tree = maketree(ab, options)

    # ------------------------------ compute tree-to-vert mapping
    ab = np.zeros((ml, nd * 2))
    for ax in range(nd):  #  compute aabb's
        ab[:, ax] = np.minimum(pc[:, ax], pd[:, ax])
        ab[:, nd + ax] = np.maximum(pc[:, ax], pd[:, ax])
    tm, _ = maprect(tree, ab)

    # ----------------------------- compute line-to-line queries
    li, ip, lj = queryset(tree, tm, linekernel, pc, pd, pa, pb)

    # ----------------------------- re-index onto full obj. list
    lp = np.zeros((ml, 2), dtype=int)
    lp[:, 1] = -1

    if li.size == 0:
        return lp, lj, tree

    lp[li, :] = ip

    return lp, lj, tree


def linekernel(l1, l2, pa, pb, pc, pd):
    """
    LINEKERNEL - d-dimensional line//line intersection kernel routine.

    Parameters
    ----------
    l1 : np.ndarray
        Indices of lines from the first set.
    l2 : np.ndarray
        Indices of lines from the second set.
    pa, pb : np.ndarray
        (n_lines_A, n_dim) arrays defining the endpoints of the first set of lines.
    pc, pd : np.ndarray
        (n_lines_B, n_dim) arrays defining the endpoints of the second set of lines

    Returns
    -------
    i1 : np.ndarray
        Indices of intersecting lines from the first set.
    i2 : np.ndarray
        Indices of intersecting lines from the second set.

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Translation of the MESH2D function `LINEKERNEL`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    m1 = len(l1)
    m2 = len(l2)

    # -------------------------- push line/vert onto n*m tile
    l1 = np.repeat(l1, m2)
    l2 = np.tile(l2, m1)
    # ------------------------- compute O(n*m) intersections
    ok, tp, tq = linenear(pa[l1, :], pb[l1, :], pc[l2, :], pd[l2, :])

    rt = 1.0 + np.finfo(float).eps
    mask = (np.abs(tp) <= rt) & (np.abs(tq) <= rt)
    mask &= ok

    i1 = l1[mask]
    i2 = l2[mask]

    return i1, i2
