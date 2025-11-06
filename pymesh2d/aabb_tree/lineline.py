import numpy as np

from .maketree import maketree
from .maprect import maprect
from .queryset import queryset


def lineline(pa, pb, pc, pd, tree=None, options=None):
    """
    LINELINE - intersection between line segments in d-dimensional space.

    Parameters
    ----------
    pa, pb : np.ndarray
        (n_lines_A, n_dim) arrays defining the endpoints of the first set of lines.
    pc, pd : np.ndarray
        (n_lines_B, n_dim) arrays defining the endpoints of the second set of lines.
    tree : dict, optional
        Precomputed AABB tree from `maketree`.
    options : dict, optional
        Options for tree construction.

    Returns
    -------
    lp : np.ndarray
        (n_lines_B, 2) start/end indices of intersecting lines from the first set.
    lj : np.ndarray
        Indices of intersecting lines from the first set.
    tree : dict
        AABB tree used for the spatial query.

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

    # ---------------- Basic checks ----------------
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

    # ---------------- Build AABB tree if needed ----------------
    if tree is None:
        ab = np.zeros((nl, nd * 2))
        for ax in range(nd):
            ab[:, ax] = np.minimum(pa[:, ax], pb[:, ax])
            ab[:, nd + ax] = np.maximum(pa[:, ax], pb[:, ax])
        tree = maketree(ab, options)

    # ---------------- Compute AABB for query lines ----------------
    ab = np.zeros((ml, nd * 2))
    for ax in range(nd):
        ab[:, ax] = np.minimum(pc[:, ax], pd[:, ax])
        ab[:, nd + ax] = np.maximum(pc[:, ax], pd[:, ax])
    tm = maprect(tree, ab)

    # ---------------- Perform queries ----------------
    li, ip, lj = queryset(tree, tm, linekernel, pc, pd, pa, pb)

    # ---------------- Re-index results ----------------
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
    Darren Engwirda, Locally-optimal Delaunay-refinement
    and optimisation-based mesh generation, Ph.D. Thesis,
    School of Mathematics and Statistics, The University
    of Sydney, September 2014.
    """

    m1 = len(l1)
    m2 = len(l2)

    # Expand combinations of all (l1, l2)
    l1 = np.repeat(l1, m2)
    l2 = np.tile(l2, m1)

    ok, tp, tq = linenear(pa[l1, :], pb[l1, :], pc[l2, :], pd[l2, :])

    rt = 1.0 + np.finfo(float).eps
    mask = (np.abs(tp) <= rt) & (np.abs(tq) <= rt)
    mask &= ok

    i1 = l1[mask]
    i2 = l2[mask]

    return i1, i2


def linenear(pa, pb, pc, pd, tol=1e-12):
    """
    LINENEAR - Compute parametric intersections between 2D/3D line segments.

    Parameters
    ----------
    pa, pb : np.ndarray
        (N, D) arrays of endpoints of the first set of segments.
    pc, pd : np.ndarray
        (N, D) arrays of endpoints of the second set of segments.
    tol : float
        Numerical tolerance.

    Returns
    -------
    ok : np.ndarray (bool)
        True where lines are not parallel.
    ta, tb : np.ndarray
        Parametric coordinates along [PA,PB] and [PC,PD] respectively.

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

    dA = pb - pa
    dB = pd - pc
    dP = pc - pa

    if pa.shape[1] == 2:
        # 2D case
        denom = dA[:, 0] * dB[:, 1] - dA[:, 1] * dB[:, 0]
        ok = np.abs(denom) > tol

        ta = np.where(ok, (dP[:, 0] * dB[:, 1] - dP[:, 1] * dB[:, 0]) / denom, np.nan)
        tb = np.where(ok, (dP[:, 0] * dA[:, 1] - dP[:, 1] * dA[:, 0]) / denom, np.nan)

    elif pa.shape[1] == 3:
        # 3D case — use vector projection method
        cross = np.cross(dA, dB)
        denom = np.linalg.norm(cross, axis=1) ** 2
        ok = denom > tol

        ta = np.empty_like(denom)
        tb = np.empty_like(denom)
        ta.fill(np.nan)
        tb.fill(np.nan)

        mask = ok
        if np.any(mask):
            dP_mask = dP[mask]
            dA_mask = dA[mask]
            dB_mask = dB[mask]
            crossP = np.cross(dP_mask, dB_mask)
            crossQ = np.cross(dP_mask, dA_mask)
            denom_mask = denom[mask][:, None]
            ta[mask] = np.sum(crossP * cross[mask], axis=1) / denom[mask]
            tb[mask] = np.sum(crossQ * cross[mask], axis=1) / denom[mask]
    else:
        raise ValueError("Only 2D or 3D line intersection is supported.")

    return ok, ta, tb
