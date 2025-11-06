import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findtria(pp, tt, pj, tree=None, options=None):
    """
    FINDTRIA - spatial queries for collections of d-simplexes.

    Parameters
    ----------
    pp : np.ndarray
        (n_points, n_dim) coordinates of vertices.
    tt : np.ndarray
        (n_simplexes, n_vertices_per_simplex) connectivity (0-based indices).
    pj : np.ndarray
        (n_queries, n_dim) coordinates of query points.
    tree : dict, optional
        Precomputed AABB tree from `maketree`.
    options : dict, optional
        Parameters to control AABB tree creation.

    Returns
    -------
    tp : np.ndarray
        (n_queries, 2) start/end indices per query point.
    tj : np.ndarray
        Array of simplex indices intersecting each point.
    tree : dict
        AABB tree used for the query.

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

    # Basic checks
    if not (
        isinstance(pp, np.ndarray)
        and isinstance(tt, np.ndarray)
        and isinstance(pj, np.ndarray)
    ):
        raise TypeError("pp, tt, and pj must be numpy arrays.")

    if pj.size == 0:
        return np.zeros((0, 2), dtype=int), np.array([], dtype=int), tree

    if pp.ndim != 2 or tt.ndim != 2 or pj.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")

    if pp.shape[1] < 2 or pp.shape[1] > tt.shape[1]:
        raise ValueError("Incorrect input dimensions.")

    if tt.shape[1] < 3:
        raise ValueError("Triangles must have at least 3 vertices.")

    if pj.shape[1] != pp.shape[1]:
        raise ValueError("pj and pp must have the same dimensionality.")

    # Build bounding boxes if tree is not given
    if tree is None:
        bi = pp[tt[:, 0], :].copy()
        bj = pp[tt[:, 0], :].copy()
        for ii in range(1, tt.shape[1]):
            bi = np.minimum(bi, pp[tt[:, ii], :])
            bj = np.maximum(bj, pp[tt[:, ii], :])
        bb = np.hstack([bi, bj])
        tree = maketree(bb, options)

    # Map query vertices to tree nodes
    tm, _ = mapvert(tree, pj)

    # Compute tolerance
    x0 = np.min(pp, axis=0)
    x1 = np.max(pp, axis=0)
    rt = np.prod(x1 - x0) * np.finfo(float).eps ** 0.8

    # Perform spatial queries
    ti, ip, tj = queryset(tree, tm, triakern, pj, pp, tt, rt)

    # Build range index array
    tp = np.zeros((pj.shape[0], 2), dtype=int)
    tp[:, 1] = -1
    if ti.size == 0:
        return tp, tj, tree
    tp[ti, :] = ip

    return tp, tj, tree


def triakern(pk, tk, pi, pp, tt, rt):
    """
    TRIAKERN - Compute point/simplex intersections within a tile.

    Parameters
    ----------
    pk : (M,) array
        Indices of query points.
    tk : (N,) array
        Indices of simplexes.
    pi : (P,ND) array
        Query points.
    pp : (V,ND) array
        Vertex coordinates.
    tt : (T,nv) array
        Simplex connectivity (0-based indices).
    rt : float
        Relative tolerance for point-in-simplex tests.

    Returns
    -------
    ip : array
        Query point indices where intersection occurs.
    it : array
        Simplex indices where intersection occurs.

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
    mt = len(tk)

    pk = np.repeat(pk, mt)
    tk = np.tile(tk, mp)

    n_vertices = tt.shape[1]

    if n_vertices == 3:
        inside = intria2(pp, tt[tk, :], pi[pk, :], rt)
    elif n_vertices == 4:
        inside = intria3(pp, tt[tk, :], pi[pk, :], rt)
    else:
        ii, jj = intrian(pp, tt[tk, :], pi[pk, :])
        ip = pk[ii]
        it = tk[jj]
        return ip, it

    ip = pk[inside]
    it = tk[inside]
    return ip, it


def intria2(pp, tt, pi, rt):
    """
    INTRIA2 - Returns True for points enclosed by 2-simplexes (triangles).

    Parameters
    ----------
    pp : (V,2) array
        Vertex coordinates.
    tt : (T,3) array
        Triangle connectivity (0-based indices).
    pi : (P,2) array
        Query points.
    rt : float
        Relative tolerance for point-in-triangle tests.

    Returns
    -------
    inside : (T,) boolean array
        True for triangles containing the query point.

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
    t1, t2, t3 = tt[:, 0], tt[:, 1], tt[:, 2]
    vi = pp[t1, :] - pi
    vj = pp[t2, :] - pi
    vk = pp[t3, :] - pi

    aa = np.zeros((tt.shape[0], 3))
    aa[:, 0] = vi[:, 0] * vj[:, 1] - vj[:, 0] * vi[:, 1]
    aa[:, 1] = vj[:, 0] * vk[:, 1] - vk[:, 0] * vj[:, 1]
    aa[:, 2] = vk[:, 0] * vi[:, 1] - vi[:, 0] * vk[:, 1]

    rt2 = rt**2
    inside = (
        (aa[:, 0] * aa[:, 1] >= -rt2)
        & (aa[:, 1] * aa[:, 2] >= -rt2)
        & (aa[:, 2] * aa[:, 0] >= -rt2)
    )

    return inside


def intria3(pp, tt, pi, rt):
    """
    INTRIA3 - Returns True for points enclosed by 3-simplexes (tetrahedra).

    Parameters
    ----------
    pp : (V,3) array
        Vertex coordinates.
    tt : (T,4) array
        Tetrahedron connectivity (0-based indices).
    pi : (P,3) array
        Query points.
    rt : float
        Relative tolerance for point-in-tetrahedron tests.

    Returns
    -------
    inside : (T,) boolean array
        True for tetrahedra containing the query point.

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
    t1, t2, t3, t4 = tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]
    v1 = pi - pp[t1, :]
    v2 = pi - pp[t2, :]
    v3 = pi - pp[t3, :]
    v4 = pi - pp[t4, :]

    aa = np.zeros((tt.shape[0], 4))
    aa[:, 0] = (
        v1[:, 0] * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1])
        - v1[:, 1] * (v2[:, 0] * v3[:, 2] - v2[:, 2] * v3[:, 0])
        + v1[:, 2] * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])
    )
    aa[:, 1] = (
        v1[:, 0] * (v4[:, 1] * v2[:, 2] - v4[:, 2] * v2[:, 1])
        - v1[:, 1] * (v4[:, 0] * v2[:, 2] - v4[:, 2] * v2[:, 0])
        + v1[:, 2] * (v4[:, 0] * v2[:, 1] - v4[:, 1] * v2[:, 0])
    )
    aa[:, 2] = (
        v2[:, 0] * (v4[:, 1] * v3[:, 2] - v4[:, 2] * v3[:, 1])
        - v2[:, 1] * (v4[:, 0] * v3[:, 2] - v4[:, 2] * v3[:, 0])
        + v2[:, 2] * (v4[:, 0] * v3[:, 1] - v4[:, 1] * v3[:, 0])
    )
    aa[:, 3] = (
        v3[:, 0] * (v4[:, 1] * v1[:, 2] - v4[:, 2] * v1[:, 1])
        - v3[:, 1] * (v4[:, 0] * v1[:, 2] - v4[:, 2] * v1[:, 0])
        + v3[:, 2] * (v4[:, 0] * v1[:, 1] - v4[:, 1] * v1[:, 0])
    )

    rt2 = rt**2
    inside = (
        (aa[:, 0] * aa[:, 1] >= -rt2)
        & (aa[:, 0] * aa[:, 2] >= -rt2)
        & (aa[:, 0] * aa[:, 3] >= -rt2)
        & (aa[:, 1] * aa[:, 2] >= -rt2)
        & (aa[:, 1] * aa[:, 3] >= -rt2)
        & (aa[:, 2] * aa[:, 3] >= -rt2)
    )

    return inside


def intrian(pp, tt, pi):
    """
    INTRIAN - General n-simplex point-location using barycentric coordinates.

    Parameters
    ----------
    pp : (V,ND) array
        Vertex coordinates.
    tt : (T,nv) array
        Simplex connectivity (0-based indices).
    pi : (P,ND) array
        Query points.

    Returns
    -------
    ii : array
        Indices of query points inside simplexes.
    jj : array
        Indices of simplexes containing the query points.

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
    np_, pd = pi.shape
    nt, td = tt.shape

    mm = np.zeros((pd, pd, nt))
    for id_ in range(pd):
        for jd in range(pd):
            mm[id_, jd, :] = pp[tt[:, jd], id_] - pp[tt[:, td - 1], id_]

    xx = np.zeros((pd, np_, nt))
    vp = np.zeros((pd, np_))

    for ti in range(nt):
        for id_ in range(pd):
            vp[id_, :] = pi[:, id_] - pp[tt[ti, td - 1], id_]
        # Solve linear systems (LU equivalent)
        xx[:, :, ti] = np.linalg.solve(mm[:, :, ti], vp)

    in_mask = np.all(xx >= -(np.finfo(float).eps ** 0.8), axis=0) & (
        np.sum(xx, axis=0) <= 1.0 + np.finfo(float).eps ** 0.8
    )

    ii, jj = np.where(in_mask.T)
    return ii, jj
