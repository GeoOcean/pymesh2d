import numpy as np
from .inpoly_mat import inpoly_mat

def inpoly(vert, node, edge=None, ftol=None):
    """
    inpoly compute "points-in-polygon" queries.
    
    Parameters
    ----------
    vert : (N, 2) ndarray
        Query points.
    node : (M, 2) ndarray
        Polygon vertices.
    edge : (P, 2) ndarray, optional
        Polygon edges (indices into node). If None, polygon assumed closed in order.
    ftol : float, optional
        Tolerance for boundary tests. Default = eps**0.85.

    Returns
    -------
    STAT : (N,) boolean ndarray
        True if inside polygon.
    BNDS : (N,) boolean ndarray
        True if on polygon boundary.
    """

    node = np.asarray(node, dtype=float)
    vert = np.asarray(vert, dtype=float)
    if edge is None:
        nnod = node.shape[0]
        edge = np.vstack([np.column_stack([np.arange(nnod - 1), np.arange(1, nnod)]),
                          [nnod - 1, 0]])
    else:
        edge = np.asarray(edge, dtype=int)

    if ftol is None:
        ftol = np.finfo(float).eps ** 0.85

    nnod = node.shape[0]
    nvrt = vert.shape[0]

    # --- Basic checks
    if edge.min() < 0 or edge.max() > nnod:
        raise ValueError("inpoly: invalid EDGE input array.")

    STAT = np.zeros(nvrt, dtype=bool)
    BNDS = np.zeros(nvrt, dtype=bool)

    # --- prune points using bbox
    nmin = node.min(axis=0)
    nmax = node.max(axis=0)
    ddxy = nmax - nmin
    lbar = ddxy.sum() / 2.0
    veps = ftol * lbar

    mask = ((vert[:, 0] >= nmin[0] - veps) &
            (vert[:, 0] <= nmax[0] + veps) &
            (vert[:, 1] >= nmin[1] - veps) &
            (vert[:, 1] <= nmax[1] + veps))

    if not np.any(mask):
        return STAT, BNDS

    vmask = np.where(mask)[0]
    vsub = vert[mask, :].copy()
    nsub = node.copy()

    # --- flip if needed (ensure y-axis is "long")
    vmin = vsub.min(axis=0)
    vmax = vsub.max(axis=0)
    ddxy = vmax - vmin
    if ddxy[0] > ddxy[1]:
        vsub = vsub[:, [1, 0]]
        nsub = nsub[:, [1, 0]]

    # --- reorder edges to point upwards
    #edge = np.atleast_2d(edge) 
    swap = nsub[edge[:, 1], 1] < nsub[edge[:, 0], 1]
    edge[swap] = edge[swap][:, [1, 0]]

    ivec = np.lexsort((vsub[:, 0], vsub[:, 1]))
    vsub = vsub[ivec, :]

    # --- call the "matlab-like" core routine
    stat, bnds = inpoly_mat(vsub, nsub, edge, ftol, lbar)

    # --- invert sorting
    inv = np.argsort(ivec)
    stat = stat[inv]
    bnds = bnds[inv]

    STAT[vmask] = stat
    BNDS[vmask] = bnds

    return STAT, BNDS