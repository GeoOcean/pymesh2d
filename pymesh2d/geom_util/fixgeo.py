import numpy as np


def fixgeo(
    node=None,
    PSLG=None,
    part=None,
    findball=None,
    findline=None,
    lineline=None,
    linenear=None,
):
    """
    Attempt to repair issues in polygonal geometry definitions.

    This function takes an input polygonal geometry and performs several
    corrective operations to ensure a valid and consistent geometric structure.
    The output is a "repaired" version of the input nodes, edges, and parts.

    Parameters
    ----------
    NODE : ndarray of shape (N, 2)
        XY-coordinates of the polygon vertices.
    EDGE : ndarray of shape (E, 2)
        Array of polygon edge indices. Each row defines one edge as
        `[start_vertex, end_vertex]`.
    PART : list of lists or list of ndarrays
        Geometry partitions, where each element contains indices into `EDGE`
        defining one polygonal region.

    Returns
    -------
    NNEW : ndarray of shape (N', 2)
        Repaired vertex coordinates with redundant nodes merged.
    ENEW : ndarray of shape (E', 2)
        Updated edge connectivity with duplicates removed and intersections resolved.
    PNEW : list of lists or list of ndarrays
        Updated geometry partitions consistent with the repaired edges.

    Notes
    -----
    The following operations are performed:

    1. Redundant (coincident) nodes are merged ("zipped" together).
    2. Duplicate edges are removed.
    3. Edges are split where they intersect existing nodes.
    4. Edges are split where they intersect other edges.

    These operations ensure the geometry is topologically valid and suitable
    for constrained Delaunay triangulation and mesh refinement.

    References
    ----------
    Translation of the MESH2D function `FIXGEO2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    See also
    --------
    refine : Generate constrained Delaunay triangulations.
    bfsgio : Partition geometries using breadth-first traversal.
    """

    if node is None:
        return None, None, []
    # ---------------------------------------------- extract ARGS
    node = np.asarray(node)
    if PSLG is None:
        nnum = node.shape[0]
        PSLG = np.vstack(
            [
                np.column_stack([np.arange(0, nnum - 1), np.arange(1, nnum)]),
                [nnum - 1, 0],
            ]
        )
    else:
        PSLG = np.asarray(PSLG, dtype=int)
    # ---------------------------------------------- default PART
    if part is None:
        enum = PSLG.shape[0]
        part = [np.arange(enum)]

    # ---------------------------------------------- basic checks
    if not (
        isinstance(node, np.ndarray)
        and isinstance(PSLG, np.ndarray)
        and isinstance(part, (list, tuple))
    ):
        raise TypeError("fixgeo: incorrect input class")

    if node.ndim != 2 or PSLG.ndim != 2:
        raise ValueError("fixgeo: incorrect dimensions")
    if node.shape[1] != 2 or PSLG.shape[1] != 2:
        raise ValueError("fixgeo: incorrect dimensions")

    nnum = node.shape[0]
    enum = PSLG.shape[0]
    if PSLG.min() < 0 or PSLG.max() >= nnum:
        raise ValueError("fixgeo: invalid EDGE input array")

    pmin = [np.min(p) for p in part]
    pmax = [np.max(p) for p in part]
    if np.min(pmin) < 0 or np.max(pmax) >= enum:
        raise ValueError("fixgeo: invalid PART input array")

    # ------------------------------------ try to "fix" geometry
    while True:
        nnum = node.shape[0]
        enum = PSLG.shape[0]
        # --------------------------------- prune redundant nodes
        node, PSLG, part = prunenode(node, PSLG, part, findball)
        # -------------------------------- prune redundant edges
        node, PSLG, part = pruneedge(node, PSLG, part)
        # --------------------------------- node//edge intersect!
        done = False
        while not done:
            node, PSLG, part, done = splitnode(node, PSLG, part, findline)
        # -------------------------------- edge//edge intersect!
        done = False
        while not done:
            node, PSLG, part, done = splitedge(node, PSLG, part, lineline, linenear)
        # --------------------------------- iterate if any change
        if node.shape[0] == nnum and PSLG.shape[0] == enum:
            break

    return node, PSLG, part


def prunenode(node, PSLG, part, findball):
    """
    PRUNENODE "prune" redundant nodes by "zipping" those within tolerance of each other.

    Parameters
    ----------
    NODE : ndarray of shape (N, 2)
        XY-coordinates of the polygon vertices.
    EDGE : ndarray of shape (E, 2)
        Array of polygon edge indices. Each row defines one edge as
        `[start_vertex, end_vertex]`.
    PART : list of lists or list of ndarrays
        Geometry partitions, where each element contains indices into `EDGE`
        defining one polygonal region.
    findball : function
        Function to find neighboring nodes within a specified tolerance.

    Returns
    -------
    NODE : ndarray of shape (N', 2)
        Updated vertex coordinates with redundant nodes removed.
    EDGE : ndarray of shape (E, 2)
        Updated edge connectivity.
    PART : list of lists or list of ndarrays
        Updated geometry partitions.

    References
    ----------
    Translation of the MESH2D function `PRUNENODE`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    _done = True
    # ------------------------------------- calc. "zip" tolerance
    nmin = node.min(axis=0)
    nmax = node.max(axis=0)
    ndel = nmax - nmin
    ztol = np.finfo(float).eps ** 0.80
    zlen = ztol * np.max(ndel)
    # ------------------------------------- index clustered nodes
    ball = np.column_stack([node, np.full(node.shape[0], zlen**2)])
    vp, vi = findball(ball, node)
    # ------------------------------------- "zip" clustered nodes
    iv = np.argsort(vp[:, 1] - vp[:, 0])
    izip = np.zeros(node.shape[0], dtype=int)
    imap = np.zeros(node.shape[0], dtype=int)

    for ii in iv[::-1]:
        for ip in range(vp[ii, 0], vp[ii, 1] + 1):
            jj = vi[ip]
            if izip[ii] == 0 and izip[jj] == 0 and ii != jj:
                _done = False
                # ----------------------------- "zip" node JJ into II
                izip[jj] = ii
    # ------------------------------------- re-index nodes//edges
    next_id = 1
    for kk in range(vp.shape[0]):
        if izip[kk] == 0:
            imap[kk] = next_id
            next_id += 1

    imap[izip != 0] = imap[izip[izip != 0]]
    PSLG = imap[PSLG]
    node = node[izip == 0, :]

    return node, PSLG, part


def pruneedge(node, PSLG, part):
    """
    PRUNEEDGE "prune" redundant topology.

    Parameters
    ----------
    NODE : ndarray of shape (N, 2)
        XY-coordinates of the polygon vertices.
    EDGE : ndarray of shape (E, 2)
        Array of polygon edge indices. Each row defines one edge as
        `[start_vertex, end_vertex]`.
    PART : list of lists or list of ndarrays
        Geometry partitions, where each element contains indices into `EDGE`
        defining one polygonal region.

    Returns
    -------
    NODE : ndarray of shape (N, 2)
        Updated vertex coordinates with redundant nodes removed.
    EDGE : ndarray of shape (E, 2)
        Updated edge connectivity.
    PART : list of lists or list of ndarrays
        Updated geometry partitions.

    References
    ----------
    Translation of the MESH2D function `PRUNEEDGE`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    PSLG_sorted = np.sort(PSLG, axis=1)
    # ------------------------------------- prune redundant topo.
    _, ivec, jvec = np.unique(
        PSLG_sorted, axis=0, return_index=True, return_inverse=True
    )
    PSLG = PSLG[ivec, :]

    for i, p in enumerate(part):
        # --------------------------------- re-index part labels!
        part[i] = np.unique(jvec[p])

    # ------------------------------------ prune collapsed topo.
    keep = PSLG[:, 0] != PSLG[:, 1]
    jvec = np.zeros(PSLG.shape[0], dtype=int)
    jvec[keep] = 1
    jvec = np.cumsum(jvec)
    PSLG = PSLG[keep, :]

    for i, p in enumerate(part):
        # --------------------------------- re-index part labels!
        part[i] = np.unique(jvec[p])

    return node, PSLG, part


def splitnode(node, PSLG, part, findline):
    """
    SPLITNODE "split" PSLG about intersecting nodes.

    Parameters
    ----------
    NODE : ndarray of shape (N, 2)
        XY-coordinates of the polygon vertices.
    EDGE : ndarray of shape (E, 2)
        Array of polygon edge indices. Each row defines one edge as
        `[start_vertex, end_vertex]`.
    PART : list of lists or list of ndarrays
        Geometry partitions, where each element contains indices into `EDGE`
        defining one polygonal region.
    FINDLINE : callable
        Function to find line intersections.

    Returns
    -------
    NODE : ndarray of shape (N, 2)
        Updated vertex coordinates with new nodes added.
    EDGE : ndarray of shape (E, 2)
        Updated edge connectivity.
    PART : list of lists or list of ndarrays
        Updated geometry partitions.
    DONE : bool
        Flag indicating if any changes were made.

    References
    ----------
    Translation of the MESH2D function `SPLITNODE`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    done = True
    mark = np.zeros(PSLG.shape[0], dtype=bool)
    ediv = np.zeros(PSLG.shape[0], dtype=int)
    pair = []
    # ------------------------------------- node//edge intersect!
    lp, li = findline(node[PSLG[:, 0], :], node[PSLG[:, 1], :], node)
    # ------------------------------------- node//edge splitting!
    for ii in range(lp.shape[0]):
        for ip in range(lp[ii, 0], lp[ii, 1] + 1):
            jj = li[ip]
            ni, nj = PSLG[jj]
            if ni != ii and nj != ii and not mark[jj]:
                done = False
                # ----------------------------- mark seen, descendent
                mark[jj] = True
                pair.append([jj, ii])

    if not pair:
        return node, PSLG, part, done
    # ------------------------------------- re-index intersection
    pair = np.array(pair)
    inod = PSLG[pair[:, 0], 0]
    jnod = PSLG[pair[:, 0], 1]
    xnod = pair[:, 1]

    ediv[pair[:, 0]] = np.arange(1, pair.shape[0] + 1) + PSLG.shape[0]
    PSLG[pair[:, 0], 0] = inod
    PSLG[pair[:, 0], 1] = xnod
    PSLG = np.vstack([PSLG, np.column_stack([xnod, jnod])])
    # ------------------------------------- re-index edge in part
    for i, p in enumerate(part):
        enew = ediv[p]
        part[i] = np.hstack([p, enew[enew != 0]])

    return node, PSLG, part, done


def splitedge(node, PSLG, part, lineline, linenear):
    """
    SPLITEDGE "split" PSLG about intersecting edges.

    Parameters
    ----------
    NODE : ndarray of shape (N, 2)
        XY-coordinates of the polygon vertices.
    EDGE : ndarray of shape (E, 2)
        Array of polygon edge indices. Each row defines one edge as
        `[start_vertex, end_vertex]`.
    PART : list of lists or list of ndarrays
        Geometry partitions, where each element contains indices into `EDGE`
        defining one polygonal region.
    LINELINE : callable
        Function to find line intersections.
    LINENEAR : callable
        Function to find linear intersections.

    Returns
    -------
    NODE : ndarray of shape (N, 2)
        Updated vertex coordinates with new nodes added.
    EDGE : ndarray of shape (E, 2)
        Updated edge connectivity.
    PART : list of lists or list of ndarrays
        Updated geometry partitions.
    DONE : bool
        Flag indicating if any changes were made.

    References
    ----------
    Translation of the MESH2D function `SPLITEDGE`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    done = True
    mark = np.zeros(PSLG.shape[0], dtype=int)
    pair = []
    ediv = np.zeros(PSLG.shape[0], dtype=int)
    # ------------------------------------- edge//edge intersect!
    lp, li = lineline(
        node[PSLG[:, 0], :],
        node[PSLG[:, 1], :],
        node[PSLG[:, 0], :],
        node[PSLG[:, 1], :],
    )
    # ---------------------------------- parse NaN delimited data
    for ii in range(lp.shape[0]):
        for ip in range(lp[ii, 0], lp[ii, 1] + 1):
            jj = li[ip]
            if mark[ii] == 0 and mark[jj] == 0 and ii != jj:
                done = False
                # ------------------------- mark seen, edge-pairs
                mark[ii] = 1
                mark[jj] = 1
                pair.append([ii, jj])

    if not pair:
        return node, PSLG, part, done
    # ------------------------------------- re-index intersection
    pair = np.array(pair)
    okay, tval, sval = linenear(
        node[PSLG[pair[:, 0], 0], :],
        node[PSLG[pair[:, 0], 1], :],
        node[PSLG[pair[:, 1], 0], :],
        node[PSLG[pair[:, 1], 1], :],
    )

    pmid = 0.5 * (node[PSLG[pair[:, 0], 1], :] + node[PSLG[pair[:, 0], 0], :])
    pdel = 0.5 * (node[PSLG[pair[:, 0], 1], :] - node[PSLG[pair[:, 0], 0], :])
    ppos = pmid + np.column_stack([tval, tval]) * pdel

    qmid = 0.5 * (node[PSLG[pair[:, 1], 1], :] + node[PSLG[pair[:, 1], 0], :])
    qdel = 0.5 * (node[PSLG[pair[:, 1], 1], :] - node[PSLG[pair[:, 1], 0], :])
    qpos = qmid + np.column_stack([sval, sval]) * qdel

    xnod = np.arange(1, pair.shape[0] + 1) + node.shape[0]
    PSLG = np.vstack(
        [
            PSLG,
            np.column_stack([xnod, PSLG[pair[:, 0], 1]]),
            np.column_stack([xnod, PSLG[pair[:, 1], 1]]),
        ]
    )
    node = np.vstack([node, 0.5 * (ppos + qpos)])
    # ------------------------------------ re-index edge in part
    for i, p in enumerate(part):
        enew = ediv[p]
        part[i] = np.hstack([p, enew[enew != 0]])

    return node, PSLG, part, done
