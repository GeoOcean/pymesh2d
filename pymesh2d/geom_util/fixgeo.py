import numpy as np

from pymesh2d.aabb_tree.findball import findball
from pymesh2d.aabb_tree.findline import findline
from pymesh2d.aabb_tree.lineline import lineline
from pymesh2d.aabb_tree.linenear import linenear


def fixgeo(node, edge, part):
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
    if edge is None:
        nnum = node.shape[0]
        edge = np.vstack(
            [
                np.column_stack([np.arange(0, nnum - 1), np.arange(1, nnum)]),
                [nnum - 1, 0],
            ]
        )
    else:
        edge = np.asarray(edge, dtype=int)
    # ---------------------------------------------- default PART
    if part is None:
        enum = edge.shape[0]
        part = [np.arange(enum)]

    # ---------------------------------------------- basic checks
    if not (
        isinstance(node, np.ndarray)
        and isinstance(edge, np.ndarray)
        and isinstance(part, (list, tuple))
    ):
        raise TypeError("fixgeo: incorrect input class")

    if node.ndim != 2 or edge.ndim != 2:
        raise ValueError("fixgeo: incorrect dimensions")
    if node.shape[1] != 2 or edge.shape[1] != 2:
        raise ValueError("fixgeo: incorrect dimensions")

    nnum = node.shape[0]
    enum = edge.shape[0]
    if edge.min() < 0 or edge.max() >= nnum:
        raise ValueError("fixgeo: invalid EDGE input array")

    pmin = [np.min(p) for p in part]
    pmax = [np.max(p) for p in part]
    if np.min(pmin) < 0 or np.max(pmax) >= enum:
        raise ValueError("fixgeo: invalid PART input array")

    # ------------------------------------ try to "fix" geometry
    while True:
        nnum = node.shape[0]
        enum = edge.shape[0]
        # --------------------------------- prune redundant nodes
        node, edge, part = prunenode(node, edge, part)
        # -------------------------------- prune redundant edges
        node, edge, part = pruneedge(node, edge, part)
        # --------------------------------- node//edge intersect!
        done = False
        while not done:
            node, edge, part, done = splitnode(node, edge, part)
        # -------------------------------- edge//edge intersect!
        done = False
        while not done:
            node, edge, part, done = splitedge(node, edge, part)
        # --------------------------------- iterate if any change
        if node.shape[0] == nnum and edge.shape[0] == enum:
            break

    return node, edge, part


def prunenode(node, edge, part):
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
    vp, vi, _ = findball(ball, node)
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
    next_id = 0
    for kk in range(vp.shape[0]):
        if izip[kk] == 0:
            imap[kk] = next_id
            next_id += 1

    imap[izip != 0] = imap[izip[izip != 0]]
    edge = imap[edge]
    node = node[izip == 0, :]

    return node, edge, part


def pruneedge(node, edge, part):
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

    edge_sorted = np.sort(edge, axis=1)
    # ------------------------------------- prune redundant topo.
    _, ivec, jvec = np.unique(
        edge_sorted, axis=0, return_index=True, return_inverse=True
    )
    edge = edge[ivec, :]

    for i, p in enumerate(part):
        # --------------------------------- re-index part labels!
        part[i] = np.unique(jvec[p])

    # ------------------------------------ prune collapsed topo.
    keep = edge[:, 0] != edge[:, 1]
    jvec = np.zeros(edge.shape[0], dtype=int)
    jvec[keep] = 1
    jvec = np.cumsum(jvec)
    # Adjust for 0-indexed Python: subtract 1 from non-zero values
    jvec[jvec > 0] = jvec[jvec > 0] - 1
    edge = edge[keep, :]

    for i, p in enumerate(part):
        # --------------------------------- re-index part labels!
        part[i] = np.unique(jvec[p])

    return node, edge, part


def splitnode(node, edge, part):
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
    mark = np.zeros(edge.shape[0], dtype=bool)
    ediv = np.zeros(edge.shape[0], dtype=int)
    pair = []
    # ------------------------------------- node//edge intersect!
    lp, li, _ = findline(node[edge[:, 0], :], node[edge[:, 1], :], node)
    # ------------------------------------- node//edge splitting!
    for ii in range(lp.shape[0]):
        for ip in range(lp[ii, 0], lp[ii, 1] + 1):
            jj = li[ip]
            ni, nj = edge[jj]
            if ni != ii and nj != ii and not mark[jj]:
                done = False
                # ----------------------------- mark seen, descendent
                mark[jj] = True
                pair.append([jj, ii])

    if not pair:
        return node, edge, part, done
    # ------------------------------------- re-index intersection
    pair = np.array(pair)
    inod = edge[pair[:, 0], 0]
    jnod = edge[pair[:, 0], 1]
    xnod = pair[:, 1]

    ediv[pair[:, 0]] = np.arange(pair.shape[0]) + edge.shape[0]
    edge[pair[:, 0], 0] = inod
    edge[pair[:, 0], 1] = xnod
    edge = np.vstack([edge, np.column_stack([xnod, jnod])])
    # ------------------------------------- re-index edge in part
    for i, p in enumerate(part):
        enew = ediv[p]
        part[i] = np.hstack([p, enew[enew != 0]])

    return node, edge, part, done


def splitedge(node, edge, part):
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
    mark = np.zeros(edge.shape[0], dtype=int)
    pair = []
    ediv = np.zeros(edge.shape[0], dtype=int)
    flag = np.zeros(node.shape[0], dtype=int)
    # ------------------------------------- edge//edge intersect!
    lp, li = lineline(
        node[edge[:, 0], :],
        node[edge[:, 1], :],
        node[edge[:, 0], :],
        node[edge[:, 1], :],
    )
    # ---------------------------------- parse NaN delimited data
    for ii in range(lp.shape[0]):
        # Mark nodes belonging to edge ii
        flag[edge[ii, 0]] = ii
        flag[edge[ii, 1]] = ii

        for ip in range(lp[ii, 0], lp[ii, 1] + 1):
            jj = li[ip]
            if mark[ii] == 0 and mark[jj] == 0 and ii != jj:
                ni = edge[jj, 0]
                nj = edge[jj, 1]
                # Check if nodes of edge jj are not endpoints of edge ii
                if flag[ni] != ii and flag[nj] != ii:
                    done = False
                    # ------------------------- mark seen, edge-pairs
                    mark[ii] = 1
                    mark[jj] = 1
                    pair.append([ii, jj])

    if not pair:
        return node, edge, part, done
    # ------------------------------------- re-index intersection
    pair = np.array(pair)
    okay, tval, sval = linenear(
        node[edge[pair[:, 0], 0], :],
        node[edge[pair[:, 0], 1], :],
        node[edge[pair[:, 1], 0], :],
        node[edge[pair[:, 1], 1], :],
    )

    pmid = 0.5 * (node[edge[pair[:, 0], 1], :] + node[edge[pair[:, 0], 0], :])
    pdel = 0.5 * (node[edge[pair[:, 0], 1], :] - node[edge[pair[:, 0], 0], :])
    ppos = pmid + np.column_stack([tval, tval]) * pdel

    qmid = 0.5 * (node[edge[pair[:, 1], 1], :] + node[edge[pair[:, 1], 0], :])
    qdel = 0.5 * (node[edge[pair[:, 1], 1], :] - node[edge[pair[:, 1], 0], :])
    qpos = qmid + np.column_stack([sval, sval]) * qdel

    # Save original edge endpoints before modification
    inod = edge[pair[:, 0], 0].copy()
    jnod = edge[pair[:, 0], 1].copy()
    anod = edge[pair[:, 1], 0].copy()
    bnod = edge[pair[:, 1], 1].copy()

    # Compute new node indices (before adding nodes to array)
    nn = pair.shape[0]
    xnod = node.shape[0] + np.arange(nn)  # 0-indexed for Python

    # Compute new edge indices (before modifying edge array)
    # MATLAB: iedg = (1:nn)' + size(PSLG,1) + 0 * size(pair,1)
    # MATLAB: jedg = (1:nn)' + size(PSLG,1) + 1 * size(pair,1)
    # Python: convert to 0-indexed
    base_idx = edge.shape[0]
    iedg = base_idx + np.arange(nn)  # 0-indexed for Python
    jedg = base_idx + nn + np.arange(nn)  # 0-indexed for Python

    # Set ediv mapping
    ediv[pair[:, 0]] = iedg
    ediv[pair[:, 1]] = jedg

    # Update first edges in pairs: [inod, xnod]
    edge[pair[:, 0], 0] = inod
    edge[pair[:, 0], 1] = xnod

    # Update second edges in pairs: [anod, xnod]
    edge[pair[:, 1], 0] = anod
    edge[pair[:, 1], 1] = xnod

    # Add new edges: [xnod, jnod] and [xnod, bnod]
    edge = np.vstack([edge, np.column_stack([xnod, jnod])])
    edge = np.vstack([edge, np.column_stack([xnod, bnod])])

    # Add new nodes (must be done last, after computing all indices)
    node = np.vstack([node, 0.5 * (ppos + qpos)])

    # ------------------------------------ re-index edge in part
    for i, p in enumerate(part):
        enew = ediv[p]
        enew = enew[enew != 0]
        part[i] = np.hstack([p, enew])

    return node, edge, part, done
