import numpy as np

def fixgeo(node=None, PSLG=None, part=None,
            findball=None, findline=None, lineline=None, linenear=None):
    """
    fixgeo attempts to "fix" issues with geometry definitions.

    Parameters
    ----------
    node : ndarray (N, 2)
        Polygon vertices.
    PSLG : ndarray (E, 2)
        Polygon edges, each row is a pair of indices into `node`.
    part : list of arrays
        Each entry is a list of edge indices defining one partition.
    findball, findline, lineline, linenear : callables
        External helper functions, must be provided.

    Returns
    -------
    node : ndarray
        Repaired vertices.
    PSLG : ndarray
        Repaired edges.
    part : list of arrays
        Repaired partitions.
    """

    if node is None:
        return None, None, []

    node = np.asarray(node)
    if PSLG is None:
        nnum = node.shape[0]
        PSLG = np.vstack([np.column_stack([np.arange(0, nnum - 1),
                                           np.arange(1, nnum)]),
                          [nnum - 1, 0]])
    else:
        PSLG = np.asarray(PSLG, dtype=int)

    if part is None:
        enum = PSLG.shape[0]
        part = [np.arange(enum)]

    # ---------- basic checks
    if not (isinstance(node, np.ndarray) and
            isinstance(PSLG, np.ndarray) and
            isinstance(part, (list, tuple))):
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

    # ---------- main loop
    while True:
        nnum = node.shape[0]
        enum = PSLG.shape[0]

        node, PSLG, part = prunenode(node, PSLG, part, findball)
        node, PSLG, part = pruneedge(node, PSLG, part)

        done = False
        while not done:
            node, PSLG, part, done = splitnode(node, PSLG, part, findline)

        done = False
        while not done:
            node, PSLG, part, done = splitedge(node, PSLG, part, lineline, linenear)

        if node.shape[0] == nnum and PSLG.shape[0] == enum:
            break

    return node, PSLG, part


def prunenode(node, PSLG, part, findball):
    done = True
    nmin = node.min(axis=0)
    nmax = node.max(axis=0)
    ndel = nmax - nmin
    ztol = np.finfo(float).eps ** 0.80
    zlen = ztol * np.max(ndel)

    ball = np.column_stack([node, np.full(node.shape[0], zlen ** 2)])
    vp, vi = findball(ball, node)

    iv = np.argsort(vp[:, 1] - vp[:, 0])
    izip = np.zeros(node.shape[0], dtype=int)
    imap = np.zeros(node.shape[0], dtype=int)

    for ii in iv[::-1]:
        for ip in range(vp[ii, 0], vp[ii, 1] + 1):
            jj = vi[ip]
            if izip[ii] == 0 and izip[jj] == 0 and ii != jj:
                done = False
                izip[jj] = ii

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
    PSLG_sorted = np.sort(PSLG, axis=1)
    _, ivec, jvec = np.unique(PSLG_sorted, axis=0, return_index=True, return_inverse=True)
    PSLG = PSLG[ivec, :]

    for i, p in enumerate(part):
        part[i] = np.unique(jvec[p])

    keep = PSLG[:, 0] != PSLG[:, 1]
    jvec = np.zeros(PSLG.shape[0], dtype=int)
    jvec[keep] = 1
    jvec = np.cumsum(jvec)
    PSLG = PSLG[keep, :]

    for i, p in enumerate(part):
        part[i] = np.unique(jvec[p])

    return node, PSLG, part


def splitnode(node, PSLG, part, findline):
    done = True
    mark = np.zeros(PSLG.shape[0], dtype=bool)
    ediv = np.zeros(PSLG.shape[0], dtype=int)
    pair = []

    lp, li = findline(node[PSLG[:, 0], :], node[PSLG[:, 1], :], node)

    for ii in range(lp.shape[0]):
        for ip in range(lp[ii, 0], lp[ii, 1] + 1):
            jj = li[ip]
            ni, nj = PSLG[jj]
            if ni != ii and nj != ii and not mark[jj]:
                done = False
                mark[jj] = True
                pair.append([jj, ii])

    if not pair:
        return node, PSLG, part, done

    pair = np.array(pair)
    inod = PSLG[pair[:, 0], 0]
    jnod = PSLG[pair[:, 0], 1]
    xnod = pair[:, 1]

    ediv[pair[:, 0]] = np.arange(1, pair.shape[0] + 1) + PSLG.shape[0]
    PSLG[pair[:, 0], 0] = inod
    PSLG[pair[:, 0], 1] = xnod
    PSLG = np.vstack([PSLG, np.column_stack([xnod, jnod])])

    for i, p in enumerate(part):
        enew = ediv[p]
        part[i] = np.hstack([p, enew[enew != 0]])

    return node, PSLG, part, done


def splitedge(node, PSLG, part, lineline, linenear):
    done = True
    mark = np.zeros(PSLG.shape[0], dtype=int)
    pair = []
    ediv = np.zeros(PSLG.shape[0], dtype=int)

    lp, li = lineline(node[PSLG[:, 0], :], node[PSLG[:, 1], :],
                      node[PSLG[:, 0], :], node[PSLG[:, 1], :])

    for ii in range(lp.shape[0]):
        for ip in range(lp[ii, 0], lp[ii, 1] + 1):
            jj = li[ip]
            if mark[ii] == 0 and mark[jj] == 0 and ii != jj:
                done = False
                mark[ii] = 1
                mark[jj] = 1
                pair.append([ii, jj])

    if not pair:
        return node, PSLG, part, done

    pair = np.array(pair)
    okay, tval, sval = linenear(node[PSLG[pair[:, 0], 0], :],
                                node[PSLG[pair[:, 0], 1], :],
                                node[PSLG[pair[:, 1], 0], :],
                                node[PSLG[pair[:, 1], 1], :])

    pmid = 0.5 * (node[PSLG[pair[:, 0], 1], :] + node[PSLG[pair[:, 0], 0], :])
    pdel = 0.5 * (node[PSLG[pair[:, 0], 1], :] - node[PSLG[pair[:, 0], 0], :])
    ppos = pmid + np.column_stack([tval, tval]) * pdel

    qmid = 0.5 * (node[PSLG[pair[:, 1], 1], :] + node[PSLG[pair[:, 1], 0], :])
    qdel = 0.5 * (node[PSLG[pair[:, 1], 1], :] - node[PSLG[pair[:, 1], 0], :])
    qpos = qmid + np.column_stack([sval, sval]) * qdel

    xnod = np.arange(1, pair.shape[0] + 1) + node.shape[0]
    PSLG = np.vstack([PSLG,
                      np.column_stack([xnod, PSLG[pair[:, 0], 1]]),
                      np.column_stack([xnod, PSLG[pair[:, 1], 1]])])
    node = np.vstack([node, 0.5 * (ppos + qpos)])

    for i, p in enumerate(part):
        enew = ediv[p]
        part[i] = np.hstack([p, enew[enew != 0]])

    return node, PSLG, part, done
