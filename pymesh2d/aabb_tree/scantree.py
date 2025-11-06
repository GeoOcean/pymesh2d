import numpy as np


def scantree(tr, pi, fn):
    """
    SCANTREE find the tree-to-item mappings.

    Parameters
    ----------
    tr : dict
        AABB tree with fields 'xx', 'ii', 'll'.
    pi : ndarray (N, d)
        Query collection.
    fn : callable
        Partition function with signature j1, j2 = fn(pj, ni, nj).

    Returns
    -------
    tm : dict
        Tree-to-item mapping:
        - 'ii': array of tree indices
        - 'll': list of item lists
    im : dict
        Item-to-tree mapping:
        - 'ii': array of item indices
        - 'll': list of node lists

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

    tm = {"ii": [], "ll": []}
    im = {"ii": [], "ll": []}

    # Quick return
    if pi is None or len(pi) == 0:
        return tm, im
    if tr is None or len(tr) == 0:
        return tm, im

    if not isinstance(tr, dict) or not isinstance(pi, np.ndarray) or not callable(fn):
        raise TypeError("scantree: incorrect input class.")

    if not all(k in tr for k in ("xx", "ii", "ll")):
        raise ValueError("scantree: incorrect AABB struct.")

    n_nodes = tr["ii"].shape[0]

    tm["ii"] = np.zeros(n_nodes, dtype=int)
    tm["ll"] = [None] * n_nodes

    ss = np.zeros(n_nodes, dtype=int)  # stack of nodes
    sl = [None] * n_nodes  # stack of item lists
    sl[0] = np.arange(pi.shape[0])  # root has all items

    tf = np.array([len(x) > 0 for x in tr["ll"]])

    ss[0] = 0  # root index (Python 0-based)
    ns = 1
    no = 0

    # Traverse tree
    while ns > 0:
        ns -= 1
        ni = ss[ns]  # pop

        if tf[ni]:
            tm["ii"][no] = ni
            tm["ll"][no] = sl[ns]
            no += 1

        if tr["ii"][ni, 1] != 0:  # has children
            c1 = tr["ii"][ni, 1]  # first child index
            c2 = tr["ii"][ni, 1] + 1  # second child index

            j1, j2 = fn(pi[sl[ns], :], tr["xx"][c1, :], tr["xx"][c2, :])

            l1 = sl[ns][j1]
            l2 = sl[ns][j2]

            if l1.size > 0:
                ss[ns] = c1
                sl[ns] = l1
                ns += 1
            if l2.size > 0:
                ss[ns] = c2
                sl[ns] = l2
                ns += 1

    # Trim allocation
    tm["ii"] = tm["ii"][:no]
    tm["ll"] = tm["ll"][:no]

    # If only tm requested
    if tm and im is None:
        return tm

    # Inverse map
    ic = []
    jc = tm["ll"]

    for ip in range(no):
        ni = tm["ii"][ip]
        ic.append(np.full(len(jc[ip]), ni, dtype=int))

    if len(ic) == 0:
        return tm, im

    ii = np.concatenate(ic)
    jj = np.concatenate(jc)

    im["ll"] = [None] * pi.shape[0]

    # Sort by jj
    jx = np.argsort(jj)
    jj = jj[jx]
    ii = ii[jx]

    diff_idx = np.nonzero(np.diff(jj) != 0)[0]
    im["ii"] = np.concatenate((jj[diff_idx], [jj[-1]]))
    bounds = np.concatenate(([0], diff_idx + 1, [len(ii)]))

    for ip in range(len(im["ii"])):
        im["ll"][ip] = ii[bounds[ip] : bounds[ip + 1]]

    return tm, im
