import numpy as np


def queryset(tr, tm, fn, *args):
    """
    QUERYSET spatial queries for AABB-indexed collections.

    Parameters
    ----------
    tr : dict
        AABB-tree with fields 'xx', 'ii', 'll'.
    tm : dict
        Query-to-tree mapping with fields 'ii', 'll'.
    fn : callable
        Intersection kernel function. Must return (qi, qj).
    *args : optional
        Extra arguments passed to fn.

    Returns
    -------
    qi : ndarray
        Array of query indices.
    qp : ndarray
        Index ranges into qj for each query.
    qj : ndarray
        Array of intersecting objects.

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

    # Quick checks
    if tr is None or len(tr) == 0:
        return np.array([]), np.array([]), np.array([])
    if not isinstance(tr, dict) or not isinstance(tm, dict):
        raise TypeError("queryset: incorrect input class.")
    if not all(k in tm for k in ("ii", "ll")):
        raise ValueError("queryset: invalid aabb-maps obj.")
    if not all(k in tr for k in ("xx", "ii", "ll")):
        raise ValueError("queryset: invalid aabb-tree obj.")

    ic = []
    jc = []

    # Loop over tiles
    for ip in range(len(tm["ii"])):
        ni = tm["ii"][ip]  # node index in tree

        # Call kernel function
        qi, qj = fn(
            tm["ll"][ip],  # query in tile
            tr["ll"][ni],  # items in tile
            *args,
        )

        ic.append(np.atleast_1d(qi))
        jc.append(np.atleast_1d(qj))

    # Concatenate results
    if len(ic) == 0 or len(jc) == 0:
        return np.array([]), np.array([]), np.array([])

    qi = np.concatenate(ic)
    qj = np.concatenate(jc)

    if qj.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Sort by query index
    sort_idx = np.argsort(qi)
    qi = qi[sort_idx]
    qj = qj[sort_idx]

    # Find boundaries
    diff_idx = np.nonzero(np.diff(qi) != 0)[0]
    ni = len(qi)

    qi_unique = np.concatenate((qi[diff_idx], [qi[-1]]))
    nj = len(qj)
    ni = len(qi_unique)

    # Build qp indexing
    qp = np.zeros((ni, 2), dtype=int)
    qp[:, 0] = np.concatenate(([0], diff_idx + 1))
    qp[:, 1] = np.concatenate((diff_idx, [nj - 1]))

    return qi_unique, qp, qj
