import numpy as np


def exchange(pp, px):
    """
    EXCHANGE: Change the "associativity" of a sparse list set.

    Parameters
    ----------
    pp (numpy.ndarray): Sparse list pointers.
        Shape (P,2) array where each row gives the start and end
        indices (1-based) into the `px` array for each list.
    px (numpy.ndarray): Sparse list indices.
        Shape (N,) array containing the items in all lists.
        The items are 1-based indices into the lists defined by `pp`.

    Returns
    -------
    qp (numpy.ndarray): Transposed sparse list pointers.
        Shape (Q,2) array where each row gives the start and end
        indices (1-based) into the `qx` array for each list.
    qx (numpy.ndarray): Transposed sparse list indices.
        Shape (M,) array containing the items in all lists. The items are
        1-based indices into the lists defined by `qp`.

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
    if not isinstance(pp, np.ndarray) or not isinstance(px, np.ndarray):
        raise ValueError("Inputs 'pp' and 'px' must be NumPy arrays.")

    if pp.shape[1] != 2 or px.ndim != 1:
        raise ValueError("Incorrect input dimensions.")

    if np.min(px) <= 0 or np.max(pp) > len(px):
        raise ValueError("Invalid list indexing.")

    # Compute list "transpose"
    qp = np.zeros((np.max(px), 2), dtype=int)
    qx = np.zeros(len(px), dtype=int)

    # Accumulate column count
    for ip in range(len(pp)):
        for ii in range(pp[ip, 0] - 1, pp[ip, 1]):
            qp[px[ii] - 1, 1] += 1

    # Deal with "empty" lists
    Z = qp[:, 1] == 0

    qp[:, 1] = np.cumsum(qp[:, 1])
    qp[:, 0] = np.concatenate(([1], qp[:-1, 1] + 1))

    # Transpose of items array
    for ip in range(len(pp)):
        for ii in range(pp[ip, 0] - 1, pp[ip, 1]):
            qx[qp[px[ii] - 1, 0] - 1] = ip + 1
            qp[px[ii] - 1, 0] += 1

    qp[:, 0] = np.concatenate(([1], qp[:-1, 1] + 1))

    # Deal with "empty" lists
    qp[Z, 0] = 0
    qp[Z, 1] = -1

    return qp, qx
