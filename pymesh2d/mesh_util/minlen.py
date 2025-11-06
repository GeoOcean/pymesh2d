import numpy as np

def minlen(pp, tt):
    """
    minlen return the minimum length edge for each triangle in
    a two-dimensional triangulation.

    Parameters
    ----------
    pp : (N,2) array
        Vertex coordinates.
    tt : (T,>=3) array
        Triangulation connectivity (indices already 0-based).

    Returns
    -------
    ll : (T,) array
        Squared length of the shortest edge in each triangle.
    ei : (T,) array
        Local edge index (0,1,2) of the shortest edge.
    """

    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("minlen:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("minlen:incorrectDimensions")
    if pp.shape[1] != 2 or tt.shape[1] < 3:
        raise ValueError("minlen:incorrectDimensions")

    nnod = pp.shape[0]
    if tt[:, :3].min() < 0 or tt[:, :3].max() >= nnod:
        raise ValueError("minlen:invalidInputs")

    # compute squared edge lengths
    l1 = np.sum((pp[tt[:, 1], :] - pp[tt[:, 0], :])**2, axis=1)
    l2 = np.sum((pp[tt[:, 2], :] - pp[tt[:, 1], :])**2, axis=1)
    l3 = np.sum((pp[tt[:, 0], :] - pp[tt[:, 2], :])**2, axis=1)

    # stack and compute min
    lengths = np.vstack([l1, l2, l3]).T
    ei = np.argmin(lengths, axis=1)
    ll = lengths[np.arange(lengths.shape[0]), ei]

    return ll, ei