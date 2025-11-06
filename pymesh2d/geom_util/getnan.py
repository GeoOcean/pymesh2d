import numpy as np

def getnan(data=None, filt=0.0):
    """
    getnan parse a NaN-delimited polygon into a PSLG.

    Parameters
    ----------
    data : ndarray (D, 2)
        Array of coordinates with NaN delimiters between polygons.
    filt : float or (2,)
        Length filter: polygons with axis-aligned extents smaller than `filt`
        are discarded. If scalar, applied to both x and y extents.

    Returns
    -------
    node : ndarray (N, 2)
        Coordinates of vertices.
    edge : ndarray (E, 2)
        PSLG edges between vertices.
    """

    if data is None:
        return np.zeros((0, 2)), np.zeros((0, 2), dtype=int)

    data = np.asarray(data, dtype=float)

    if not (isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating)):
        raise TypeError("getnan: incorrect input class")

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("getnan: incorrect dimensions")

    filt = np.atleast_1d(filt).astype(float)
    if filt.size == 1:
        filt = np.repeat(filt, 2)
    elif filt.size > 2:
        raise ValueError("getnan: filt must be scalar or length 2")

    # Find NaN delimiters
    nvec = np.where(np.isnan(data[:, 0]))[0].tolist()

    if len(nvec) == 0:
        nvec = [data.shape[0]]

    if nvec[-1] != data.shape[0]:
        nvec.append(data.shape[0])

    node_list = []
    edge_list = []
    next_idx = 0
    nout = 0

    # Parse polygons
    for stop in nvec:
        pnew = data[next_idx:stop, :]
        next_idx = stop + 1  # skip the NaN row

        if pnew.size == 0:
            continue

        pmin = pnew.min(axis=0)
        pmax = pnew.max(axis=0)
        pdel = pmax - pmin

        # Keep only "large enough" polygons
        if np.any(pdel > filt):
            nnew = pnew.shape[0]

            # Build edges
            enew = np.vstack([
                np.column_stack([np.arange(0, nnew - 1), np.arange(1, nnew)]),
                [nnew - 1, 0]  # close polygon
            ])
            enew = enew + nout

            node_list.append(pnew)
            edge_list.append(enew)

            nout += nnew

    if node_list:
        node = np.vstack(node_list)
        edge = np.vstack(edge_list)
    else:
        node = np.zeros((0, 2))
        edge = np.zeros((0, 2), dtype=int)

    return node, edge
