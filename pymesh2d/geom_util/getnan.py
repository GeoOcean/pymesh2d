import numpy as np


def getnan(data=None, filt=0.0):
    """
    Parse a NaN-delimited polygon definition into a PSLG (Planar Straight Line Graph).

    This function converts a list of polygons separated by NaN rows into a
    PSLG representation suitable for meshing or geometric processing.
    Each polygon is defined by a sequence of vertex coordinates, with `NaN`
    values marking the breaks between individual polygons. Optionally,
    small polygons can be filtered out based on a minimum bounding-box size.

    Parameters
    ----------
    NANS : ndarray of shape (D, 2)
        Array of polygon vertex coordinates.
        Consecutive vertices define polygon edges, and rows containing NaN
        values indicate breaks between polygons.
    FILT : array_like of length 2
        Minimum axis-aligned feature size `[dx_min, dy_min]`.
        Polygons whose extents are smaller than these thresholds are removed
        from the output.

    Returns
    -------
    NODE : ndarray of shape (N, 2)
        Array of unique vertex coordinates for the PSLG.
    EDGE : ndarray of shape (E, 2)
        Array of edge connectivity.
        Each row defines one polygon edge as `[start_vertex, end_vertex]`, such that
        `NODE[EDGE[j, 0], :]` and `NODE[EDGE[j, 1], :]` are the endpoints
        of the j-th edge.

    Notes
    -----
    - Input polygons must be defined in vertex order (either clockwise or counterclockwise).
    - NaN rows are used as delimiters between individual polygons.
    - Very small polygons, defined by bounding boxes smaller than the `FILT` thresholds,
      are discarded automatically.
    - The resulting PSLG (`NODE`, `EDGE`) can be passed directly to mesh generators
      such as `refine`.

    References
    ----------
    Translation of the MESH2D function `GETNAN2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    See also
    --------
    fixgeo : Repair geometry definitions.
    bfsgeo : Partition polygonal geometry via breadth-first traversal.
    refine : Generate constrained Delaunay triangulations.
    """

    # --------------------------------------------- basic checks
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

    # --------------------------------- parse NaN delimited data
    nvec = np.where(np.isnan(data[:, 0]))[0].tolist()

    if len(nvec) == 0:  # no NaN's at all!
        nvec = [data.shape[0]]

    if nvec[-1] != data.shape[0]:  # append last poly
        nvec.append(data.shape[0])

    node_list = []
    edge_list = []
    next_idx = 0
    nout = 0

    for stop in nvec:
        pnew = data[next_idx:stop, :]
        next_idx = stop + 1

        if pnew.size == 0:
            continue

        pmin = pnew.min(axis=0)
        pmax = pnew.max(axis=0)
        pdel = pmax - pmin

        if np.any(pdel > filt):
            # --------------------------------- push polygon onto output
            nnew = pnew.shape[0]

            enew = np.vstack(
                [
                    np.column_stack([np.arange(0, nnew - 1), np.arange(1, nnew)]),
                    [nnew - 1, 0],
                ]
            )
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
