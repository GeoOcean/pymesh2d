import numpy as np

def relhfn(vert, tria, hvrt):
    """
    relhfn calc. relative edge-length for a 2-simplex triangulation.

    Parameters
    ----------
    vert : (V,2) array
        Vertex coordinates.
    tria : (T,3) array
        Triangle connectivity.
    hvrt : (V,1) array
        Mesh-spacing values per vertex.

    Returns
    -------
    hrel : (E,) array
        Relative edge lengths.
    """
    if not (isinstance(vert, np.ndarray) and
            isinstance(tria, np.ndarray) and
            isinstance(hvrt, np.ndarray)):
        raise TypeError("relhfn:incorrectInputClass")

    if vert.ndim != 2 or tria.ndim != 2:
        raise ValueError("relhfn:incorrectDimensions")
    if vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("relhfn:incorrectDimensions")
    if len(hvrt.shape) != 1 or hvrt.shape[0] != vert.shape[0]:
        raise ValueError("relhfn:incorrectDimensions")

    nnod = vert.shape[0]

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) >= nnod:
        raise ValueError("relhfn:invalidInputs")

    # --- compute unique edges
    eset = np.vstack([
        tria[:, [0, 1]],
        tria[:, [1, 2]],
        tria[:, [2, 0]]
    ])

    # sort each edge so (i,j) == (j,i)
    eset = np.sort(eset, axis=1)

    # unique edges
    eset = np.unique(eset, axis=0)

    # edge vectors
    evec = vert[eset[:, 1], :] - vert[eset[:, 0], :]

    # edge lengths
    elen = np.sqrt(np.sum(evec**2, axis=1))

    # average hvrt at edge midpoints
    hmid = hvrt[eset[:, 1]] + hvrt[eset[:, 0]]
    hmid = 0.5 * hmid

    # relative edge-length
    hrel = elen / hmid

    return hrel
