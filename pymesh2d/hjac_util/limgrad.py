import numpy as np

def limgrad(edge, elen, ffun, dfdx, imax):
    """
    LIMGRAD imposes "gradient-limits" on a function defined over an undirected graph.
    
    This is a faithful 0-based translation of Darren Engwirda's MATLAB version (18/04/2017).

    Parameters
    ----------
    edge : (NE, 2) ndarray of int
        Undirected graph edges (0-based indices).
    elen : (NE,) ndarray of float
        Edge lengths.
    ffun : (N,) ndarray of float
        Function values at graph nodes.
    dfdx : float
        Gradient limit.
    imax : int
        Maximum number of iterations.

    Returns
    -------
    ffun : (N,) ndarray of float
        Gradient-limited function.
    flag : bool
        True if convergence reached before imax iterations.
    """

    # ---------------------------- Basic input checks
    edge = np.asarray(edge, dtype=int)
    elen = np.asarray(elen, dtype=float).reshape(-1)
    ffun = np.asarray(ffun, dtype=float).reshape(-1)

    nnod = ffun.shape[0]

    if edge.ndim != 2 or edge.shape[1] < 2:
        raise ValueError("limgrad:incorrectDimensions - EDGE must be (NE,2)")
    if elen.ndim > 2 or ffun.ndim > 2:
        raise ValueError("limgrad:incorrectDimensions - ELEN/FFUN must be vectors")
    if elen.shape[0] != edge.shape[0]:
        raise ValueError("limgrad:incorrectDimensions - ELEN and EDGE must have same length")
    if dfdx < 0.0 or imax < 0:
        raise ValueError("limgrad:invalidInputArgument - DFDX or IMAX invalid")

    if edge[:, :2].min() < 0 or edge[:, :2].max() >= nnod:
        raise ValueError("limgrad:invalidInputArgument - invalid EDGE indices")

    # ---------------------------- Build adjacency references
    # nvec : concatenated node list for all edges
    # ivec : corresponding edge index for each node occurrence
    nvec = np.concatenate([edge[:, 0], edge[:, 1]])
    ivec = np.concatenate([
        np.arange(edge.shape[0], dtype=int),
        np.arange(edge.shape[0], dtype=int)
    ])

    sort_idx = np.argsort(nvec)
    nvec = nvec[sort_idx]
    ivec = ivec[sort_idx]

    # mark nodes that are connected
    mark = np.zeros(nnod, dtype=bool)
    mark[edge[:, 0]] = True
    mark[edge[:, 1]] = True

    # find boundaries in sorted nvec
    idxx = np.where(np.diff(nvec) > 0)[0]

    # nptr[i,0]: start index in ivec of edges adjacent to node i
    # nptr[i,1]: end index in ivec of edges adjacent to node i
    nptr = np.full((nnod, 2), -1, dtype=int)
    if idxx.size > 0:
        nptr[mark, 0] = np.concatenate(([0], idxx + 1))
        nptr[mark, 1] = np.concatenate((idxx, [len(nvec) - 1]))
    else:
        # single node connected
        nptr[mark, 0] = 0
        nptr[mark, 1] = len(nvec) - 1

    # ---------------------------- Initialize
    aset = np.zeros(nnod, dtype=int)  # active set iteration index
    ftol = np.min(ffun) * np.sqrt(np.finfo(float).eps)

    # ---------------------------- Iterative relaxation
    for iter in range(1, imax.astype(int) + 1):
        # Find nodes active in this pass
        aidx = np.where(aset == iter - 1)[0]
        if aidx.size == 0:
            break

        # Reorder by function value for better convergence
        aidx = aidx[np.argsort(ffun[aidx])]

        # Visit each active node and its neighboring edges
        for npos in aidx:
            for jpos in range(nptr[npos, 0], nptr[npos, 1] + 1):
                epos = ivec[jpos]
                nod1, nod2 = edge[epos, :2]

                # Limit about the min-value node
                if ffun[nod1] > ffun[nod2]:
                    fun1 = ffun[nod2] + elen[epos] * dfdx
                    if ffun[nod1] > fun1 + ftol:
                        ffun[nod1] = fun1
                        aset[nod1] = iter
                else:
                    fun2 = ffun[nod1] + elen[epos] * dfdx
                    if ffun[nod2] > fun2 + ftol:
                        ffun[nod2] = fun2
                        aset[nod2] = iter

    flag = (iter < imax)
    return ffun, flag


