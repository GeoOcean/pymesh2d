import numpy as np
from ..aabb_tree.findtria import findtria

def trihfn(test, vert, tria, tree, hfun):
    """
    trihfn evaluate a discrete mesh-size function defined on 
    a 2-simplex triangulation embedded in R^2.

    Parameters
    ----------
    test : (Q,2) array
        Query points coordinates.
    vert : (V,2) array
        Vertices coordinates.
    tria : (T,3) array
        Triangular connectivity (0-based indices in Python).
    tree : dict or object
        Spatial indexing structure for {vert,tria}.
    hfun : (V,) array
        Mesh-size values at vertices.

    Returns
    -------
    hval : (Q,) array
        Interpolated values at query points.
    """
    # -------------------------- basic checks
    if not (isinstance(test, np.ndarray) and
            isinstance(vert, np.ndarray) and
            isinstance(tria, np.ndarray) and
            isinstance(hfun, np.ndarray)):
        raise TypeError("trihfn:incorrectInputClass")

    if test.ndim != 2 or vert.ndim != 2 or tria.ndim != 2 or hfun.ndim != 1:
        raise ValueError("trihfn:incorrectDimensions")

    if test.shape[1] != 2 or vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("trihfn:incorrectDimensions")

    if vert.shape[0] != hfun.shape[0]:
        raise ValueError("trihfn:incorrectDimensions")

    nvrt = vert.shape[0]
    if tria.min() < 0 or tria.max() >= nvrt:
        raise ValueError("trihfn:invalidInputs: invalid TRIA array")

    # ---------------------- test-to-tria queries
    tp, tj, _ = findtria(vert, tria, test, tree)

    if tp is None or len(tp) == 0:
        in_mask = np.zeros(test.shape[0], dtype=bool)
        ti = np.array([], dtype=int)
    else:
        in_mask = tp[:, 0] > 0
        ti = tj[tp[in_mask, 0]]

    # ---------------------- initialise output
    hval = np.full(test.shape[0], np.max(hfun))

    # ---------------------- linear interpolation
    if np.any(in_mask):
        d1 = test[in_mask, :] - vert[tria[ti, 0], :]
        d2 = test[in_mask, :] - vert[tria[ti, 1], :]
        d3 = test[in_mask, :] - vert[tria[ti, 2], :]

        a3 = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        a2 = np.abs(d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0])
        a1 = np.abs(d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0])

        hval[in_mask] = (
            a1 * hfun[tria[ti, 0]] +
            a2 * hfun[tria[ti, 1]] +
            a3 * hfun[tria[ti, 2]]
        ) / (a1 + a2 + a3)

    return hval