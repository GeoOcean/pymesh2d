import numpy as np
from ..mesh_util.tricon import tricon
from ..hjac_util.limgrad import limgrad

def limhfn(vert, tria, hfun, dhdx):
    """
    limhfn impose gradient limits on a discrete mesh-size 
    function defined over a 2-simplex triangulation.

    Parameters
    ----------
    vert : (V,2) array
        XY coordinates of vertices.
    tria : (T,3) array
        Triangular connectivity (0-based indices in Python).
    hfun : (V,) array
        Mesh size values at each vertex.
    dhdx : float
        Gradient limit.

    Returns
    -------
    hfun : (V,) array
        Gradient-limited mesh size function.
    """
    # -------------------------- basic checks
    if not (isinstance(vert, np.ndarray) and
            isinstance(tria, np.ndarray) and
            isinstance(hfun, np.ndarray) and
            np.isscalar(dhdx)):
        raise TypeError("limhfn:incorrectInputClass")

    if (vert.ndim != 2 or
        tria.ndim != 2 or
        hfun.ndim != 1 or
        vert.shape[1] != 2 or
        tria.shape[1] < 3 or
        vert.shape[0] != hfun.shape[0]):
        raise ValueError("limhfn:incorrectDimensions")

    nvrt = vert.shape[0]

    if tria.min() < 0 or tria.max() >= nvrt:
        raise ValueError("limhfn:invalidInputArgument: invalid TRIA array")

    # -------------------- impose gradient limits over mesh edges
    edge, tria = tricon(tria)

    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec ** 2, axis=1))

    # -------------------- impose gradient limits over edge graph
    hfun, _ = limgrad(edge, elen, hfun, dhdx, np.sqrt(nvrt))

    return hfun
