import numpy as np

def trideg(pp, tt):
    """
    trideg calc. topological degree for vertices in a 2-simplex triangulation.

    Parameters
    ----------
    pp : (N,D) array
        Vertex coordinates (2D or higher).
    tt : (T,3) array
        Triangle connectivity (0-based indices).

    Returns
    -------
    vdeg : (N,) array
        Vertex degrees (number of triangles incident to each vertex).
    """
    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("trideg:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("trideg:incorrectDimensions")
    if pp.shape[1] < 2 or tt.shape[1] < 3:
        raise ValueError("trideg:incorrectDimensions")

    nvrt = pp.shape[0]
    ntri = tt.shape[0]

    if np.min(tt[:, :3]) < 0 or np.max(tt[:, :3]) >= nvrt:
        raise ValueError("trideg:invalidInputs")

    # --- compute vertex degrees
    vdeg = np.zeros(nvrt, dtype=int)

    for tri in tt[:, :3]:
        vdeg[tri] += 1

    return vdeg
