import numpy as np
from ..aabb_tree.maketree import maketree

def idxtri(vert, tria):
    """
    idxtri create a spatial-indexing structure for a 2-simplex
    triangulation embedded in the two-dimensional plane.

    Parameters
    ----------
    vert : (V,2) ndarray
        Array of XY coordinates.
    tria : (T,3) ndarray
        Array of triangles (indices 0-based).

    Returns
    -------
    tree : dict
        AABB-tree structure (here returned as a dict with 'bmin','bmax','opts').
    """

    if not (isinstance(vert, np.ndarray) and isinstance(tria, np.ndarray)):
        raise TypeError("idxtri:incorrectInputClass")

    if vert.ndim != 2 or tria.ndim != 2:
        raise ValueError("idxtri:incorrectDimensions")
    if vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("idxtri:incorrectDimensions")

    nvrt = vert.shape[0]

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) >= nvrt:
        raise ValueError("idxtri:invalidInputs")

    # ------------------------------ calc. AABB indexing for TRIA
    bmin = vert[tria[:, 0], :].copy()
    bmax = vert[tria[:, 0], :].copy()

    for ii in range(tria.shape[1]):
        bmin = np.minimum(bmin, vert[tria[:, ii], :])
        bmax = np.maximum(bmax, vert[tria[:, ii], :])

    # ------------------------------ opts (MATLAB/Octave specific, we mimic)
    opts = {}
    opts["nobj"] = 16  # default for Python, like MATLAB

    # ------------------------------ build tree (dummy version)
    tree = maketree(np.hstack([bmin, bmax]), opts)

    return tree
