import numpy as np
from .triarea import triarea

def triscr(pp, tt):
    """
    triscr calc. area-length ratios for triangles in a 2-simplex
    triangulation in the 2D plane.

    Parameters
    ----------
    pp : (N,2) array
        Vertex coordinates.
    tt : (T,3) array
        Triangle connectivity (0-based indices).

    Returns
    -------
    tscr : (T,) array
        Area-length ratios for each triangle.
    """
    # constant
    scal = 4.0 * np.sqrt(3.0) / 3.0

    # --- triangle areas (calls triarea for checks and computation)
    area = triarea(pp, tt)

    # --- squared edge lengths
    lrms = (
        np.sum((pp[tt[:, 1], :] - pp[tt[:, 0], :])**2, axis=1) +
        np.sum((pp[tt[:, 2], :] - pp[tt[:, 1], :])**2, axis=1) +
        np.sum((pp[tt[:, 2], :] - pp[tt[:, 0], :])**2, axis=1)
    )

    # average squared length
    lrms = (lrms / 3.0) ** 1.0

    # area-length ratio
    tscr = scal * area / lrms

    return tscr
