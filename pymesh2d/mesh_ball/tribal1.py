import numpy as np

from .pwrbal1 import pwrbal1


def tribal1(pp, ee):
    """
    Compute the circumballs associated with a 1-simplex triangulation
    embedded in R² or R³.

    This function calculates the circumscribing balls for the set of
    1-simplexes (edges) in a triangulation.

    Parameters
    ----------
    PP : ndarray of shape (N, D)
        Coordinates of the vertices in the triangulation, where `D` = 2 or 3.
    EE : ndarray of shape (E, 2)
        Array of edge indices defining the 1-simplexes.

    Returns
    -------
    BB : ndarray of shape (E, 3)
        Circumballs associated with each edge, where each row is `[XC, YC, RC²]`
        — the center coordinates and squared radius of the circumball.

    Notes
    -----
    - Each circumball is the smallest sphere (or circle in 2D) passing through
      the vertices of the edge segment.
    - Useful for Delaunay-based refinement and geometric quality evaluation.

    References
    ----------
    Translation of the MESH2D function `TRIBAL1`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    return pwrbal1(pp, np.zeros((pp.shape[0], 1)), ee)
