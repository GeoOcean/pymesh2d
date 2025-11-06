import numpy as np
from .pwrbal1 import pwrbal1

def tribal1(pp, ee):
    """
    TRIBAL1 compute the circumballs associated with a 1-simplex
    triangulation embedded in R^2 or R^3.

    Parameters
    ----------
    pp : (N,2) or (N,3) array
        Node coordinates.
    ee : (E,2) array
        Edge connectivity.

    Returns
    -------
    bb : (E,3) or (E,4) array
        Circumscribing balls [xc,yc,rc^2] (2D) or [xc,yc,zc,rc^2] (3D).
    """
    return pwrbal1(pp, np.zeros((pp.shape[0], 1)), ee)
