import numpy as np
from .pwrbal2 import pwrbal2

def tribal2(pp, tt):
    """
    TRIBAL2 compute the circumballs associated with a 2-simplex
    triangulation embedded in R^2 or R^3.

    Parameters
    ----------
    pp : (N,2) or (N,3) array
        Node coordinates.
    tt : (T,3) array
        Triangle connectivity.

    Returns
    -------
    bb : (T,3) or (T,4) array
        Circumscribing balls [xc,yc,rc^2] (2D) or [xc,yc,zc,rc^2] (3D).
    """
    return pwrbal2(pp, np.zeros((pp.shape[0], 1)), tt)
