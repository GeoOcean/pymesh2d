import numpy as np

def cdtbal1(pp, ee):
    """
    CDTBAL1 compute the circumballs associated with a 1-simplex
    triangulation embedded in R^2.

    Parameters
    ----------
    pp : (N,2) array
        Points coordinates.
    ee : (E,2) array
        Edges connectivity.

    Returns
    -------
    bb : (E,3) array
        Circumballs [xc, yc, rc^2].
    """
    # ----------------------- basic checks
    if not (isinstance(pp, np.ndarray) and isinstance(ee, np.ndarray)):
        raise TypeError("cdtbal1:incorrectInputClass")

    if pp.ndim != 2 or ee.ndim != 2:
        raise ValueError("cdtbal1:incorrectDimensions")

    if pp.shape[1] != 2 or ee.shape[1] < 2:
        raise ValueError("cdtbal1:incorrectDimensions")

    # ----------------------- compute circumballs
    bb = np.zeros((ee.shape[0], 3))

    bb[:, 0:2] = 0.5 * (pp[ee[:, 0], :] + pp[ee[:, 1], :])
    bb[:, 2] = 0.25 * np.sum((pp[ee[:, 0], :] - pp[ee[:, 1], :]) ** 2, axis=1)

    return bb
