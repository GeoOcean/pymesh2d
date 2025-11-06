import numpy as np
from .tribal2 import tribal2

def cdtbal2(pp, ee, tt):
    """
    CDTBAL2 compute the modified circumballs associated with a
    constrained 2-simplex Delaunay triangulation in R^2.

    Parameters
    ----------
    pp : (N,2) array
        Point coordinates.
    ee : (E,>=5) array
        Edge array with extra info.
    tt : (T,>=6) array
        Triangle array with extra info.

    Returns
    -------
    cc : (T,3) array
        Circumballs [xc, yc, rc^2].
    """
    # ------------------- basic checks
    if not (isinstance(pp, np.ndarray) and 
            isinstance(ee, np.ndarray) and 
            isinstance(tt, np.ndarray)):
        raise TypeError("cdtbal2:incorrectInputClass")

    if pp.ndim != 2 or ee.ndim != 2 or tt.ndim != 2:
        raise ValueError("cdtbal2:incorrectDimensions")

    if pp.shape[1] != 2 or ee.shape[1] < 5 or tt.shape[1] < 6:
        raise ValueError("cdtbal2:incorrectDimensions")

    # ------------------- calc circumballs
    cc = tribal2(pp, tt)

    # ------------------- replace with face-balls if smaller
    cc = minfac2(cc, pp, ee, tt, 0, 1, 2)
    cc = minfac2(cc, pp, ee, tt, 1, 2, 0)
    cc = minfac2(cc, pp, ee, tt, 2, 0, 1)

    return cc


def minfac2(cc, pp, ee, tt, ni, nj, nk):
    """
    MINFAC2 modify the set of circumballs to constrain centres
    to the boundaries of the CDT.

    Parameters
    ----------
    cc : (T,3) array
        Circumballs [xc, yc, rc^2].
    pp : (N,2) array
        Points.
    ee : (E,>=5) array
        Edge array.
    tt : (T,>=6) array
        Triangle array.
    ni, nj, nk : int
        Local indices of triangle vertices.

    Returns
    -------
    cc : (T,3) array
        Updated circumballs.
    """
    # outer edge flag
    EF = ee[tt[:, ni+3], 4] > 0

    # edge balls centres
    bc = 0.5 * (pp[tt[EF, ni], :] + pp[tt[EF, nj], :])

    # edge radii
    br = np.sum((bc - pp[tt[EF, ni], :])**2, axis=1) \
       + np.sum((bc - pp[tt[EF, nj], :])**2, axis=1)
    br = br * 0.5

    # enclosing radii
    ll = np.sum((bc - pp[tt[EF, nk], :])**2, axis=1)

    # replace if min
    bi = (br >= ll) & (br <= cc[EF, 2])
    ei = np.where(EF)[0]
    ti = ei[bi]

    cc[ti, 0:2] = bc[bi, :]
    cc[ti, 2] = br[bi]

    return cc
