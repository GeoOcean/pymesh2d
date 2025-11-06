import numpy as np

def inv_2x2(AA):
    """
    INV_2X2 calc. the inverses for a block of 2-by-2 matrices.
    Each inverse is actually DET(A) * A^(-1), to improve robustness.

    Parameters
    ----------
    AA : (2,2,N) array
        Block of 2x2 matrices.

    Returns
    -------
    II : (2,2,N) array
        Incomplete inverses (det * inv(A)).
    DA : (N,) array
        Determinants of each 2x2 matrix.
    """
    if not isinstance(AA, np.ndarray):
        raise TypeError("inv_2x2:incorrectInputClass")

    if AA.ndim > 3:
        raise ValueError("inv_2x2:incorrectDimensions")

    if AA.shape[0] != 2 or AA.shape[1] != 2:
        raise ValueError("inv_2x2:incorrectDimensions")

    # build inv(A)
    II = np.zeros_like(AA)
    DA = det_2x2(AA)

    II[0, 0, :] = AA[1, 1, :]
    II[1, 1, :] = AA[0, 0, :]
    II[0, 1, :] = -AA[0, 1, :]
    II[1, 0, :] = -AA[1, 0, :]

    return II, DA


def det_2x2(AA):
    """
    Determinant for block of 2x2 matrices.

    Parameters
    ----------
    AA : (2,2,N) array

    Returns
    -------
    DA : (N,) array
    """
    return AA[0, 0, :] * AA[1, 1, :] - AA[0, 1, :] * AA[1, 0, :]
