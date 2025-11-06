import numpy as np

def inv_3x3(AA):
    """
    INV_3X3 calc. the inverses for a block of 3-by-3 matrices.
    Each inverse is actually DET(A) * A^(-1), to improve robustness.

    Parameters
    ----------
    AA : (3,3,N) array
        Block of 3x3 matrices.

    Returns
    -------
    II : (3,3,N) array
        Incomplete inverses (det * inv(A)).
    DA : (N,) array
        Determinants of each 3x3 matrix.
    """
    if not isinstance(AA, np.ndarray):
        raise TypeError("inv_3x3:incorrectInputClass")

    if AA.ndim > 3:
        raise ValueError("inv_3x3:incorrectDimensions")

    if AA.shape[0] != 3 or AA.shape[1] != 3:
        raise ValueError("inv_3x3:incorrectDimensions")

    # build inv(A)
    II = np.zeros_like(AA)
    DA = det_3x3(AA)

    II[0, 0, :] = AA[2, 2, :] * AA[1, 1, :] - AA[2, 1, :] * AA[1, 2, :]
    II[0, 1, :] = AA[2, 1, :] * AA[0, 2, :] - AA[2, 2, :] * AA[0, 1, :]
    II[0, 2, :] = AA[1, 2, :] * AA[0, 1, :] - AA[1, 1, :] * AA[0, 2, :]

    II[1, 0, :] = AA[2, 0, :] * AA[1, 2, :] - AA[2, 2, :] * AA[1, 0, :]
    II[1, 1, :] = AA[2, 2, :] * AA[0, 0, :] - AA[2, 0, :] * AA[0, 2, :]
    II[1, 2, :] = AA[1, 0, :] * AA[0, 2, :] - AA[1, 2, :] * AA[0, 0, :]

    II[2, 0, :] = AA[2, 1, :] * AA[1, 0, :] - AA[2, 0, :] * AA[1, 1, :]
    II[2, 1, :] = AA[2, 0, :] * AA[0, 1, :] - AA[2, 1, :] * AA[0, 0, :]
    II[2, 2, :] = AA[1, 1, :] * AA[0, 0, :] - AA[1, 0, :] * AA[0, 1, :]

    return II, DA


def det_3x3(AA):
    """
    Determinant for block of 3x3 matrices.

    Parameters
    ----------
    AA : (3,3,N) array

    Returns
    -------
    DA : (N,) array
    """
    return (
        AA[0, 0, :] * (AA[1, 1, :] * AA[2, 2, :] - AA[1, 2, :] * AA[2, 1, :])
        - AA[0, 1, :] * (AA[1, 0, :] * AA[2, 2, :] - AA[1, 2, :] * AA[2, 0, :])
        + AA[0, 2, :] * (AA[1, 0, :] * AA[2, 1, :] - AA[1, 1, :] * AA[2, 0, :])
    )
