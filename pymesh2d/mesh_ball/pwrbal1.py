import numpy as np


def pwrbal1(pp, pw, ee):
    """
    Compute the orthoballs (power balls) associated with a 1-simplex triangulation
    embedded in R² or R³.

    This function calculates the *power balls* for the set of 1-simplexes (edges)
    in a triangulation. Each power ball represents a weighted equivalent of a
    circumball, accounting for vertex weights.

    Parameters
    ----------
    PP : ndarray of shape (N, D)
        Coordinates of the vertices in the triangulation, where `D` = 2 or 3.
    PW : ndarray of shape (N,)
        Vector of vertex weights.
    TT : ndarray of shape (E, 2)
        Array of edge indices defining the 1-simplexes.

    Returns
    -------
    BB : ndarray of shape (E, 3)
        Power balls associated with each edge, where each row is `[XC, YC, RC²]`
        — the center coordinates and squared radius of the orthoball.

    Notes
    -----
    - The orthoball is the generalization of the circumball to weighted Delaunay
      triangulations (power diagrams).
    - Useful for constructing and analyzing weighted triangulations in 2D or 3D.

    References
    ----------
    Translation of the MESH2D function `PWRBAL1`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # --------------------------------------------- basic checks
    if not (
        isinstance(pp, np.ndarray)
        and isinstance(pw, np.ndarray)
        and isinstance(ee, np.ndarray)
    ):
        raise TypeError("pwrbal1:incorrectInputClass")

    if pp.ndim != 2 or pw.ndim != 2 or ee.ndim != 2:
        raise ValueError("pwrbal1:incorrectDimensions")

    if pp.shape[0] != pw.shape[0] or ee.shape[1] < 2 or pp.shape[1] < 2:
        raise ValueError("pwrbal1:incorrectDimensions")

    dim = pp.shape[1]

    if dim == 2:
        # ------------------------------------------- lin offset
        pp12 = pp[ee[:, 0], :] - pp[ee[:, 1], :]
        ww12 = pw[ee[:, 0], 0] - pw[ee[:, 1], 0]
        dp12 = np.sum(pp12 * pp12, axis=1)

        tpwr = 0.5 * (ww12 + dp12) / dp12

        ball = np.zeros((ee.shape[0], 3))
        ball[:, 0:2] = pp[ee[:, 0], :] - tpwr[:, None] * pp12

        vsq1 = pp[ee[:, 0], :] - ball[:, 0:2]
        vsq2 = pp[ee[:, 1], :] - ball[:, 0:2]

        # -------------------------------------------- mean radii
        rsq1 = np.sum(vsq1**2, axis=1) - pw[ee[:, 0], 0]
        rsq2 = np.sum(vsq2**2, axis=1) - pw[ee[:, 1], 0]

        ball[:, 2] = (rsq1 + rsq2) / 2.0

    elif dim == 3:
        # -------------------------------------------- lin offset
        pp12 = pp[ee[:, 0], :] - pp[ee[:, 1], :]
        ww12 = pw[ee[:, 0], 0] - pw[ee[:, 1], 0]
        dp12 = np.sum(pp12 * pp12, axis=1)

        tpwr = 0.5 * (ww12 + dp12) / dp12

        ball = np.zeros((ee.shape[0], 4))
        ball[:, 0:3] = pp[ee[:, 0], :] - tpwr[:, None] * pp12

        vsq1 = pp[ee[:, 0], :] - ball[:, 0:3]
        vsq2 = pp[ee[:, 1], :] - ball[:, 0:3]

        # -------------------------------------------- mean radii
        rsq1 = np.sum(vsq1**2, axis=1) - pw[ee[:, 0], 0]
        rsq2 = np.sum(vsq2**2, axis=1) - pw[ee[:, 1], 0]

        ball[:, 3] = (rsq1 + rsq2) / 2.0

    else:
        raise ValueError("pwrbal1:unsupportedDimension")

    return ball
