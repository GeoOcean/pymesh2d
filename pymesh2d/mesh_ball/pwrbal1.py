import numpy as np

def pwrbal1(pp, pw, ee):
    """
    PWRBAL1 compute the ortho-balls associated with a 1-simplex
    triangulation embedded in R^2 or R^3.

    Parameters
    ----------
    pp : (N,2) or (N,3) array
        Point coordinates.
    pw : (N,1) array
        Vertex weights.
    ee : (E,2) array
        Edge connectivity.

    Returns
    -------
    ball : (E,3) or (E,4) array
        Ortho-balls [xc, yc, rc^2] (2D) or [xc, yc, zc, rc^2] (3D).
    """
    if not (isinstance(pp, np.ndarray) and
            isinstance(pw, np.ndarray) and
            isinstance(ee, np.ndarray)):
        raise TypeError("pwrbal1:incorrectInputClass")

    if pp.ndim != 2 or pw.ndim != 2 or ee.ndim != 2:
        raise ValueError("pwrbal1:incorrectDimensions")

    if pp.shape[0] != pw.shape[0] or ee.shape[1] < 2 or pp.shape[1] < 2:
        raise ValueError("pwrbal1:incorrectDimensions")

    dim = pp.shape[1]

    if dim == 2:
        # linear offset
        pp12 = pp[ee[:, 0], :] - pp[ee[:, 1], :]
        ww12 = pw[ee[:, 0], 0] - pw[ee[:, 1], 0]
        dp12 = np.sum(pp12 * pp12, axis=1)

        tpwr = 0.5 * (ww12 + dp12) / dp12

        ball = np.zeros((ee.shape[0], 3))
        ball[:, 0:2] = pp[ee[:, 0], :] - tpwr[:, None] * pp12

        vsq1 = pp[ee[:, 0], :] - ball[:, 0:2]
        vsq2 = pp[ee[:, 1], :] - ball[:, 0:2]

        # mean radii
        rsq1 = np.sum(vsq1**2, axis=1) - pw[ee[:, 0], 0]
        rsq2 = np.sum(vsq2**2, axis=1) - pw[ee[:, 1], 0]

        ball[:, 2] = (rsq1 + rsq2) / 2.0

    elif dim == 3:
        # linear offset
        pp12 = pp[ee[:, 0], :] - pp[ee[:, 1], :]
        ww12 = pw[ee[:, 0], 0] - pw[ee[:, 1], 0]
        dp12 = np.sum(pp12 * pp12, axis=1)

        tpwr = 0.5 * (ww12 + dp12) / dp12

        ball = np.zeros((ee.shape[0], 4))
        ball[:, 0:3] = pp[ee[:, 0], :] - tpwr[:, None] * pp12

        vsq1 = pp[ee[:, 0], :] - ball[:, 0:3]
        vsq2 = pp[ee[:, 1], :] - ball[:, 0:3]

        # mean radii
        rsq1 = np.sum(vsq1**2, axis=1) - pw[ee[:, 0], 0]
        rsq2 = np.sum(vsq2**2, axis=1) - pw[ee[:, 1], 0]

        ball[:, 3] = (rsq1 + rsq2) / 2.0

    else:
        raise ValueError("pwrbal1:unsupportedDimension")

    return ball
