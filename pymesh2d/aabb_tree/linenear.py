import numpy as np


def linenear(pa, pb, pc, pd):
    """
    LINENEAR - Compute the nearest points on line segments in d-dimensional space.

    Line parameters are bounded on [-1, +1].

    Parameters
    ----------
    pa, pb : np.ndarray
        (N, D) arrays defining the endpoints of the first set of segments.
    pc, pd : np.ndarray
        (N, D) arrays defining the endpoints of the second set of segments.

    Returns
    -------
    ok : np.ndarray (bool)
        True where lines are not degenerate or parallel.
    tp : np.ndarray
        Parametric coordinates along [pa, pb] in [-1, +1].
    tq : np.ndarray
        Parametric coordinates along [pc, pd] in [-1, +1].

    References
    ----------
    Translation of the MESH2D function `LINENEAR`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # Midpoints and half-length direction vectors
    m1 = 0.5 * (pa + pb)
    D1 = 0.5 * (pb - pa)

    m2 = 0.5 * (pc + pd)
    D2 = 0.5 * (pd - pc)

    # Projection terms (scalar products)
    r1 = np.sum(m2 * D1, axis=1) - np.sum(m1 * D1, axis=1)
    r2 = np.sum(m1 * D2, axis=1) - np.sum(m2 * D2, axis=1)

    # Coefficients of the linear system
    A1 = np.sum(D1 * D1, axis=1)
    A2 = -np.sum(D1 * D2, axis=1)
    A3 = -np.sum(D1 * D2, axis=1)
    A4 = np.sum(D2 * D2, axis=1)

    # Determinant
    dd = A1 * A4 - A2 * A3

    # Numerators
    tp = A4 * r1 - A2 * r2
    tq = -A3 * r1 + A1 * r2

    # Robust tolerance
    rt = np.max(np.abs(np.stack([A1, A2, A3, A4], axis=1)), axis=1)
    rt *= np.finfo(float).eps ** 0.8

    # Lines are valid if not parallel
    ok = np.abs(dd) > rt

    # Initialize results
    tp_out = np.zeros_like(tp)
    tq_out = np.zeros_like(tq)

    # Compute valid intersections
    tp_out[ok] = tp[ok] / dd[ok]
    tq_out[ok] = tq[ok] / dd[ok]

    return ok, tp_out, tq_out
