import numpy as np
import matplotlib.pyplot as plt

def getiso(xpos, ypos, zdat, ilev, filt=0.0):
    """
    getiso extract an iso-contour from a structured 2D dataset.

    Parameters
    ----------
    xpos, ypos : ndarray (N, M)
        Grid coordinates (must be same shape as zdat).
    zdat : ndarray (N, M)
        Scalar field values.
    ilev : float
        Isocontour level.
    filt : float, optional
        Minimum length scale filter (default 0).

    Returns
    -------
    node : ndarray (K, 2)
        Coordinates of contour vertices.
    edge : ndarray (E, 2)
        PSLG edges between contour vertices.
    """

    # --- basic checks
    if not (isinstance(xpos, np.ndarray) and
            isinstance(ypos, np.ndarray) and
            isinstance(zdat, np.ndarray) and
            np.isscalar(ilev) and
            np.isscalar(filt)):
        raise TypeError("getiso: incorrect input class")

    if xpos.shape != ypos.shape or xpos.shape != zdat.shape:
        raise ValueError("getiso: incorrect dimensions")

    # --- compute the contour with matplotlib
    fig, ax = plt.subplots()
    cs = ax.contour(xpos, ypos, zdat, levels=[ilev])
    plt.close(fig)  # close the figure, we just need data

    node = []
    edge = []

    for collection in cs.collections:
        for path in collection.get_paths():
            ppts = path.vertices
            numc = ppts.shape[0]

            pmin = ppts.min(axis=0)
            pmax = ppts.max(axis=0)
            pdel = pmax - pmin

            if np.min(pdel) >= filt:
                if np.allclose(ppts[0], ppts[-1]):
                    # closed loop
                    enew = np.vstack([
                        np.column_stack([np.arange(0, numc - 1),
                                         np.arange(1, numc)]),
                        [numc - 1, 0]
                    ])
                else:
                    # open contour
                    enew = np.column_stack([np.arange(0, numc - 1),
                                            np.arange(1, numc)])

                offset = len(node)
                enew = enew + offset

                node.extend(ppts.tolist())
                edge.extend(enew.tolist())

    node = np.array(node)
    edge = np.array(edge, dtype=int)

    return node, edge
