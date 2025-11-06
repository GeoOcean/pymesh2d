import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def drawtree(tr, ax=None):
    """
    drawtree: draw an aabb-tree generated using maketree.
    Works for trees in R^2 and R^3.

    Parameters
    ----------
    tr : dict
        AABB-tree structure.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
        Default is None.

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Darren Engwirda, Locally-optimal Delaunay-refinement
    and optimisation-based mesh generation, Ph.D. Thesis,
    School of Mathematics and Statistics, The University
    of Sydney, September 2014.
    """

    # --- basic checks
    if not isinstance(tr, dict) or not all(k in tr for k in ("xx", "ii", "ll")):
        raise ValueError("Incorrect aabb-tree.")

    # --- find "leaf" nodes
    lf = np.array([len(ll) > 0 for ll in tr["ll"]])

    fc = (0.95, 0.95, 0.55)
    ec = (0.15, 0.15, 0.15)

    dim = tr["xx"].shape[1]

    if dim == 4:
        # ------------------------- tree in R^2
        np_leaf = np.count_nonzero(lf)

        # nodes
        pp = np.vstack(
            [
                np.column_stack((tr["xx"][lf, 0], tr["xx"][lf, 1])),
                np.column_stack((tr["xx"][lf, 2], tr["xx"][lf, 1])),
                np.column_stack((tr["xx"][lf, 2], tr["xx"][lf, 3])),
                np.column_stack((tr["xx"][lf, 0], tr["xx"][lf, 3])),
            ]
        )

        # faces
        bb = np.column_stack([np.arange(np_leaf) + np_leaf * i for i in range(4)])
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        for face in bb:
            polygon = plt.Polygon(pp[face], facecolor=fc, edgecolor=ec, alpha=0.2)
            ax.add_patch(polygon)

    elif dim == 6:
        # ------------------------- tree in R^3
        np_leaf = np.count_nonzero(lf)

        # nodes
        pp = np.vstack(
            [
                np.column_stack((tr["xx"][lf, 0], tr["xx"][lf, 1], tr["xx"][lf, 2])),
                np.column_stack((tr["xx"][lf, 3], tr["xx"][lf, 1], tr["xx"][lf, 2])),
                np.column_stack((tr["xx"][lf, 3], tr["xx"][lf, 4], tr["xx"][lf, 2])),
                np.column_stack((tr["xx"][lf, 0], tr["xx"][lf, 4], tr["xx"][lf, 2])),
                np.column_stack((tr["xx"][lf, 0], tr["xx"][lf, 1], tr["xx"][lf, 5])),
                np.column_stack((tr["xx"][lf, 3], tr["xx"][lf, 1], tr["xx"][lf, 5])),
                np.column_stack((tr["xx"][lf, 3], tr["xx"][lf, 4], tr["xx"][lf, 5])),
                np.column_stack((tr["xx"][lf, 0], tr["xx"][lf, 4], tr["xx"][lf, 5])),
            ]
        )

        # faces
        bb = np.vstack(
            [
                np.column_stack(
                    [
                        np.arange(np_leaf) + np_leaf * i,
                        np.arange(np_leaf) + np_leaf * j,
                        np.arange(np_leaf) + np_leaf * k,
                        np.arange(np_leaf) + np_leaf * l,
                    ]
                )
                for (i, j, k, l) in [
                    (0, 1, 2, 3),
                    (4, 5, 6, 7),
                    (0, 3, 7, 4),
                    (3, 2, 6, 7),
                    (2, 1, 5, 6),
                    (1, 0, 4, 5),
                ]
            ]
        )
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # build faces
        faces = [[pp[idx] for idx in face] for face in bb]
        poly3d = Poly3DCollection(faces, facecolors=[fc], edgecolors=[ec], alpha=0.2)
        ax.add_collection3d(poly3d)

        ax.set_box_aspect([1, 1, 1])

    else:
        raise ValueError("Unsupported tree dimensionality.")
