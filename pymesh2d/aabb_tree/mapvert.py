import numpy as np

from .scantree import scantree


def mapvert(tr, pi):
    """
    MAPVERT find the tree-to-vertex mappings.

    Parameters
    ----------
    tr : dict
        AABB tree (from maketree).
    pi : ndarray of shape (N, ND)
        Query vertices.

    Returns
    -------
    tm : dict
        Tree-to-item mapping:
        - 'ii': array of tree indices
        - 'll': list of item lists
    im : dict
        Item-to-tree mapping:
        - 'ii': array of item indices
        - 'll': list of node lists

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

    # MATLAB: if (nargout == +1) ...
    # En Python, on renvoie toujours les deux mais im peut être None
    results = scantree(tr, pi, partvert)
    if len(results) == 1:
        return results[0], None
    else:
        return results


def partvert(pi, b1, b2):
    """
    PARTVERT partition points between boxes B1,B2 for SCANTREE.

    Parameters
    ----------
    pi : ndarray of shape (N, ND)
        Query points.
    b1, b2 : ndarray of shape (2*ND,)
        Bounding boxes.

    Returns
    -------
    j1, j2 : ndarray of bool of shape (N,)
        Mask of points inside b1 and b2 respectively.

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
    nd = b1.shape[0] // 2

    j1 = np.ones(pi.shape[0], dtype=bool)
    j2 = np.ones(pi.shape[0], dtype=bool)

    for ax in range(nd):
        j1 &= (pi[:, ax] >= b1[ax]) & (pi[:, ax] <= b1[ax + nd])
        j2 &= (pi[:, ax] >= b2[ax]) & (pi[:, ax] <= b2[ax + nd])

    return j1, j2
