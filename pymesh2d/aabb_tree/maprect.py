import numpy as np

from .scantree import scantree


def maprect(tr, pr):
    """
    MAPRECT: Map rectangles to AABB-tree nodes.

    Parameters:
        tr (dict): AABB-tree structure.
        pr (numpy.ndarray): Rectangles to map.

    Returns:
        tm (dict): Tree-to-item mapping.
        im (dict, optional): Item-to-tree mapping.

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
    if tr is None or pr is None:
        raise ValueError("Invalid input arguments.")

    if len(pr.shape) != 2:
        raise ValueError("Input 'pr' must be a 2D array.")

    if "ii" not in tr or "xx" not in tr or "ll" not in tr:
        raise ValueError("Invalid AABB-tree structure.")

    # Call scantree with the partrect function
    tm, im = scantree(tr, pr, partrect)
    return tm, im


def partrect(pr, b1, b2):
    """
    PARTRECT: Partition rectangles between boxes B1 and B2 for SCANTREE.

    Parameters:
        pr (numpy.ndarray): Rectangles to partition.
        b1 (numpy.ndarray): Bounds of the first box.
        b2 (numpy.ndarray): Bounds of the second box.

    Returns:
        j1 (numpy.ndarray): Boolean mask for rectangles in B1.
        j2 (numpy.ndarray): Boolean mask for rectangles in B2.

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
    j1 = np.ones(pr.shape[0], dtype=bool)
    j2 = np.ones(pr.shape[0], dtype=bool)

    nd = b1.shape[1] // 2

    for ax in range(nd):
        # Check if rectangles are inside bounds along axis AX for B1
        j1 = j1 & (pr[:, ax + nd] >= b1[ax]) & (pr[:, ax] <= b1[ax + nd])

        # Check if rectangles are inside bounds along axis AX for B2
        j2 = j2 & (pr[:, ax + nd] >= b2[ax]) & (pr[:, ax] <= b2[ax + nd])

    return j1, j2
