import numpy as np

from pymesh2d.aabb_tree.findtria import findtria
from pymesh2d.mesh_util.bfstri import bfstri
from pymesh2d.mesh_util.deltri import deltri
from pymesh2d.mesh_util.setset import setset


def bfsgeo(node, edge, seed):
    """
    Partition polygonal geometry into regions using a breadth-first search.

    This function divides a 2D polygonal geometry into manifold partitions
    by expanding regions around a set of user-defined seed points.
    The expansion proceeds in a breadth-first manner until polygon boundaries
    are encountered, effectively grouping edges into connected geometric parts.

    Parameters
    ----------
    NODE : ndarray of shape (N, 2)
        XY-coordinates of the polygon vertices.
    EDGE : ndarray of shape (E, 2)
        Array of polygon edge indices. Each row defines one edge, such that:
        `NODE[EDGE[j, 0], :]` and `NODE[EDGE[j, 1], :]` are the endpoints
        of the j-th edge.
    SEED : ndarray of shape (S, 2)
        XY-coordinates of the seed points used to initiate the expansion.

    Returns
    -------
    NODE : ndarray of shape (N, 2)
        The vertex coordinates (returned unchanged).
    EDGE : ndarray of shape (E, 2)
        The polygon edge connectivity (returned unchanged).
    PART : list of lists or list of ndarrays
        A list of geometry partitions, one per seed.
        Each `PART[k]` contains the indices of edges (referring to `EDGE`)
        that define the k-th partition.

    Notes
    -----
    - The breadth-first expansion continues until the traversal reaches
      the outer or inner boundaries of the polygon.
    - Useful for decomposing complex, non-manifold geometries into
      well-defined manifold parts suitable for triangulation.
    - Often used as a preprocessing step before applying mesh generation
      or refinement algorithms such as `refine`.

    References
    ----------
    Translation of the MESH2D function `BFSGEO2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    See also
    --------
    refine : Generate constrained Delaunay triangulations.
    fixgeo : Clean or validate polygonal geometries.
    bfstri : Partition triangulations via breadth-first traversal.
    """

    # ---------------- basic checks
    if not (
        isinstance(node, np.ndarray)
        and isinstance(edge, np.ndarray)
        and isinstance(seed, np.ndarray)
    ):
        raise TypeError("bfsgeo: incorrect input class")

    if node.ndim != 2 or edge.ndim != 2 or seed.ndim != 2:
        raise ValueError("bfsgeo: incorrect dimensions")

    if node.shape[1] != 2 or edge.shape[1] != 2 or seed.shape[1] != 2:
        raise ValueError("bfsgeo: incorrect dimensions")

    nnod = node.shape[0]
    _nedg = edge.shape[0]

    # ---------------- basic checks on indices
    if edge.min() < 0 or edge.max() > nnod:
        raise ValueError("bfsgeo: invalid EDGE input array")

    # ---------------- assemble full CDT
    node, edge, tria = deltri(node, edge)

    # ---------------- find seeds in CDT
    sptr, stri = findtria(node, tria, seed)

    okay = sptr[:, 1] >= sptr[:, 0]
    itri = stri[sptr[okay, 0]]

    # ---------------- PART for all seed
    part = []

    for ipos in range(itri.shape[0]):
        # --- BFS about current tria
        mark = bfstri(edge, tria, itri[ipos])

        # --- match tria/poly edges
        edge = np.vstack(
            [tria[mark][:, [0, 1]], tria[mark][:, [1, 2]], tria[mark][:, [2, 0]]]
        )

        edge = np.sort(edge, axis=1)
        PSLG_sorted = np.sort(edge, axis=1)

        same, epos = setset(edge[:, :2], PSLG_sorted)

        # --- find match multiplicity
        epos = epos[epos > 0]
        epos = np.sort(epos)

        if epos.size == 0:
            part.append([])
            continue

        eidx = np.where(np.diff(epos) != 0)[0]
        eptr = np.hstack(([0], eidx + 1, [epos.size]))
        enum = eptr[1:] - eptr[:-1]

        # --- select singly-matched edges
        part.append(epos[eptr[:-1][enum == 1]])

    return node, edge, part
