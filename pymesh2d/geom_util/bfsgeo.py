import numpy as np


def bfsgeo(node, PSLG, seed, deltri, findtria, bfstri, setset):
    """
    bfsgeo partition geometry about "seeds" via breadth-first search.

    Parameters
    ----------
    node : ndarray (N, 2)
        Polygon vertices.
    PSLG : ndarray (E, 2)
        Polygon edges, each row is a pair of indices into `node`.
    seed : ndarray (S, 2)
        Seed points.
    deltri : callable
        Function (node, PSLG) -> (node, PSLG, tria).
    findtria : callable
        Function (node, tria, seed) -> (sptr, stri).
    bfstri : callable
        Function (PSLG, tria, itri) -> mark.
    setset : callable
        Function (edge_subset, PSLG) -> (same, epos).

    Returns
    -------
    node : ndarray
        Possibly updated vertices.
    PSLG : ndarray
        Possibly updated edges.
    part : list of lists
        Each sublist contains edge indices into PSLG defining one partition.

    Notes
    -----
    MESH2D is designed to provide a simple and easy-to-understand 
    implementation of Delaunay-based mesh-generation techniques.
    Traduction of Matlab Mesh2D repository by Darren Engwirda
    """

    # ---------------- basic checks
    if not (
        isinstance(node, np.ndarray)
        and isinstance(PSLG, np.ndarray)
        and isinstance(seed, np.ndarray)
    ):
        raise TypeError("bfsgeo: incorrect input class")

    if node.ndim != 2 or PSLG.ndim != 2 or seed.ndim != 2:
        raise ValueError("bfsgeo: incorrect dimensions")

    if node.shape[1] != 2 or PSLG.shape[1] != 2 or seed.shape[1] != 2:
        raise ValueError("bfsgeo: incorrect dimensions")

    nnod = node.shape[0]
    nedg = PSLG.shape[0]

    # ---------------- basic checks on indices
    if PSLG.min() < 1 or PSLG.max() > nnod:
        raise ValueError("bfsgeo: invalid EDGE input array")

    # ---------------- assemble full CDT
    node, PSLG, tria = deltri(node, PSLG)

    # ---------------- find seeds in CDT
    sptr, stri = findtria(node, tria, seed)

    okay = sptr[:, 1] >= sptr[:, 0]
    itri = stri[sptr[okay, 0]]

    # ---------------- partitions
    part = []

    for ipos in range(itri.shape[0]):
        # --- BFS about current tria
        mark = bfstri(PSLG, tria, itri[ipos])

        # --- match tria/poly edges
        edge = np.vstack(
            [tria[mark][:, [0, 1]], tria[mark][:, [1, 2]], tria[mark][:, [2, 0]]]
        )

        edge = np.sort(edge, axis=1)
        PSLG_sorted = np.sort(PSLG, axis=1)

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

    return node, PSLG, part
