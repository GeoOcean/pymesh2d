import numpy as np
from ..mesh_cost.triarea import triarea
from ..poly_test.inpoly import inpoly
from .cfmtri import cfmtri


def deltri(vert=None, conn=None, node=None, PSLG=None, part=None, kind="constrained"):
    """
    Python translation of Darren Engwirda's deltri.m
    Constrained Delaunay triangulation (2D).
    """

    if vert is None: vert = np.empty((0, 2))
    if conn is None: conn = np.empty((0, 2), dtype=int)
    if node is None: node = np.empty((0, 2))
    if PSLG is None: PSLG = np.empty((0, 2), dtype=int)
    if part is None: part = []

    vert = np.asarray(vert, float)
    conn = np.asarray(conn, int)
    node = np.asarray(node, float)
    PSLG = np.asarray(PSLG, int)
    kind = kind.lower()

    # -------------------------------- basic checks
    nvrt = vert.shape[0]
    if conn.size and (conn.min() < 0 or conn.max() >= nvrt):
        raise ValueError("deltri:invalidInputs (invalid CONN indices)")

    if node.size:
        nnod = node.shape[0]
        if PSLG.size and (PSLG.min() < 0 or PSLG.max() >= nnod):
            raise ValueError("deltri:invalidInputs (invalid PSLG indices)")
        for p in part:
            if np.min(p) < 0 or np.max(p) >= PSLG.shape[0]:
                raise ValueError("deltri:invalidInputs (invalid PART indices)")

    # -------------------------------- compute constrained triangulation
    vert, conn, tria = cfmtri(vert, conn)

    # -------------------------------- compute "inside" status
    tnum = np.zeros(tria.shape[0], dtype=int)
    if node.size and PSLG.size and part:
        tmid = (vert[tria[:, 0], :] +
                vert[tria[:, 1], :] +
                vert[tria[:, 2], :]) / 3.0

        for ppos, pedges in enumerate(part, start=1):
            stat, _ = inpoly(tmid, node, PSLG[pedges, :])
            tnum[stat] = ppos

        # Keep only interior triangles
        mask = tnum > 0
        tria = tria[mask, :]
        tnum = tnum[mask]

    # -------------------------------- flip for correct orientation
    area = triarea(vert, tria)
    neg = area < 0.0
    if np.any(neg):
        tria[neg, :] = tria[neg][:, [0, 2, 1]]

    return vert, conn, tria, tnum
