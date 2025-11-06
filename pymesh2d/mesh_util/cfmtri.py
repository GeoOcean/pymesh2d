import numpy as np
from .setset import setset
from scipy.spatial import Delaunay

def cfmtri(vert, econ):
    """
    cfmtri compute a conforming 2-simplex Delaunay triangulation
    in the two-dimensional plane.

    Parameters
    ----------
    vert : (V,2) array
        XY coordinates of vertices to be triangulated.
    econ : (C,2) array
        Constraining edges, indices into vert.

    Returns
    -------
    vert : (V,2) array
        Updated vertices (rescaled back to original space).
    econ : (E,2) array
        Updated constraining edges.
    tria : (T,3) array
        Triangulation connectivity.
    """

    if not isinstance(vert, np.ndarray) or not isinstance(econ, np.ndarray):
        raise TypeError("cfmtri:incorrectInputClass")

    if vert.ndim != 2 or econ.ndim != 2:
        raise ValueError("cfmtri:incorrectDimensions")

    if vert.shape[1] != 2 or econ.shape[1] != 2:
        raise ValueError("cfmtri:incorrectDimensions")

    # --- rescale geometry to [-1,1]
    vmax = np.max(vert, axis=0)
    vmin = np.min(vert, axis=0)

    vdel = np.mean(vmax - vmin) * 0.5
    vmid = (vmax + vmin) * 0.5

    vert = (vert - vmid) / vdel

    # --- iterative recovery of constrained edges
    while True:
        tria = delaunay2(vert)

        nv = vert.shape[0]
        nt = tria.shape[0]

        # build edge set
        ee = np.zeros((nt * 3, 2), dtype=int)
        ee[0:nt, :] = tria[:, [0, 1]]
        ee[nt:2*nt, :] = tria[:, [1, 2]]
        ee[2*nt:3*nt, :] = tria[:, [2, 0]]

        # find constraints within edge set
        in_mask, _ = setset(econ, ee)

        if np.all(in_mask):
            break

        # unrecovered edge centres
        vm = (vert[econ[~in_mask, 0], :] +
              vert[econ[~in_mask, 1], :]) * 0.5

        # unrecovered edge indexes
        ev = np.arange(nv, nv + vm.shape[0])
        en = np.vstack([
            np.column_stack([econ[~in_mask, 0], ev]),
            np.column_stack([econ[~in_mask, 1], ev])
        ])

        # update arrays
        vert = np.vstack([vert, vm])
        econ = np.vstack([econ[in_mask, :], en])

    # --- undo scaling
    vert = vert * vdel + vmid

    return vert, econ, tria

def delaunay2(points, options=None):
    n, d = points.shape
    
    if options is None:
        if d >= 4:
            options = "Qt Qbb Qc Qx"
        else:
            options = "Qt Qbb Qc"

    tri = Delaunay(points, qhull_options=options)

    t = tri.simplices
    
    return t
