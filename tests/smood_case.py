"""
Shared test scenario for the `smood` orthogonalization/merge pipeline.

`smood` has no coverage in the pre-existing `tridemo`-based test suite (it
post-processes a triangulation for Delft3D-FM/UGRID export and is not part
of the `tridemo` demo set). This module builds a small, fast, deterministic
lon/lat triangulation so that `smood`'s numerical output can be pinned down
as a regression reference before refactoring `pymesh2d.ortho_merge`.
"""
import numpy as np

from pymesh2d.refine import refine
from pymesh2d.smooth import smooth


def build_smood_input():
    """
    Build a small triangulated mesh over a lon/lat square domain, suitable
    as input to `smood` (which reprojects to a local UTM CRS internally).

    Returns
    -------
    vert, conn, tria, tnum : as returned by `refine`/`smooth`.
    """
    node = np.array(
        [
            [-4.0, 43.4],
            [-3.9, 43.4],
            [-3.9, 43.5],
            [-4.0, 43.5],
        ]
    )
    edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    opts = {"disp": np.inf}
    hfun = 0.01
    vert, etri, tria, tnum = refine(node, edge, [], opts, hfun)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum, {"disp": np.inf})
    return vert, etri, tria, tnum


SMOOD_OPTS = {"disp": np.inf, "iter": 2}
