import numpy as np
from ..refine import refine
from .limhfn import limhfn

def lfshfn(node=None, PSLG=None, part=None, opts=None):
    """
    lfshfn calc. a discrete 'local-feature-size' estimate 
    for a polygonal domain embedded in R^2.

    Parameters
    ----------
    node : (N,2) array
        Polygonal vertices.
    PSLG : (E,2) array
        Edge indexing (optional).
    part : list of arrays
        Partition of edges for multiply-connected geometry.
    opts : dict
        Options.

    Returns
    -------
    vert : (V,2) array
        XY coordinates of vertices in triangulation.
    tria : (T,3) array
        Triangular connectivity.
    hlfs : (V,) array
        Local feature size per vertex.
    """
    if node is None:
        node = np.empty((0, 2))
    if PSLG is None:
        PSLG = np.empty((0, 2), dtype=int)
    if part is None:
        part = []
    if opts is None:
        opts = {}

    # ------------------------------ build coarse background grid
    opts = makeopt(opts)

    # Placeholder: refine must be defined
    # Expected return: vert, conn, tria, tnum
    vert, conn, tria, tnum = refine(node, PSLG, part, opts)
    
    # ------------------------------ estimate local-feature-size
    hlfs = np.full(vert.shape[0], np.inf)

    # Edge lengths
    evec = vert[conn[:, 1], :] - vert[conn[:, 0], :]
    elen = np.sqrt(np.sum(evec ** 2, axis=1))
    hlen = elen.copy()

    for epos in range(conn.shape[0]):
        ivrt = conn[epos, 0]
        jvrt = conn[epos, 1]

        hlfs[ivrt] = min(hlfs[ivrt], hlen[epos])
        hlfs[jvrt] = min(hlfs[jvrt], hlen[epos])

    # ------------------------------ push gradient limits on HFUN
    DHDX = opts['dhdx']

    hlfs = limhfn(vert, tria, hlfs, DHDX)

    return vert, tria, hlfs


def makeopt(opts):
    """
    Setup the options dictionary for lfshfn.
    """
    # clone to avoid side-effects
    opts = dict(opts)

    if 'kind' not in opts:
        opts['kind'] = 'delaunay'
    else:
        if opts['kind'].lower() not in ('delfront', 'delaunay'):
            raise ValueError("lfshfn:invalidOption: Invalid refinement KIND.")

    if 'rho2' not in opts:
        opts['rho2'] = np.sqrt(2.0)
    else:
        if not np.isscalar(opts['rho2']):
            raise ValueError("lfshfn:incorrectDimensions")
        if opts['rho2'] < 1.0:
            raise ValueError("lfshfn:invalidOptionValues: rho2 must be >= 1.")

    if 'dhdx' not in opts:
        opts['dhdx'] = 0.25
    else:
        if not np.isscalar(opts['dhdx']):
            raise ValueError("lfshfn:incorrectDimensions")
        if opts['dhdx'] <= 0.0:
            raise ValueError("lfshfn:invalidOptionValues: dhdx must be > 0.")

    return opts