import numpy as np
from ..aabb_tree.findtria import findtria   # same structure as used in trihfn
from ...mesh_utils.wavenumhunt import wavenumhunt        # your previous function
from ..hjac_util.limgrad import limgrad     # to limit gradients (optional)
from ..mesh_util.tricon import tricon     # used for edge-based gradient limiting

def disphfn(test, vert, tria, tree, hfun):
    """
    disphfn
    Generate a mesh-size function (hfun) based on local wave number 
    resolution for a constant wave period, using Hunt (1979) dispersion.

    Parameters
    ----------
    T : float
        Constant wave period [s].
    depth : (V,) array
        Local water depth at each vertex [m].
    vert : (V,2) array
        XY-coordinates of mesh vertices.
    tria : (T,3) array
        Triangular connectivity (0-based indices).
    N : int, optional
        Desired number of mesh nodes per wavelength (default = 20).
    dhdx : float, optional
        Maximum allowed gradient for mesh size (default = 0.25).

    Returns
    -------
    hfun : (V,) array
        Target mesh size field at vertices.
    L : (V,) array
        Local wavelength [m].
    k : (V,) array
        Local wave number [1/m].
    """
    N=20
    T=10.0
    depth = 1
    dhdx = 0.25
    # ------------------------------ basic checks
    if not isinstance(T, (int, float, np.number)):
        raise TypeError("disphfn: T must be a scalar.")
    if depth.ndim != 1 or depth.shape[0] != vert.shape[0]:
        raise ValueError("disphfn: 'depth' must be 1D with same length as vert.")
    if np.any(depth <= 0):
        raise ValueError("disphfn: depths must be strictly positive.")

    # ------------------------------ compute wavelength and wavenumber
    L, k = wavenumhunt(T, depth)

    # ------------------------------ define target mesh spacing
    # Example: N points per wavelength -> h = L / N
    hfun = L / float(N)

    # ------------------------------ limit gradient (optional)
    # Compute all unique mesh edges
    edge, _ = tricon(tria)
    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec ** 2, axis=1))

    # Apply gradient limiter (same concept as limhfn)
    hfun, _ = limgrad(edge, elen, hfun, dhdx, np.sqrt(vert.shape[0]))

    return hfun
