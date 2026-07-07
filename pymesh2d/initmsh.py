import os
import sys


def initmsh():
    """
    Initialize the environment for pymesh2d.

    This helper function sets up the Python environment required for
    `pymesh2d`, ensuring that utility and example paths are properly
    configured. It mirrors the initialization behavior of MATLAB’s
    `initmsh` function used in the original `MESH2D` package.

    Notes
    -----
    This routine is primarily used to ensure that dependent modules
    such as `refine`, `smooth`, and `tridemo` are accessible within
    the current Python session.

    References
    ----------
    Translation of the MESH2D function `INITMSH`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    filepath = os.path.dirname(os.path.abspath(__file__))

    subdirs = [
        "aabb_tree",
        "geom_util",
        "hfun_util",
        "hjac_util",
        "mesh_ball",
        "mesh_cost",
        "mesh_file",
        "mesh_util",
        "poly_test",
    ]

    for sub in subdirs:
        fullpath = os.path.join(filepath, sub)
        if fullpath not in sys.path:
            sys.path.append(fullpath)
