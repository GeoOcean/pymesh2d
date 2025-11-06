import os
import sys

def initmsh():
    """
    INITMSH: helper function to set up Python's path for MESH2D.
    Equivalent to MATLAB initmsh.m
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
        "poly_test"
    ]

    for sub in subdirs:
        fullpath = os.path.join(filepath, sub)
        if fullpath not in sys.path:
            sys.path.append(fullpath)
