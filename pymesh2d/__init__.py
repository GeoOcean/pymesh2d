"""
pymesh2d: Delaunay-based unstructured mesh generation for 2D polygonal geometries.

Python translation of MESH2D (Darren Engwirda), extended with Delft3D-FM /
UGRID-oriented mesh post-processing utilities (see :mod:`pymesh2d.smood` and
:mod:`pymesh2d.ortho_merge`).

This top-level import only exposes the lightweight, dependency-cheap core API.
Geo-processing modules (:mod:`pymesh2d.geomesh_util`, :mod:`pymesh2d.ortho_merge`,
:mod:`pymesh2d.geom_util`) pull in heavier optional dependencies (netCDF4,
rasterio, pyproj, shapely) and should be imported explicitly by the code that
needs them.
"""

from .refine import refine
from .smooth import smooth
from .smood import smood
from .tricost import tricost
from .mesh_cost.triscr import triscr
from .mesh_cost.triang import triang
from .hfun_util.trihfn import trihfn
from .triread import triread
from .mesh_file.loadmsh import loadmsh
from .mesh_file.savemsh import savemsh

__all__ = [
    "refine",
    "smooth",
    "smood",
    "tricost",
    "triscr",
    "triang",
    "trihfn",
    "triread",
    "loadmsh",
    "savemsh",
]
