import numpy as np
from pymesh2d.geom_util.proj_util import get_utm_crs_from_crs

from pymesh2d.geom_util.poly_util import polygon_to_node_edge
node, edge = polygon_to_node_edge(input polygone)

crs = crs_input_polygone

from pymesh2d.refine import refine
from pymesh2d.smooth import smooth

opts = {"kind": "delaunay"}

vert, etri, tria, tnum = refine(node, edge, [], opts)
vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)

from pymesh2d.geom_util.proj_util import reproject_node

vert = reproject_node(vert, get_utm_crs_from_crs(crs), crs)

z = np.zeros(vert.shape[0])

import numpy as np

from pymesh2d.geomesh_util.grd_util import adcirc2DFlowFM

adcirc2DFlowFM(np.column_stack((vert, z)), tria, "GEBCO_grid.nc")