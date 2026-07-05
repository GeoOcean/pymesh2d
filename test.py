import pickle

import numpy as np

from pymesh2d.smood import smood

# mesh_data.pkl stores a dict of arrays; the coordinates are in a projected
# (metres) CRS, so run smood in planar mode (spherical=False). Use spherical=True
# only when vert is lon/lat degrees.
with open("mesh_data.pkl", "rb") as f:
    data = pickle.load(f)

vert_c = np.asarray(data["vert_c"], dtype=np.float64)
etri_c = np.asarray(data["etri_c"], dtype=np.int64)
tria_c = np.asarray(data["tria_c"], dtype=np.int64)
tnum_c = np.asarray(data["tnum_c"], dtype=np.int64)

vert_2, etri_2, tria_2, tnum_2 = smood(
    vert_c,
    etri_c,
    tria_c,
    tnum_c,
    opts={
        "spherical": False,
        "require_both_criteria": True,
        "iter": 24,
        "preserve_merged_quads": True,
    },
)
