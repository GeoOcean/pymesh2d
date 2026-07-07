"""
Package-wide physical/geodetic constants for spherical (lon/lat) geometry.

Shared by the spherical mesh-generation path (:mod:`pymesh2d.geom_util.sphere`)
and the ``ortho_merge`` orthogonalization pipeline.
"""

import numpy as np

# Earth radius (WGS84 spherical approximation), aligned with Delft's
# `physicalconsts`. Used for lon/lat <-> local-metric distance conversions.
EARTH_RADIUS = 6378137.0
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
EARTH_RADIUS_DEG2RAD = EARTH_RADIUS * DEG2RAD
EARTH_RADIUS_SQ = EARTH_RADIUS * EARTH_RADIUS

# Distance-to-pole tolerance (degrees) below which pole-specific handling
# kicks in for spherical distance/circumcenter computations.
DTOL_POLE = 1.0e-6
