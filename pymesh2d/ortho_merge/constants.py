"""
Named constants shared across the ``ortho_merge`` orthogonalization/merge pipeline.

The general geodetic constants live in :mod:`pymesh2d.constants` and are
re-exported here for backwards compatibility.
"""

from ..constants import (  # noqa: F401  (re-exported)
    EARTH_RADIUS,
    DEG2RAD,
    RAD2DEG,
    EARTH_RADIUS_DEG2RAD,
    EARTH_RADIUS_SQ,
    DTOL_POLE,
)

# Default "small flow link" threshold (Delft3D-FM convention): an internal
# edge is flagged as a small flow link when the distance between the two
# adjacent triangle circumcenters is below
# `0.9 * threshold * 0.5 * (sqrt(area1) + sqrt(area2))`.
DEFAULT_SMALLLINK_THRESHOLD = 0.11

# Base amplitude of the per-edge orthogonalization displacement inside
# `apply_combined_ortho_smoother_to_zone` (conservative, to avoid overshoot).
DEFAULT_ORTHO_ALPHA = 0.025
