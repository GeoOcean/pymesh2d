"""
Spherical (lon/lat) counterparts of the planar geometry primitives used by
:mod:`pymesh2d.refine`.

Conventions
-----------
- Vertex coordinates stay in **lon/lat degrees** throughout; nothing is
  projected. Triangulation topology (Delaunay/point-in-polygon) is computed on
  the raw coordinates, while every *measured* quantity uses the sphere:
- All lengths and ball radii are **great-circle metres** (squared where the
  planar code uses squared quantities), so mesh-size functions expressed in
  metres apply directly.
- Ball rows follow the planar layout ``[lon_c, lat_c, r**2]`` with the centre
  in degrees and the squared radius in metres**2; containment means
  "great-circle distance to the centre <= r".
- New points are placed **along geodesics** (great-circle interpolation).

Local, element-scale scalar constructions (e.g. the off-centre distance
algebra in ``refine``) keep their planar formulas applied to great-circle
lengths: for element sizes far below the Earth radius the curvature
corrections are O((L/R)**2) and negligible.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..constants import DEG2RAD, EARTH_RADIUS, RAD2DEG

# ---------------------------------------------------------------------------
# Geodesic primitives (vectorized; lon/lat degrees <-> unit 3-vectors)
# ---------------------------------------------------------------------------


def _to_unit3(lonlat: np.ndarray) -> np.ndarray:
    """(N,2) lon/lat degrees -> (N,3) unit vectors."""
    lon = np.asarray(lonlat, dtype=np.float64)[:, 0] * DEG2RAD
    lat = np.asarray(lonlat, dtype=np.float64)[:, 1] * DEG2RAD
    cl = np.cos(lat)
    return np.column_stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)])


def _to_lonlat(p3: np.ndarray) -> np.ndarray:
    """(N,3) vectors (any radius) -> (N,2) lon/lat degrees."""
    x, y, z = p3[:, 0], p3[:, 1], p3[:, 2]
    lon = np.arctan2(y, x) * RAD2DEG
    lat = np.arctan2(z, np.hypot(x, y)) * RAD2DEG
    return np.column_stack([lon, lat])


def to_conformal(lonlat: np.ndarray) -> np.ndarray:
    """
    Lon/lat degrees -> Mercator (conformal) plane coordinates.

    Used only to compute a *metric-consistent* Delaunay connectivity: Mercator
    is conformal, so small circles map to small circles at every latitude and
    the planar Delaunay of the projected points matches the geodesic Delaunay
    (the same property the UTM/tmerc baseline relies on). Vertices themselves
    stay in lon/lat; only the triangulation predicate uses this frame.
    """
    lonlat = np.asarray(lonlat, dtype=np.float64)
    lat = np.clip(lonlat[:, 1], -89.9999, 89.9999) * DEG2RAD
    x = lonlat[:, 0]
    y = RAD2DEG * np.log(np.tan(np.pi / 4.0 + lat / 2.0))
    return np.column_stack([x, y])


def from_conformal(xy: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_conformal` (Mercator plane -> lon/lat degrees)."""
    xy = np.asarray(xy, dtype=np.float64)
    lon = xy[:, 0]
    lat = (2.0 * np.arctan(np.exp(xy[:, 1] * DEG2RAD)) - np.pi / 2.0) * RAD2DEG
    return np.column_stack([lon, lat])


def sphdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance in metres between lon/lat points (N,2)x(N,2)."""
    pa = _to_unit3(np.atleast_2d(a))
    pb = _to_unit3(np.atleast_2d(b))
    # atan2 form: accurate for both small and large separations
    cross = np.cross(pa, pb)
    sin_d = np.sqrt(np.sum(cross * cross, axis=1))
    cos_d = np.sum(pa * pb, axis=1)
    return EARTH_RADIUS * np.arctan2(sin_d, cos_d)


def sphdist2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared great-circle distance in metres**2."""
    d = sphdist(a, b)
    return d * d


def sphmid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic midpoint of lon/lat point pairs (N,2)."""
    pa = _to_unit3(np.atleast_2d(a))
    pb = _to_unit3(np.atleast_2d(b))
    pm = pa + pb
    norm = np.sqrt(np.sum(pm * pm, axis=1))
    # antipodal pairs (norm ~ 0) do not occur for mesh-scale edges; guard anyway
    norm = np.where(norm < 1.0e-15, 1.0, norm)
    return _to_lonlat(pm / norm[:, None])


def sphmove(a: np.ndarray, b: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Point at great-circle distance ``dist`` (metres) from ``a`` along the
    geodesic towards ``b``. Vectorized over rows; degenerate pairs (a == b)
    return ``a``.
    """
    pa = _to_unit3(np.atleast_2d(a))
    pb = _to_unit3(np.atleast_2d(b))
    dist = np.asarray(dist, dtype=np.float64).ravel()

    cos_ab = np.clip(np.sum(pa * pb, axis=1), -1.0, 1.0)
    # unit tangent at a towards b:  t = (pb - pa*cos_ab) / |...|
    t = pb - pa * cos_ab[:, None]
    tn = np.sqrt(np.sum(t * t, axis=1))
    ok = tn > 1.0e-15
    t = np.where(ok[:, None], t / np.where(tn, tn, 1.0)[:, None], 0.0)

    ang = dist / EARTH_RADIUS
    p = pa * np.cos(ang)[:, None] + t * np.sin(ang)[:, None]
    out = _to_lonlat(p)
    if np.any(~ok):
        aa = np.atleast_2d(np.asarray(a, dtype=np.float64))
        out[~ok] = aa[~ok]
    return out


# ---------------------------------------------------------------------------
# Spherical counterparts of the refine geometry kernels
# ---------------------------------------------------------------------------


def cdtbal1_sph(pp: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """
    Spherical version of :func:`pymesh2d.mesh_ball.cdtbal1.cdtbal1`:
    diametric balls of edges — geodesic midpoints (lon/lat degrees) and
    squared half great-circle lengths (metres**2).
    """
    bb = np.zeros((ee.shape[0], 3))
    a = pp[ee[:, 0], :2]
    b = pp[ee[:, 1], :2]
    bb[:, 0:2] = sphmid(a, b)
    half = 0.5 * sphdist(a, b)
    bb[:, 2] = half * half
    return bb


def minlen_sph(pp: np.ndarray, tt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spherical version of :func:`pymesh2d.mesh_util.minlen.minlen`: squared
    great-circle edge lengths (metres**2), plus the local index of the
    minimum edge per triangle.
    """
    l1 = sphdist2(pp[tt[:, 0], :2], pp[tt[:, 1], :2])
    l2 = sphdist2(pp[tt[:, 1], :2], pp[tt[:, 2], :2])
    l3 = sphdist2(pp[tt[:, 2], :2], pp[tt[:, 0], :2])
    lengths = np.vstack([l1, l2, l3]).T
    ei = np.argmin(lengths, axis=1)
    ll = lengths[np.arange(lengths.shape[0]), ei]
    return ll, ei


def tribal2_sph(pp: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """
    Spherical circumballs of triangles: centre in lon/lat degrees,
    radius**2 in great-circle metres**2.

    The circumcircle is solved in a **local tangent plane (metres)** centred on
    the triangle's first vertex (equirectangular / local ENU frame), using the
    same linear-solve as the planar :func:`pymesh2d.mesh_ball.pwrbal2.pwrbal2`.
    Working in the local metric — rather than global unit 3-vectors — is
    numerically robust for triangles far smaller than the Earth radius (where
    unit-vector differences underflow) and reproduces the planar circumradius
    for tiny/degenerate slivers, which is essential: an inflated radius there
    would spuriously trip the radius-edge test and cause runaway refinement.
    """
    lon0 = pp[tt[:, 0], 0]
    lat0 = pp[tt[:, 0], 1]
    sx = EARTH_RADIUS * DEG2RAD * np.cos(lat0 * DEG2RAD)  # metres per deg lon
    sy = EARTH_RADIUS * DEG2RAD  # metres per deg lat

    # local metric coords with vertex 0 at the origin
    bx = (pp[tt[:, 1], 0] - lon0) * sx
    by = (pp[tt[:, 1], 1] - lat0) * sy
    cx = (pp[tt[:, 2], 0] - lon0) * sx
    cy = (pp[tt[:, 2], 1] - lat0) * sy

    # circumcentre solve (pwrbal2 with zero weights, vertex 0 at origin):
    #   [2bx 2by][ox]   [bx^2+by^2]
    #   [2cx 2cy][oy] = [cx^2+cy^2]
    dd = 2.0 * (bx * cy - by * cx)
    rb = bx * bx + by * by
    rc = cx * cx + cy * cy
    safe = dd != 0.0
    dd_s = np.where(safe, dd, 1.0)
    ox = (cy * rb - by * rc) / dd_s
    oy = (bx * rc - cx * rb) / dd_s

    # squared radius = mean of squared distances to the three vertices (metres),
    # matching pwrbal2's averaging exactly.
    r0 = ox * ox + oy * oy
    r1 = (ox - bx) ** 2 + (oy - by) ** 2
    r2 = (ox - cx) ** 2 + (oy - cy) ** 2
    cc = np.zeros((tt.shape[0], 3))
    cc[:, 0] = lon0 + np.where(safe, ox / sx, 0.0)
    cc[:, 1] = lat0 + np.where(safe, oy / sy, 0.0)
    cc[:, 2] = (r0 + r1 + r2) / 3.0
    return cc


def cdtbal2_sph(pp: np.ndarray, ee: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """
    Spherical version of :func:`pymesh2d.mesh_ball.cdtbal2.cdtbal2`:
    triangle circumballs, replaced by boundary-edge diametric balls when
    those are smaller (keeps the balls inside the constrained domain).
    """
    cc = tribal2_sph(pp, tt)
    for ni, nj, nk in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        cc = _minfac2_sph(cc, pp, ee, tt, ni, nj, nk)
    return cc


def _minfac2_sph(cc, pp, ee, tt, ni, nj, nk):
    """Spherical version of ``cdtbal2.minfac2`` (great-circle metric)."""
    EF = ee[tt[:, ni + 3], 4] > 0
    if not np.any(EF):
        return cc

    pi = pp[tt[EF, ni], :2]
    pj = pp[tt[EF, nj], :2]
    pk = pp[tt[EF, nk], :2]

    bc = sphmid(pi, pj)
    br = 0.5 * (sphdist2(bc, pi) + sphdist2(bc, pj))
    ll = sphdist2(bc, pk)

    bi = (br >= ll) & (br <= cc[EF, 2])
    ei = np.where(EF)[0]
    ti = ei[bi]
    cc[ti, 0:2] = bc[bi, :]
    cc[ti, 2] = br[bi]
    return cc


def findball_sph(bb: np.ndarray, pp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Spherical version of :func:`pymesh2d.aabb_tree.findball.findball`.

    ``bb`` rows are ``[lon_c, lat_c, r**2]`` with the radius in metres; query
    points ``pp`` are lon/lat degrees. A point is inside a ball when its
    great-circle distance to the centre is <= r. The AABB tree is built over
    per-ball lon/lat bounding rectangles derived from the metre radius.
    """
    from ..aabb_tree.maketree import maketree
    from ..aabb_tree.mapvert import mapvert
    from ..aabb_tree.queryset import queryset

    bp, bj = np.array([]), np.array([])
    bb = np.asarray(bb, dtype=float)
    pp = np.asarray(pp, dtype=float)
    if bb.size == 0:
        return bp, bj, None

    # lon/lat bounding half-widths of each great-circle ball
    r_m = np.sqrt(np.maximum(bb[:, 2], 0.0))
    r_arc = r_m / EARTH_RADIUS  # arc angle (radians)
    dlat = r_arc * RAD2DEG
    lat_hi = np.minimum(np.abs(bb[:, 1]) + dlat, 90.0)
    coslat = np.maximum(np.cos(lat_hi * DEG2RAD), 1.0e-9)
    dlon = np.minimum(dlat / coslat, 180.0)

    ab = np.column_stack(
        [bb[:, 0] - dlon, bb[:, 1] - dlat, bb[:, 0] + dlon, bb[:, 1] + dlat]
    )
    tr = maketree(ab, None)

    # Exact great-circle containment via the chord metric: with unit vectors
    # u, v, the arc a satisfies a <= r_arc  <=>  |u - v|^2 <= (2 sin(r_arc/2))^2.
    # Precomputing the unit vectors once makes the per-pair kernel as cheap as
    # the planar one (a subtract + squared-sum), avoiding per-pair trig.
    pp3 = _to_unit3(pp)
    bb3 = _to_unit3(bb[:, 0:2])
    chord2 = (2.0 * np.sin(0.5 * r_arc)) ** 2
    bbk = np.column_stack([bb3, chord2])  # (B, 4): [ux, uy, uz, chord2_thresh]

    tm, _ = mapvert(tr, pp)
    bi, ip, bj_arr = queryset(tr, tm, _ballkern_sph, pp3, bbk)

    # reindex onto the full query list (same layout as the planar findball)
    bp = np.zeros((pp.shape[0], 2), dtype=int)
    bp[:, 1] = -1
    if bi.size > 0:
        bp[bi, :] = ip
    return bp, bj_arr, tr


def _ballkern_sph(pk, bk, pp3, bbk):
    """
    Great-circle ball-vertex kernel using the chord metric (matches
    ``ballkern``'s cost). ``pp3`` are unit vectors of the query points,
    ``bbk`` rows are ``[ux, uy, uz, chord2_thresh]`` for the ball centres.
    """
    mp = len(pk)
    mb = len(bk)
    bk_tiled = np.tile(bk, mp)
    pk_tiled = np.repeat(pk, mb)

    diff = pp3[pk_tiled, :] - bbk[bk_tiled, 0:3]
    dd = np.einsum("ij,ij->i", diff, diff)
    inside = dd <= bbk[bk_tiled, 3]
    return pk_tiled[inside], bk_tiled[inside]


def isfeat_sph(pp: np.ndarray, ee: np.ndarray, tt: np.ndarray):
    """
    Spherical version of :func:`pymesh2d.mesh_util.isfeat.isfeat`: "sharp
    feature" detection with the apex angles measured in the local tangent
    plane at each shared vertex (delta-lon scaled by cos(lat) of the apex)
    instead of raw lon/lat coordinate vectors, which are distorted by the
    cos(lat) anisotropy. Structure and thresholds mirror the planar routine.
    """
    isf = np.zeros((tt.shape[0],), dtype=bool)
    bv = np.zeros((tt.shape[0], 3), dtype=bool)

    EI = [2, 0, 1]
    EJ = [0, 1, 2]
    NI = [2, 0, 1]
    NJ = [0, 1, 2]
    NK = [1, 2, 0]

    for ii in range(3):
        ei = tt[:, EI[ii] + 3]
        ej = tt[:, EJ[ii] + 3]
        bi = ee[ei, 4] >= 1
        bj = ee[ej, 4] >= 1

        ok = bi & bj
        if not np.any(ok):
            continue

        ni = tt[ok, NI[ii]]
        nj = tt[ok, NJ[ii]]
        nk = tt[ok, NK[ii]]
        # edge vectors in the tangent plane at the apex nj
        coslat = np.cos(pp[nj, 1] * DEG2RAD)
        vi = pp[ni, :2] - pp[nj, :2]
        vj = pp[nk, :2] - pp[nj, :2]
        vi = np.column_stack([vi[:, 0] * coslat, vi[:, 1]])
        vj = np.column_stack([vj[:, 0] * coslat, vj[:, 1]])

        li = np.sqrt(np.sum(vi**2, axis=1))
        lj = np.sqrt(np.sum(vj**2, axis=1))
        ll = li * lj
        aa = np.sum(vi * vj, axis=1) / ll

        bv[ok, ii] = aa >= 0.80
        isf[ok] = isf[ok] | bv[ok, ii]

    return isf, bv
