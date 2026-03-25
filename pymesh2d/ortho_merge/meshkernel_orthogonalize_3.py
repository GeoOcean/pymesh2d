"""
Local orthogonalization of a UGRID mesh (*_net.nc) using |cosphi| (3D circumcenter).
V3: v2 + pymesh2d-inspired small-link fixes (flip, circumcenter-direction, aggressive, optional merge).

- Edge flip: try flipping small-link edges (convex quad) to lengthen link without moving nodes.
- Small-link displacement: move opposite vertices along circumcenter separation direction.
- Aggressive mode when few/single small link remains (larger step).
- Optional merge_circumcenters at end to convert remaining small-link pairs to quads.
- [good] zones: skip if n_small_zone=0; line search accepts no-degradation (v2 Improvement 5).
- All internal indexing is 0-based (invalid = -1). NetCDF load converts to 0-based.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import deque

import numpy as np
from netCDF4 import Dataset

from ..geomesh_util.grd_util import adcirc2DFlowFM


# ---------------------------------------------------------------------------
# Orthogonality: constants, UGRID load, geometry, circumcenter, cosphi (1-based internally).
# ---------------------------------------------------------------------------

# Constants (aligned with Delft physicalconsts)
EARTH_RADIUS = 6378137.0
DEG2RAD = np.pi / 180.0
EARTH_RADIUS_DEG2RAD = EARTH_RADIUS * DEG2RAD
EARTH_RADIUS_SQ = EARTH_RADIUS * EARTH_RADIUS
DTOL_POLE = 1.0e-6
RAD2DEG = 180.0 / np.pi

# Toggle verbose "[ZONE] ..." logs during orthogonalization.
# The ortho+merge pipeline can set this to False to keep output compact.
VERBOSE_ZONE_LOGS: bool = True


def _ugrid_fill_value(var) -> int:
    """Return FillValue or _FillValue for a NetCDF variable; default -1."""
    return int(
        getattr(var, "_FillValue", getattr(var, "FillValue", -1))
    )


def _load_ugrid(netcdf_path: str) -> Tuple:
    """Load mesh2d_node_x/y, face_nodes, edge_nodes, edge_faces (1-based), face_x/y if present."""
    with Dataset(netcdf_path, "r") as ds:
        node_x = np.asarray(ds["mesh2d_node_x"][:], dtype=np.float64).ravel()
        node_y = np.asarray(ds["mesh2d_node_y"][:], dtype=np.float64).ravel()
        face_nodes = np.asarray(ds["mesh2d_face_nodes"][:], dtype=np.int64)
        edge_nodes = np.asarray(ds["mesh2d_edge_nodes"][:], dtype=np.int64)
        if edge_nodes.ndim == 1:
            edge_nodes = edge_nodes.reshape(-1, 2)
        # Normalize to 1-based so _to_0b yields correct 0-based indices
        vfn = ds.variables["mesh2d_face_nodes"]
        start_fn = int(getattr(vfn, "start_index", 1))
        if start_fn == 0:
            fill_fn = _ugrid_fill_value(vfn)
            valid_fn = (face_nodes >= 0) & (face_nodes != fill_fn)
            out_fn = np.zeros_like(face_nodes)
            out_fn[valid_fn] = face_nodes[valid_fn] + 1
            face_nodes = out_fn
        ven = ds.variables["mesh2d_edge_nodes"]
        start_en = int(getattr(ven, "start_index", 1))
        if start_en == 0:
            fill_en = _ugrid_fill_value(ven)
            valid_en = (edge_nodes >= 0) & (edge_nodes != fill_en)
            out_en = np.zeros_like(edge_nodes)
            out_en[valid_en] = edge_nodes[valid_en] + 1
            edge_nodes = out_en
        face_x_file = face_y_file = None
        if "mesh2d_face_x" in ds.variables and "mesh2d_face_y" in ds.variables:
            face_x_file = np.asarray(ds["mesh2d_face_x"][:], dtype=np.float64).ravel()
            face_y_file = np.asarray(ds["mesh2d_face_y"][:], dtype=np.float64).ravel()
        edge_faces = None
        if "mesh2d_edge_faces" in ds.variables:
            edge_faces = np.asarray(ds["mesh2d_edge_faces"][:], dtype=np.int64)
            if edge_faces.ndim == 1:
                edge_faces = edge_faces.reshape(-1, 2)
            start = int(getattr(ds.variables["mesh2d_edge_faces"], "start_index", 1))
            if start == 0:
                fill = _ugrid_fill_value(ds.variables["mesh2d_edge_faces"])
                valid = (edge_faces >= 0) & (edge_faces != fill)
                out = np.zeros_like(edge_faces)
                out[valid] = edge_faces[valid] + 1
                edge_faces = out
    return node_x, node_y, face_nodes, edge_nodes, edge_faces, face_x_file, face_y_file


def _getdx_vec(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, jsferic: int = 1
) -> np.ndarray:
    """Vectorized version of Delft's _getdx. x1, y1, x2, y2 same shape."""
    if jsferic != 1:
        return (x2 - x1).astype(np.float64)
    pole1 = np.abs(np.abs(y1) - 90.0) <= DTOL_POLE
    pole2 = np.abs(np.abs(y2) - 90.0) <= DTOL_POLE
    different_poles = pole1 != pole2
    xx1 = x1.astype(np.float64)
    mask_hi = (xx1 - x2) > 180.0
    mask_lo = (xx1 - x2) < -180.0
    xx1 = np.where(mask_hi, xx1 - 360.0, xx1)
    xx1 = np.where(mask_lo, xx1 + 360.0, xx1)
    c = np.cos(0.5 * (y1 + y2) * DEG2RAD)
    out = EARTH_RADIUS_DEG2RAD * c * (x2 - xx1)
    out = np.where(different_poles, 0.0, out)
    return out


def _getdy_vec(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, jsferic: int = 1
) -> np.ndarray:
    """Vectorized version of Delft's _getdy."""
    if jsferic != 1:
        return (y2 - y1).astype(np.float64)
    return (EARTH_RADIUS_DEG2RAD * (y2 - y1)).astype(np.float64)


def _face_centers(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Barycentric cell centers (comp_masscenter2D jsferic=1)."""
    n_faces = face_nodes.shape[0]
    if face_mask is not None:
        face_x = np.full(n_faces, np.nan, dtype=np.float64)
        face_y = np.full(n_faces, np.nan, dtype=np.float64)
        faces_to_do = np.where(face_mask)[0]
    else:
        face_x = np.zeros(n_faces, dtype=np.float64)
        face_y = np.zeros(n_faces, dtype=np.float64)
        faces_to_do = np.arange(n_faces)
    for f in faces_to_do:
        nodes = face_nodes[f, :]
        nodes = nodes[nodes > 0]
        if nodes.size == 0:
            continue
        idx = nodes - 1
        xin = node_x[idx].astype(float)
        yin = node_y[idx].astype(float)
        n = xin.size
        x, y0 = xin.copy(), yin[np.argmin(np.abs(yin))]
        x0 = np.min(x)
        if np.max(x) - x0 > 180.0:
            x = np.where(x < np.max(x) - 180.0, x + 360.0, x)
            x0 = np.min(x)
        area = xcg = ycg = 0.0
        for i in range(n):
            ip1 = (i + 1) % n
            dx0 = _getdx(x0, y0, x[i], yin[i], 1)
            dy0 = _getdy(x0, y0, x[i], yin[i], 1)
            dx1 = _getdx(x0, y0, x[ip1], yin[ip1], 1)
            dy1 = _getdy(x0, y0, x[ip1], yin[ip1], 1)
            xc, yc = 0.5 * (dx0 + dx1), 0.5 * (dy0 + dy1)
            dxe = _getdx(x[i], yin[i], x[ip1], yin[ip1], 1)
            dye = _getdy(x[i], yin[i], x[ip1], yin[ip1], 1)
            dsx, dsy = dye, -dxe
            xds = xc * dsx + yc * dsy
            area += 0.5 * xds
            xcg += xds * xc
            ycg += xds * yc
        if abs(area) < 1e-8:
            face_x[f], face_y[f] = xin.mean(), yin.mean()
            continue
        area = np.sign(area) * max(abs(area), 1e-8)
        fac = 1.0 / (3.0 * area)
        xcg *= fac
        ycg *= fac
        face_y[f] = y0 + ycg / EARTH_RADIUS_DEG2RAD
        face_x[f] = x0 + xcg / (EARTH_RADIUS_DEG2RAD * np.cos(face_y[f] * DEG2RAD))
    return face_x, face_y


def _sphertocart3d_vec(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Vectorized: (N,) -> (N, 3) in Cartesian meters."""
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    r = EARTH_RADIUS * np.cos(lat_deg * DEG2RAD)
    xx = r * np.cos(lon_deg * DEG2RAD)
    yy = r * np.sin(lon_deg * DEG2RAD)
    zz = EARTH_RADIUS * np.sin(lat_deg * DEG2RAD)
    return np.column_stack([xx, yy, zz])


def _cart3dtospher(xx: float, yy: float, zz: float, xref_deg: float) -> Tuple[float, float]:
    """Cartesian (meters) -> spherical (deg)."""
    raddeg = 180.0 / np.pi
    x1 = np.arctan2(yy, xx) * raddeg
    y1 = np.arctan2(zz, np.sqrt(xx * xx + yy * yy)) * raddeg
    x1 = x1 + np.round((xref_deg - x1) / 360.0) * 360.0
    return float(x1), float(y1)


def _getdx(x1: float, y1: float, x2: float, y2: float, jsferic: int = 1) -> float:
    """Scalar version of Delft _getdx (copied from meshkernel_orthogonality)."""
    if jsferic != 1:
        return float(x2 - x1)
    if (abs(abs(y1) - 90.0) <= DTOL_POLE) != (abs(abs(y2) - 90.0) <= DTOL_POLE):
        return 0.0
    xx1, xx2 = x1, x2
    if xx1 - xx2 > 180.0:
        xx1 -= 360.0
    elif xx1 - xx2 < -180.0:
        xx1 += 360.0
    c = np.cos(0.5 * (y1 + y2) * DEG2RAD)
    return float(EARTH_RADIUS_DEG2RAD * c * (xx2 - xx1))


def _getdy(x1: float, y1: float, x2: float, y2: float, jsferic: int = 1) -> float:
    """Scalar version of Delft _getdy (copied from meshkernel_orthogonality)."""
    if jsferic != 1:
        return float(y2 - y1)
    return float(EARTH_RADIUS_DEG2RAD * (y2 - y1))


# ---------------------------------------------------------------------------
# Small flow links (Delft3D): circumcenters in lon/lat, dxlink < 0.9*thresh*0.5*(sqrt(ba1)+sqrt(ba2))
# Same logic as circo; no import from circo to avoid circular dependency.
# ---------------------------------------------------------------------------

def _lonlat_to_local_xy(node_x: np.ndarray, node_y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Lon/lat (deg) -> local planar (m) around reference (x0,y0) using _getdx/_getdy."""
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)
    x0 = float(np.nanmean(node_x))
    y0 = float(np.nanmean(node_y))
    dx = np.empty_like(node_x)
    dy = np.empty_like(node_y)
    for i in range(node_x.size):
        dx[i] = _getdx(x0, y0, node_x[i], node_y[i], 1)
        dy[i] = _getdy(x0, y0, node_x[i], node_y[i], 1)
    vert_xy = np.column_stack([dx, dy])
    return vert_xy, x0, y0


def _triarea_2d(pp: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """Signed triangle areas in 2D (pp: nnode x 2, tt: nface x 3, 0-based indices)."""
    ev12 = pp[tt[:, 1], :] - pp[tt[:, 0], :]
    ev13 = pp[tt[:, 2], :] - pp[tt[:, 0], :]
    area = 0.5 * (ev12[:, 0] * ev13[:, 1] - ev12[:, 1] * ev13[:, 0])
    return area


def _circumcenter_of_triangle_lonlat(
    n0: np.ndarray, n1: np.ndarray, n2: np.ndarray
) -> np.ndarray:
    """Circumcenter of triangle in lon/lat (deg), MeshKernel spherical formula."""
    x1, y1 = float(n0[0]), float(n0[1])
    x2, y2 = float(n1[0]), float(n1[1])
    x3, y3 = float(n2[0]), float(n2[1])
    dx2 = _getdx(x1, y1, x2, y2, 1)
    dy2 = _getdy(x1, y1, x2, y2, 1)
    dx3 = _getdx(x1, y1, x3, y3, 1)
    dy3 = _getdy(x1, y1, x3, y3, 1)
    den = dy2 * dx3 - dy3 * dx2
    z = (dx2 * (dx2 - dx3) + dy2 * (dy2 - dy3)) / den if abs(den) > 1e-20 else 0.0
    phi = (y1 + y2 + y3) / 3.0
    xf = 1.0 / np.cos(phi * DEG2RAD)
    cx = x1 + xf * 0.5 * (dx3 - z * dy3) * RAD2DEG / EARTH_RADIUS
    cy = y1 + 0.5 * (dy3 + z * dx3) * RAD2DEG / EARTH_RADIUS
    return np.array([cx, cy], dtype=np.float64)


def _cross_product_cartesian2d(
    seg_a: np.ndarray, seg_b: np.ndarray, point: np.ndarray
) -> float:
    """Cross product (seg_b - seg_a) x (point - seg_a) in (x,y) plane."""
    return float(
        (seg_b[0] - seg_a[0]) * (point[1] - seg_a[1])
        - (seg_b[1] - seg_a[1]) * (point[0] - seg_a[0])
    )


def _point_in_triangle_winding_lonlat(
    point: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> bool:
    """Point-in-triangle via winding in (lon, lat), MeshKernel-style."""
    tol = 1e-12
    winding = 0
    for va, vb in ((v0, v1), (v1, v2), (v2, v0)):
        cp = _cross_product_cartesian2d(va, vb, point)
        if abs(cp) <= tol:
            return True
        if va[1] <= point[1]:
            if vb[1] > point[1] and cp > 0:
                winding += 1
        else:
            if vb[1] <= point[1] and cp < 0:
                winding -= 1
    return winding != 0


def _segments_crossing_ratio_intersection_lonlat(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> Optional[Tuple[float, np.ndarray]]:
    """If (p1,p2) and (p3,p4) cross, return (ratio_first, intersection) in lon/lat."""
    x21 = _getdx(p1[0], p1[1], p2[0], p2[1], 1)
    y21 = _getdy(p1[0], p1[1], p2[0], p2[1], 1)
    x43 = _getdx(p3[0], p3[1], p4[0], p4[1], 1)
    y43 = _getdy(p3[0], p3[1], p4[0], p4[1], 1)
    x31 = _getdx(p1[0], p1[1], p3[0], p3[1], 1)
    y31 = _getdy(p1[0], p1[1], p3[0], p3[1], 1)
    det = x43 * y21 - y43 * x21
    max_val = max(abs(x21), abs(y21), abs(x43), abs(y43), 1e-30)
    if abs(det) < max(1e-10 * max_val, 1e-15):
        return None
    ratio_second = (y31 * x21 - x31 * y21) / det
    ratio_first = (y31 * x43 - x31 * y43) / det
    if not (0.0 <= ratio_first <= 1.0 and 0.0 <= ratio_second <= 1.0):
        return None
    inter = p1 + ratio_first * (p2 - p1)
    return (float(ratio_first), np.asarray(inter, dtype=np.float64))


# Minimum number of faces to use parallel circumcenter computation at init
_CIRCUM_PARALLEL_MIN_FACES = 12000


def _circumcenters_lonlat_chunk(
    vert_deg: np.ndarray,
    face_nodes: np.ndarray,
    num_interior: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray:
    """
    Worker for parallel circumcenters: compute for faces in face_indices only.
    Returns array of shape (len(face_indices), 2). Used at init when nface is large.
    """
    tria = face_nodes[:, :3]
    out_chunk = np.zeros((len(face_indices), 2), dtype=np.float64)
    for pos, t_idx in enumerate(face_indices):
        i0, i1, i2 = tria[t_idx, 0], tria[t_idx, 1], tria[t_idx, 2]
        if i0 < 0 or i1 < 0 or i2 < 0:
            out_chunk[pos] = np.nan
            continue
        v0 = vert_deg[int(i0)]
        v1 = vert_deg[int(i1)]
        v2 = vert_deg[int(i2)]
        if num_interior[t_idx] == 0:
            out_chunk[pos] = np.mean([v0, v1, v2], axis=0)
            continue
        circum = _circumcenter_of_triangle_lonlat(v0, v1, v2)
        mass = np.mean([v0, v1, v2], axis=0)
        if _point_in_triangle_winding_lonlat(circum, v0, v1, v2):
            out_chunk[pos] = circum
            continue
        for n in range(3):
            next_n = (n + 1) % 3
            va = vert_deg[tria[t_idx, n]]
            vb = vert_deg[tria[t_idx, next_n]]
            hit = _segments_crossing_ratio_intersection_lonlat(mass, circum, va, vb)
            if hit is not None:
                _, inter = hit
                out_chunk[pos] = inter
                break
        else:
            out_chunk[pos] = mass
    return out_chunk


def _circumcenters_lonlat_ugrid(
    vert_deg: np.ndarray,
    face_nodes: np.ndarray,
    edge_faces: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Circumcenters in lon/lat per face (UGRID). Boundary faces use mass center.
    Same logic as circo circumcenters_lonlat; numberOfInteriorEdges from edge_faces.
    If face_mask is provided, only compute for faces where face_mask is True (faster for zones).
    For full-mesh (face_mask is None) with many faces, uses parallel chunks to speed up init.
    """
    nface = face_nodes.shape[0]
    n_edges = edge_faces.shape[0]
    # Number of interior edges per face (edge with two valid faces)
    num_interior = np.zeros(nface, dtype=np.int32)
    for e in range(n_edges):
        f1, f2 = edge_faces[e, 0], edge_faces[e, 1]
        if f1 >= 0 and f2 >= 0 and f1 != f2:
            num_interior[int(f1)] += 1
            num_interior[int(f2)] += 1

    out = np.full((nface, 2), np.nan, dtype=np.float64) if face_mask is not None else np.zeros((nface, 2), dtype=np.float64)
    tria = face_nodes[:, :3]
    indices_to_compute = np.where(face_mask)[0] if face_mask is not None else np.arange(nface, dtype=np.int64)

    # Parallel path for full-mesh init when many faces (avoids long single-thread loop)
    use_parallel = (
        face_mask is None
        and nface >= _CIRCUM_PARALLEL_MIN_FACES
        and len(indices_to_compute) >= _CIRCUM_PARALLEL_MIN_FACES
    )
    if use_parallel:
        n_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
        chunks = np.array_split(indices_to_compute, n_workers)
        chunks = [c for c in chunks if c.size > 0]
        if len(chunks) <= 1:
            use_parallel = False
    if use_parallel:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(_circumcenters_lonlat_chunk, vert_deg, face_nodes, num_interior, chunk)
                for chunk in chunks
            ]
            for chunk, fut in zip(chunks, futures):
                out[chunk] = fut.result()
        return out

    for t_idx in indices_to_compute:
        i0, i1, i2 = tria[t_idx, 0], tria[t_idx, 1], tria[t_idx, 2]
        if i0 < 0 or i1 < 0 or i2 < 0:
            out[t_idx] = np.nan
            continue
        v0 = vert_deg[int(i0)]
        v1 = vert_deg[int(i1)]
        v2 = vert_deg[int(i2)]

        if num_interior[t_idx] == 0:
            out[t_idx] = np.mean([v0, v1, v2], axis=0)
            continue

        circum = _circumcenter_of_triangle_lonlat(v0, v1, v2)
        mass = np.mean([v0, v1, v2], axis=0)

        if _point_in_triangle_winding_lonlat(circum, v0, v1, v2):
            out[t_idx] = circum
            continue

        for n in range(3):
            next_n = (n + 1) % 3
            va = vert_deg[tria[t_idx, n]]
            vb = vert_deg[tria[t_idx, next_n]]
            hit = _segments_crossing_ratio_intersection_lonlat(mass, circum, va, vb)
            if hit is not None:
                _, inter = hit
                out[t_idx] = inter
                break
        else:
            out[t_idx] = mass

    return out


def compute_small_links_from_arrays(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    edge_nodes: np.ndarray,
    edge_faces: np.ndarray,
    removesmalllinkstrsh: float = 0.11,
    edge_indices: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray]:
    """
    Small flow links (Delft3D): dxlink < 0.9*removesmalllinkstrsh*0.5*(sqrt(ba1)+sqrt(ba2)).
    Inputs 0-based (invalid = -1). Returns (n_small, edge_indices_of_small_links).
    If edge_indices is provided, only those edges are tested (returned small_edges are a subset).
    """
    node_x = np.asarray(node_x, dtype=np.float64).ravel()
    node_y = np.asarray(node_y, dtype=np.float64).ravel()
    nface = face_nodes.shape[0]
    n_edges = edge_faces.shape[0]
    if nface == 0 or n_edges == 0:
        return 0, np.array([], dtype=np.int64)

    vert_deg = np.column_stack([node_x, node_y])
    vert_xy, _x0, _y0 = _lonlat_to_local_xy(node_x, node_y)
    edges_to_test = (
        np.asarray(edge_indices, dtype=np.int64).ravel()
        if edge_indices is not None
        else np.arange(n_edges, dtype=np.int64)
    )
    # When testing only a subset of edges, compute circumcenters only for adjacent faces (big speedup)
    face_mask = None
    if edge_indices is not None and edges_to_test.size > 0:
        face_mask = np.zeros(nface, dtype=bool)
        for e in edges_to_test:
            if e < 0 or e >= n_edges:
                continue
            f1, f2 = edge_faces[e, 0], edge_faces[e, 1]
            if f1 >= 0:
                face_mask[int(f1)] = True
            if f2 >= 0:
                face_mask[int(f2)] = True
    circum_ll = _circumcenters_lonlat_ugrid(vert_deg, face_nodes, edge_faces, face_mask=face_mask)

    tria = face_nodes[:, :3]
    valid_tria = (tria[:, 0] >= 0) & (tria[:, 1] >= 0) & (tria[:, 2] >= 0)
    ba = np.zeros(nface, dtype=np.float64)
    ba[valid_tria] = np.abs(_triarea_2d(vert_xy, tria[valid_tria]))
    small_edges: List[int] = []
    for e in edges_to_test:
        if e < 0 or e >= n_edges:
            continue
        f1, f2 = edge_faces[e, 0], edge_faces[e, 1]
        if f1 < 0 or f2 < 0 or f1 == f2:
            continue
        f1, f2 = int(f1), int(f2)
        if f1 >= nface or f2 >= nface or not valid_tria[f1] or not valid_tria[f2]:
            continue
        c1, c2 = circum_ll[f1], circum_ll[f2]
        if np.any(np.isnan(c1)) or np.any(np.isnan(c2)):
            continue
        dx = _getdx(c1[0], c1[1], c2[0], c2[1], 1)
        dy = _getdy(c1[0], c1[1], c2[0], c2[1], 1)
        dxlink = np.sqrt(dx * dx + dy * dy)
        sqrt_ba1 = np.sqrt(max(ba[f1], 1e-20))
        sqrt_ba2 = np.sqrt(max(ba[f2], 1e-20))
        dxlim = 0.9 * removesmalllinkstrsh * 0.5 * (sqrt_ba1 + sqrt_ba2)
        if dxlink < dxlim:
            small_edges.append(e)

    return len(small_edges), np.array(small_edges, dtype=np.int64)


def _signed_area_tri_deg(node_x: np.ndarray, node_y: np.ndarray, i: int, j: int, k: int) -> float:
    """Signed area (doubled) of triangle (i,j,k) in lon/lat degrees. Positive = CCW."""
    xi, yi = node_x[i], node_y[i]
    xj, yj = node_x[j], node_y[j]
    xk, yk = node_x[k], node_y[k]
    return (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)


def try_flip_small_flow_edge_ugrid(
    mesh: "MeshData",
    edge_index: int,
) -> bool:
    """
    Try to fix a small flow link by flipping the shared edge (swap diagonal of the quad).
    The two triangles must form a strictly convex quadrilateral. Modifies mesh.face_nodes in place.
    Returns True if the edge was flipped.
    """
    face_nodes = mesh.face_nodes
    edge_nodes = mesh.edge_nodes
    edge_faces = mesh.edge_faces
    node_x = mesh.node_x
    node_y = mesh.node_y
    n_face = face_nodes.shape[0]
    if edge_index < 0 or edge_index >= edge_faces.shape[0]:
        return False
    k3, k4 = int(edge_nodes[edge_index, 0]), int(edge_nodes[edge_index, 1])
    f1, f2 = int(edge_faces[edge_index, 0]), int(edge_faces[edge_index, 1])
    if k3 < 0 or k4 < 0 or f1 < 0 or f2 < 0 or f1 == f2 or f1 >= n_face or f2 >= n_face:
        return False
    tri1 = face_nodes[f1, :3].copy()
    tri2 = face_nodes[f2, :3].copy()
    opp1 = None
    for v in tri1:
        v = int(v)
        if v >= 0 and v != k3 and v != k4:
            opp1 = v
            break
    opp2 = None
    for v in tri2:
        v = int(v)
        if v >= 0 and v != k3 and v != k4:
            opp2 = v
            break
    if opp1 is None or opp2 is None or opp1 == opp2:
        return False
    # Convex quad: k3 and k4 on opposite sides of line (opp1, opp2)
    sa1 = _signed_area_tri_deg(node_x, node_y, opp1, opp2, k3)
    sa2 = _signed_area_tri_deg(node_x, node_y, opp1, opp2, k4)
    if sa1 * sa2 >= 0:
        return False
    # Both new triangles must have positive area (CCW)
    if sa1 <= 0:
        return False
    # New triangles: (opp1, opp2, k3) and (opp2, opp1, k4)
    face_nodes[f1, 0], face_nodes[f1, 1], face_nodes[f1, 2] = opp1, opp2, k3
    face_nodes[f2, 0], face_nodes[f2, 1], face_nodes[f2, 2] = opp2, opp1, k4
    return True


def try_flip_small_flow_edges_ugrid(
    mesh: "MeshData",
    small_edges_arr: np.ndarray,
    removesmalllinkstrsh: float,
    max_flip_iter: int = 20,
) -> int:
    """
    Repeatedly try to flip small flow edges (convex quad). After each flip, recompute small links.
    Returns total number of flips performed.
    """
    total_flipped = 0
    for _ in range(max_flip_iter):
        flipped_any = False
        for ei in range(small_edges_arr.size):
            e = int(small_edges_arr[ei])
            if try_flip_small_flow_edge_ugrid(mesh, e):
                total_flipped += 1
                flipped_any = True
                break
        if not flipped_any:
            break
        _, small_edges_arr = compute_small_links_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
        )
        if small_edges_arr.size == 0:
            break
    return total_flipped


def _point_in_polygon(px: float, py: float, xv: np.ndarray, yv: np.ndarray, x0: float, y0: float) -> bool:
    """Exact copy from meshkernel_orthogonality: ray casting in projected coords."""
    n = len(xv)
    if n < 3:
        return False
    dxp = _getdx(x0, y0, px, py, 1)
    dyp = _getdy(x0, y0, px, py, 1)
    count = 0
    for i in range(n):
        ip1 = (i + 1) % n
        dx_i = _getdx(x0, y0, xv[i], yv[i], 1)
        dy_i = _getdy(x0, y0, xv[i], yv[i], 1)
        dx_ip1 = _getdx(x0, y0, xv[ip1], yv[ip1], 1)
        dy_ip1 = _getdy(x0, y0, xv[ip1], yv[ip1], 1)
        if (dy_i <= dyp < dy_ip1) or (dy_ip1 <= dyp < dy_i):
            if abs(dy_ip1 - dy_i) < 1e-12:
                continue
            t = (dyp - dy_i) / (dy_ip1 - dy_i)
            x_cross = dx_i + t * (dx_ip1 - dx_i)
            if x_cross > dxp:
                count += 1
    return (count % 2) == 1


def _segment_edge_intersect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    x0: float,
    y0: float,
):
    """Exact copy from meshkernel_orthogonality: segment–edge intersection in projected coords."""
    dx1 = _getdx(x0, y0, x1, y1, 1)
    dy1 = _getdy(x0, y0, x1, y1, 1)
    dx2 = _getdx(x0, y0, x2, y2, 1)
    dy2 = _getdy(x0, y0, x2, y2, 1)
    dxa = _getdx(x0, y0, xa, ya, 1)
    dya = _getdy(x0, y0, xa, ya, 1)
    dxb = _getdx(x0, y0, xb, yb, 1)
    dyb = _getdy(x0, y0, xb, yb, 1)
    den = (dx2 - dx1) * (dyb - dya) - (dy2 - dy1) * (dxb - dxa)
    if abs(den) < 1e-15:
        return None
    t = ((dxa - dx1) * (dyb - dya) - (dya - dy1) * (dxb - dxa)) / den
    s = ((dxa - dx1) * (dy2 - dy1) - (dya - dy1) * (dx2 - dx1)) / den
    if not (0 <= t <= 1 and 0 <= s <= 1):
        return None
    xcr_l = dx1 + t * (dx2 - dx1)
    ycr_l = dy1 + t * (dy2 - dy1)
    ycr_deg = y0 + ycr_l / EARTH_RADIUS_DEG2RAD
    xcr_deg = x0 + xcr_l / (EARTH_RADIUS_DEG2RAD * np.cos(ycr_deg * DEG2RAD))
    return (xcr_deg, ycr_deg, t)


def _circumcenter3d(xv: np.ndarray, yv: np.ndarray) -> Tuple[float, float]:
    """Spherical 3D circumcenter (comp_circumcenter3D). xv, yv in deg -> (xz, yz) deg."""
    N = len(xv)
    if N < 2:
        return float(xv[0]), float(yv[0])
    xx = _sphertocart3d_vec(np.asarray(xv, dtype=np.float64), np.asarray(yv, dtype=np.float64))
    xxc, yyc, zzc = np.mean(xx[:, 0]), np.mean(xx[:, 1]), np.mean(xx[:, 2])
    dtol, deps, maxiter = 1e-8, 1e-8, 100
    ip1 = np.arange(N)
    ip1 = (ip1 + 1) % N
    ttx = xx[ip1, 0] - xx[:, 0]
    tty = xx[ip1, 1] - xx[:, 1]
    ttz = xx[ip1, 2] - xx[:, 2]
    ds = np.sqrt(ttx * ttx + tty * tty + ttz * ttz)
    valid = ds >= dtol
    dsi = np.where(valid, 1.0 / ds, 0.0)
    ttx = ttx * dsi
    tty = tty * dsi
    ttz = ttz * dsi
    xxe = 0.5 * (xx[:, 0] + xx[ip1, 0])
    yye = 0.5 * (xx[:, 1] + xx[ip1, 1])
    zze = 0.5 * (xx[:, 2] + xx[ip1, 2])
    lam = 0.0
    for _ in range(maxiter):
        A = np.zeros((4, 4))
        rhs = np.zeros(4)
        for i in range(N):
            if ds[i] < dtol:
                continue
            A[0, 0] += ttx[i] * ttx[i]
            A[0, 1] += ttx[i] * tty[i]
            A[0, 2] += ttx[i] * ttz[i]
            A[1, 1] += tty[i] * tty[i]
            A[1, 2] += tty[i] * ttz[i]
            A[2, 2] += ttz[i] * ttz[i]
            dinpr = (xxc - xxe[i]) * ttx[i] + (yyc - yye[i]) * tty[i] + (zzc - zze[i]) * ttz[i]
            rhs[0] -= dinpr * ttx[i]
            rhs[1] -= dinpr * tty[i]
            rhs[2] -= dinpr * ttz[i]
        A[0, 0] -= 2 * lam
        A[1, 1] -= 2 * lam
        A[2, 2] -= 2 * lam
        A[0, 3], A[1, 3], A[2, 3] = -2 * xxc, -2 * yyc, -2 * zzc
        A[3, 3] = 0.0
        rhs[0] += 2 * lam * xxc
        rhs[1] += 2 * lam * yyc
        rhs[2] += 2 * lam * zzc
        rhs[3] = xxc * xxc + yyc * yyc + zzc * zzc - EARTH_RADIUS_SQ
        A[1, 0], A[2, 0], A[2, 1] = A[0, 1], A[0, 2], A[1, 2]
        A[3, 0], A[3, 1], A[3, 2] = A[0, 3], A[1, 3], A[2, 3]
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            break
        xxc += sol[0]
        yyc += sol[1]
        zzc += sol[2]
        lam += sol[3]
        if sol[0] ** 2 + sol[1] ** 2 + sol[2] ** 2 < deps:
            break
    return _cart3dtospher(xxc, yyc, zzc, float(np.max(xv)))


def _face_centers_circumcenter3d(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    dcenterinside: float = 1.0,
    face_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centers = 3D circumcenter, with pull-inside when outside cell (as in Delft).
    If face_mask is provided, only compute for faces where face_mask[f] is True;
    others are left as nan.
    """
    mass_x, mass_y = _face_centers(node_x, node_y, face_nodes, face_mask=face_mask)
    n_faces = face_nodes.shape[0]
    if face_mask is not None:
        face_x = np.full(n_faces, np.nan, dtype=np.float64)
        face_y = np.full(n_faces, np.nan, dtype=np.float64)
    else:
        face_x = np.zeros(n_faces, dtype=np.float64)
        face_y = np.zeros(n_faces, dtype=np.float64)
    for f in range(n_faces):
        if face_mask is not None and not face_mask[f]:
            continue
        nodes = face_nodes[f, :]
        nodes = nodes[nodes > 0]
        if nodes.size < 2:
            if nodes.size == 1:
                i = int(nodes[0]) - 1
                face_x[f], face_y[f] = node_x[i], node_y[i]
            else:
                face_x[f], face_y[f] = mass_x[f], mass_y[f]
            continue
        idx = (nodes - 1).astype(np.intp)
        xv = node_x[idx].astype(np.float64)
        yv = node_y[idx].astype(np.float64)
        xz, yz = _circumcenter3d(xv, yv)
        if len(xv) == 3:
            iv = ih = -1
            for k in range(3):
                k2 = (k + 1) % 3
                if abs(xv[k] - xv[k2]) < 1e-10:
                    iv = k
                if abs(yv[k] - yv[k2]) < 1e-10:
                    ih = k
            if iv >= 0 and ih >= 0:
                xh = np.array([xv.min(), xv.max(), xv.max(), xv.min()])
                yh = np.array([yv.min(), yv.min(), yv.max(), yv.max()])
                xz, yz = _circumcenter3d(xh, yh)
        if 0 <= dcenterinside <= 1:
            x0, y0 = float(np.min(xv)), float(yv[np.argmin(np.abs(yv))])
            if not _point_in_polygon(xz, yz, xv, yv, x0, y0):
                best_t, xcr, ycr = 2.0, xz, yz
                for i in range(len(xv)):
                    ip1 = (i + 1) % len(xv)
                    hit = _segment_edge_intersect(
                        mass_x[f],
                        mass_y[f],
                        xz,
                        yz,
                        xv[i],
                        yv[i],
                        xv[ip1],
                        yv[ip1],
                        x0,
                        y0,
                    )
                    if hit is not None:
                        xcr_i, ycr_i, t = hit
                        if t < best_t:
                            best_t, xcr, ycr = t, xcr_i, ycr_i
                xz, yz = xcr, ycr
        face_x[f], face_y[f] = xz, yz
    return face_x, face_y


def _dcosphi_sph_vec(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    x3: np.ndarray,
    y3: np.ndarray,
    x4: np.ndarray,
    y4: np.ndarray,
) -> np.ndarray:
    """Vectorized: all args (n_edges,) -> (n_edges,) |cos(phi)|."""
    p1 = _sphertocart3d_vec(x1, y1)
    p2 = _sphertocart3d_vec(x2, y2)
    p3 = _sphertocart3d_vec(x3, y3)
    p4 = _sphertocart3d_vec(x4, y4)
    d1 = p2 - p1
    d2 = p4 - p3
    r1 = np.sqrt(np.sum(d1 * d1, axis=1))
    r2 = np.sqrt(np.sum(d2 * d2, axis=1))
    dot = np.sum(d1 * d2, axis=1)
    cosphi = np.where(
        (r1 > 0) & (r2 > 0),
        dot / (r1 * r2),
        0.0,
    )
    cosphi = np.clip(cosphi, -1.0, 1.0)
    return np.abs(cosphi)


def _opposite_sides_vec(
    xk3: np.ndarray,
    yk3: np.ndarray,
    xk4: np.ndarray,
    yk4: np.ndarray,
    xc1: np.ndarray,
    yc1: np.ndarray,
    xc2: np.ndarray,
    yc2: np.ndarray,
) -> np.ndarray:
    """Vectorized version of _opposite_sides. Returns bool (n_edges,)."""
    ex = _getdx_vec(xk3, yk3, xk4, yk4, 1)
    ey = _getdy_vec(xk3, yk3, xk4, yk4, 1)
    c1 = ex * _getdy_vec(xk3, yk3, xc1, yc1, 1) - ey * _getdx_vec(
        xk3, yk3, xc1, yc1, 1
    )
    c2 = ex * _getdy_vec(xk3, yk3, xc2, yc2, 1) - ey * _getdx_vec(
        xk3, yk3, xc2, yc2, 1
    )
    return c1 * c2 < 0.0


def _edge_faces_from_faces_edges(
    face_nodes: np.ndarray, edge_nodes: np.ndarray
) -> np.ndarray:
    """Rebuild edge_faces (1-based) from faces/edges connectivity."""
    n_edges = edge_nodes.shape[0]
    n_faces = face_nodes.shape[0]
    if n_faces == 0 or n_edges == 0:
        return np.zeros((n_edges, 2), dtype=np.int64)
    max_node = int(face_nodes.max())
    node_to_faces: List[List[int]] = [[] for _ in range(max_node + 1)]
    for f in range(n_faces):
        for n in face_nodes[f, :]:
            if n > 0:
                node_to_faces[int(n)].append(f)
    edge_faces = np.zeros((n_edges, 2), dtype=np.int64)
    for e in range(n_edges):
        n1, n2 = edge_nodes[e, :]
        if n1 <= 0 or n2 <= 0:
            continue
        common = list(set(node_to_faces[int(n1)]) & set(node_to_faces[int(n2)]))
        if len(common) >= 1:
            edge_faces[e, 0] = common[0] + 1
        if len(common) >= 2:
            edge_faces[e, 1] = common[1] + 1
    return edge_faces


# ---------------------------------------------------------------------------
# Index conversion: UGRID/orthogonality use 1-based; we use 0-based internally.
# ---------------------------------------------------------------------------

def _to_0b(arr: np.ndarray) -> np.ndarray:
    """Convert 1-based indices to 0-based (valid: 0..n-1, invalid: -1)."""
    out = np.where(arr > 0, arr - 1, -1)
    return out.astype(arr.dtype)


def _to_1b(arr: np.ndarray) -> np.ndarray:
    """Convert 0-based indices to 1-based for meshkernel_orthogonality (invalid -1 -> 0)."""
    out = np.where(arr >= 0, arr + 1, 0)
    return out.astype(arr.dtype)


# ---------------------------------------------------------------------------
# Orthogonality: |cosphi| on in-memory arrays (0-based in, cosphi_abs out)
# ---------------------------------------------------------------------------

def compute_cosphi_abs_from_arrays(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    edge_nodes: np.ndarray,
    edge_faces: Optional[np.ndarray],
    use_file_centers: bool = False,
    use_circumcenter_3d: bool = True,
    edge_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute |cosphi| for edges. Inputs 0-based (invalid = -1).
    If edge_indices is provided (0-based), only compute for those edges (and only
    the face centers needed). Returns (edge_nodes, edge_faces, cosphi_abs).
    """
    if use_file_centers:
        raise ValueError(
            "use_file_centers=True is not supported in this in-memory variant."
        )
    # Convert to 1-based for orthogonality helpers (unchanged formulas)
    face_nodes = _to_1b(face_nodes)
    edge_nodes = _to_1b(edge_nodes)
    if edge_faces is not None:
        edge_faces = _to_1b(edge_faces)

    if edge_faces is None:
        edge_faces = _edge_faces_from_faces_edges(face_nodes, edge_nodes)

    n_faces = face_nodes.shape[0]
    n_edges = edge_nodes.shape[0]

    # When edge_indices provided, only compute face centers for faces adjacent to those edges
    face_mask: Optional[np.ndarray] = None
    if edge_indices is not None:
        edge_indices = np.asarray(edge_indices, dtype=np.int64).ravel()
        face_mask = np.zeros(n_faces, dtype=bool)
        for e in edge_indices:
            if e < 0 or e >= n_edges:
                continue
            for j in (0, 1):
                f1b = edge_faces[e, j]
                if f1b > 0:
                    face_mask[int(f1b) - 1] = True

    if use_circumcenter_3d:
        face_x, face_y = _face_centers_circumcenter3d(
            node_x, node_y, face_nodes, face_mask=face_mask
        )
    else:
        face_x, face_y = _face_centers(
            node_x, node_y, face_nodes, face_mask=face_mask
        )

    cosphi_abs = np.full(n_edges, np.nan, dtype=np.float64)

    k3 = edge_nodes[:, 0]
    k4 = edge_nodes[:, 1]
    f1 = edge_faces[:, 0]
    f2 = edge_faces[:, 1]
    valid = (k3 > 0) & (k4 > 0) & (f1 > 0) & (f2 > 0) & (f1 != f2)
    idx = np.where(valid)[0]
    if edge_indices is not None:
        idx = np.intersect1d(idx, edge_indices, assume_unique=True)
    if idx.size == 0:
        return edge_nodes, edge_faces, cosphi_abs

    k3i = k3[idx] - 1
    k4i = k4[idx] - 1
    f1i = f1[idx] - 1
    f2i = f2[idx] - 1

    if not use_circumcenter_3d:
        opp = _opposite_sides_vec(
            node_x[k3i],
            node_y[k3i],
            node_x[k4i],
            node_y[k4i],
            face_x[f1i],
            face_y[f1i],
            face_x[f2i],
            face_y[f2i],
        )
        idx = idx[opp]
        k3i, k4i, f1i, f2i = k3i[opp], k4i[opp], f1i[opp], f2i[opp]
        if idx.size == 0:
            return edge_nodes, edge_faces, cosphi_abs

    dx_edge = _getdx_vec(node_x[k3i], node_y[k3i], node_x[k4i], node_y[k4i], 1)
    dy_edge = _getdy_vec(node_x[k3i], node_y[k3i], node_x[k4i], node_y[k4i], 1)
    d = np.hypot(dx_edge, dy_edge)
    valid_d = d >= 1.0e-6
    idx = idx[valid_d]
    k3i, k4i, f1i, f2i = k3i[valid_d], k4i[valid_d], f1i[valid_d], f2i[valid_d]
    if idx.size == 0:
        return edge_nodes, edge_faces, cosphi_abs

    cosphi_abs[idx] = _dcosphi_sph_vec(
        face_x[f1i],
        face_y[f1i],
        face_x[f2i],
        face_y[f2i],
        node_x[k3i],
        node_y[k3i],
        node_x[k4i],
        node_y[k4i],
    )
    return edge_nodes, edge_faces, cosphi_abs


# ---------------------------------------------------------------------------
# Utility structures for zones
# ---------------------------------------------------------------------------


@dataclass
class MeshData:
    node_x: np.ndarray
    node_y: np.ndarray
    face_nodes: np.ndarray
    edge_nodes: np.ndarray
    edge_faces: np.ndarray


def _build_node_to_faces(face_nodes: np.ndarray) -> List[List[int]]:
    """Node -> faces connectivity table (0-based indices)."""
    if face_nodes.size == 0:
        return []
    max_node = max(0, int(face_nodes.max()))
    mapping: List[List[int]] = [[] for _ in range(max_node + 1)]
    n_faces = face_nodes.shape[0]
    for f in range(n_faces):
        for n in face_nodes[f, :]:
            if n >= 0:
                mapping[int(n)].append(f)
    return mapping


def _classify_zone_nodes(
    face_nodes: np.ndarray,
    edge_nodes: np.ndarray,
    faces_zone: Set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify internal vs boundary nodes for a given zone.

    - Internal node: belongs only to faces in the zone AND has no edge
      to a node outside the zone.
    - Boundary node: any zone node that is not internal.
    """
    zone_nodes = zone_nodes_from_faces(face_nodes, faces_zone)
    if zone_nodes.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    zone_nodes_set: Set[int] = set(int(n) for n in zone_nodes.tolist())

    node_to_faces = _build_node_to_faces(face_nodes)

    # Build node -> edge indices mapping
    max_node = int(max(zone_nodes.max(), edge_nodes.max()))
    node_to_edges: List[List[int]] = [[] for _ in range(max_node + 1)]
    for e in range(edge_nodes.shape[0]):
        n1, n2 = edge_nodes[e, :]
        if n1 >= 0:
            node_to_edges[int(n1)].append(e)
        if n2 >= 0:
            node_to_edges[int(n2)].append(e)

    internal_nodes: Set[int] = set()
    boundary_nodes: Set[int] = set()

    for nid in zone_nodes_set:
        faces_n = set(node_to_faces[nid])
        # if node appears in a face outside the zone -> boundary
        if not faces_n.issubset(faces_zone):
            boundary_nodes.add(nid)
            continue

        # if connected by an edge to a node outside the zone -> boundary
        is_boundary = False
        for eidx in node_to_edges[nid]:
            n1, n2 = edge_nodes[eidx, :]
            other = int(n2 if n1 == nid else n1)
            if other not in zone_nodes_set and other >= 0:
                is_boundary = True
                break

        if is_boundary:
            boundary_nodes.add(nid)
        else:
            internal_nodes.add(nid)

    # For safety, any zone node not internal is set as boundary
    for nid in zone_nodes_set:
        if nid not in internal_nodes and nid not in boundary_nodes:
            boundary_nodes.add(nid)

    internal = np.array(sorted(list(internal_nodes)), dtype=np.int64)
    boundary = np.array(sorted(list(boundary_nodes)), dtype=np.int64)
    return internal, boundary


# ---------------------------------------------------------------------------
# Zone graph (face adjacency, BFS, zone nodes)
# ---------------------------------------------------------------------------

def build_face_adjacency(edge_faces: np.ndarray, n_faces: int) -> List[List[int]]:
    """Adjacency graph (neighboring faces via a shared edge)."""
    neigh: List[Set[int]] = [set() for _ in range(n_faces)]
    for e in range(edge_faces.shape[0]):
        f1, f2 = edge_faces[e, :]  # already 0-based
        if f1 >= 0 and f2 >= 0 and f1 != f2:
            neigh[f1].add(f2)
            neigh[f2].add(f1)
    return [sorted(list(s)) for s in neigh]


def bfs_faces(
    start_faces: Iterable[int],
    neighbors: List[List[int]],
    max_depth: int,
) -> Set[int]:
    """Return the set of faces at topological distance <= max_depth."""
    visited: Set[int] = set()
    frontier: Set[int] = set(int(f) for f in start_faces)
    depth = 0
    while frontier and depth <= max_depth:
        visited.update(frontier)
        next_frontier: Set[int] = set()
        for f in frontier:
            for g in neighbors[f]:
                if g not in visited:
                    next_frontier.add(g)
        frontier = next_frontier
        depth += 1
    return visited


def zone_nodes_from_faces(face_nodes: np.ndarray, faces_zone: Set[int]) -> np.ndarray:
    """Nodes used by a subset of faces (0-based indices)."""
    if not faces_zone:
        return np.empty(0, dtype=np.int64)
    f_idx = np.fromiter(faces_zone, dtype=np.int64)
    nodes = face_nodes[f_idx, :].ravel()
    nodes = nodes[nodes >= 0]
    return np.unique(nodes.astype(np.int64))


def _triangles_from_face_nodes(face_nodes: np.ndarray) -> np.ndarray:
    """
    Build a triangle (EDGE) array from face_nodes (0-based; -1 = invalid).

    Each polygon face (n>=3) is triangulated in fan mode around the first node.
    Returns 0-based triangle indices as expected by adcirc2DFlowFM.
    """
    tris: List[Tuple[int, int, int]] = []
    n_faces = face_nodes.shape[0]
    for f in range(n_faces):
        row = face_nodes[f, :]
        nodes = row[row >= 0]
        if nodes.size < 3:
            continue
        n0 = int(nodes[0])
        for i in range(1, nodes.size - 1):
            n1 = int(nodes[i])
            n2 = int(nodes[i + 1])
            tris.append((n0, n1, n2))
    if tris:
        return np.asarray(tris, dtype=np.int64)
    return np.empty((0, 3), dtype=np.int64)


def apply_combined_ortho_smoother_to_zone(
    mesh: MeshData,
    faces_zone: Set[int],
    cosphi_abs: np.ndarray,
    cosphi_threshold: float,
    it: int,
    max_global_iter: int,
    n_inner: int = 2,
    mu_max: float = 0.4,
    relax: float = 0.2,
    small_edges_global: Optional[np.ndarray] = None,
    removesmalllinkstrsh: float = 0.1,
) -> Tuple[bool, bool]:
    """
    Combine simple (Laplacian) smoothing and orthogonality-oriented displacement,
    with a mu(it) factor increasing as in smood:

        Δx = (1 - mu) * Δx_smooth + mu * (Δx_ortho + beta * Δx_small)

    where Δx_ortho reduces |cosphi| and Δx_small pushes circumcenters apart on small-link edges.
    Returns (improved, zone_was_good) for stats; zone_was_good is False on early exit.
    """
    if not faces_zone:
        return (False, False)

    # Zone nodes (0-based indices)
    zone_nodes = zone_nodes_from_faces(mesh.face_nodes, faces_zone)
    if zone_nodes.size == 0:
        return (False, False)

    # Internal / boundary classification
    internal_global, boundary_global = _classify_zone_nodes(
        mesh.face_nodes, mesh.edge_nodes, faces_zone
    )
    internal_set: Set[int] = set(int(n) for n in internal_global.tolist())

    # Neighbors in the zone (for smooth term); [Improvement 4] also all neighbors (incl. out-of-zone) for boundary
    zone_set: Set[int] = set(int(n) for n in zone_nodes.tolist())
    neighbors: Dict[int, Set[int]] = {int(n): set() for n in zone_nodes.tolist()}
    all_neighbors: Dict[int, Set[int]] = {int(n): set() for n in zone_nodes.tolist()}
    edges_in_zone: List[int] = []
    ring_edges: List[int] = []
    for e in range(mesh.edge_nodes.shape[0]):
        n1, n2 = mesh.edge_nodes[e, :]
        if n1 < 0 or n2 < 0:
            continue
        g1 = int(n1)
        g2 = int(n2)
        if g1 in zone_set and g2 in zone_set:
            neighbors[g1].add(g2)
            neighbors[g2].add(g1)
            edges_in_zone.append(e)
        elif (g1 in zone_set) ^ (g2 in zone_set):
            # Edge crossing the zone boundary: part of the "crown" for acceptance tests
            ring_edges.append(e)
        if g1 in zone_set:
            all_neighbors[g1].add(g2)
        if g2 in zone_set:
            all_neighbors[g2].add(g1)
    # Out-of-zone neighbors for boundary nodes (for weighted Laplacian)
    neighbors_out: Dict[int, Set[int]] = {}
    for gid in zone_nodes:
        gid = int(gid)
        out_set = all_neighbors[gid] - zone_set
        if out_set:
            neighbors_out[gid] = out_set
    w_in, w_out = 1.0, 0.3

    # If no internal edge in the zone, nothing to do
    if not edges_in_zone:
        return (False, False)

    edges_in_zone_arr = np.array(edges_in_zone, dtype=int)
    # For acceptance/rollback tests, also include a "crown" of edges crossing the zone boundary
    if ring_edges:
        eval_edges_arr = np.array(edges_in_zone + ring_edges, dtype=int)
    else:
        eval_edges_arr = edges_in_zone_arr
    # Small-link edges in this zone (when global small list is provided)
    small_edges_set = set(small_edges_global.tolist()) if small_edges_global is not None and small_edges_global.size > 0 else set()
    small_in_zone = [e for e in edges_in_zone if e in small_edges_set]
    n_small_zone_before = 0
    if small_edges_global is not None and small_edges_global.size > 0:
        _, small_zone_list = compute_small_links_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
            edge_indices=edges_in_zone_arr,
        )
        n_small_zone_before = len(small_zone_list)

    # Local quality before displacement (max |cosphi| on zone edges)
    cos_zone_before = np.abs(cosphi_abs[edges_in_zone_arr])
    mask_before = np.isfinite(cos_zone_before)
    if not np.any(mask_before):
        return (False, False)
    max_cosphi_before = float(np.max(cos_zone_before[mask_before]))
    min_cosphi_before = float(np.min(cos_zone_before[mask_before]))

    # Build a distance-based weight in the zone: nodes near the worst edges get full ortho;
    # nodes near the buffer boundary get a reduced ortho weight to avoid degrading good areas.
    core_nodes: Set[int] = set()
    for e in edges_in_zone:
        if abs(cosphi_abs[e]) > cosphi_threshold:
            n1, n2 = mesh.edge_nodes[e, :]
            if n1 >= 0:
                core_nodes.add(int(n1))
            if n2 >= 0:
                core_nodes.add(int(n2))
    dist_weight: Dict[int, float] = {}
    if core_nodes:
        # BFS on zone graph (neighbors) starting from core_nodes
        INF = 10**9
        dist: Dict[int, int] = {int(n): INF for n in zone_nodes.tolist()}
        q: deque[int] = deque()
        for n in core_nodes:
            if n in dist:
                dist[n] = 0
                q.append(n)
        while q:
            u = q.popleft()
            du = dist[u]
            for v in neighbors.get(u, set()):
                if v in dist and dist[v] == INF:
                    dist[v] = du + 1
                    q.append(v)
        # Convert distances to weights in [0.2, 1.0]
        finite_d = [d for d in dist.values() if d < INF]
        if finite_d:
            max_d = max(finite_d)
            decay = max(1.0, max_d / 2.0)
            for n, d in dist.items():
                if d >= INF:
                    w = 0.5
                else:
                    w = 1.0 - float(d) / (decay + 1e-9)
                    w = max(0.2, min(1.0, w))
                dist_weight[n] = w
    else:
        # No clearly bad edges in this zone: uniform weight
        for n in zone_nodes.tolist():
            dist_weight[int(n)] = 1.0

    # When zone is already below cosphi threshold (e.g. only small-link), use smaller steps
    # and no Laplacian (scale_smooth=0) to avoid degrading orthogonality at zone boundaries [Improvement 1]
    zone_already_good = max_cosphi_before <= cosphi_threshold
    relax_zone = 0.12 if zone_already_good else relax
    scale_smooth = 0.0 if zone_already_good else 1.0
    # [Improvement 3] Very-good zones (max_cosphi already very low, only small-link): cap relax further
    cosphi_very_good = 0.05
    if zone_already_good and max_cosphi_before < cosphi_very_good and n_small_zone_before > 0:
        relax_zone = min(relax_zone, 0.08)

    # [Improvement 5] Skip [good] zones with no small links: nothing to do, avoid useless rollbacks
    if zone_already_good and n_small_zone_before == 0:
        if VERBOSE_ZONE_LOGS:
            print(
                f"    [ZONE] it={it} faces={len(faces_zone)} [good] skip (n_small_zone=0, nothing to do)"
            )
        return (False, True)

    # Snapshot of zone node coordinates (for possible rollback)
    zone_idx = zone_nodes.astype(np.int64)  # 0-based
    x_old = mesh.node_x[zone_idx].copy()
    y_old = mesh.node_y[zone_idx].copy()

    # Factor mu(it) increasing from 0 to mu_max
    if max_global_iter > 0:
        mu_it = mu_max * float(it + 1) / float(max_global_iter)
        mu_it = max(0.0, min(mu_max, mu_it))
    else:
        mu_it = mu_max

    alpha = 0.025  # Base amplitude of ortho displacement per edge (conservative to avoid overshoot)
    tag = "good" if zone_already_good else "bad"
    # Log start for this zone (relax_zone, scale_smooth help tune when many rollbacks)
    if VERBOSE_ZONE_LOGS:
        print(
            f"    [ZONE] it={it} faces={len(faces_zone)} [{tag}] "
            f"max|cosphi|_before={max_cosphi_before:.6f} min|cosphi|_before={min_cosphi_before:.6f} "
            f"relax_zone={relax_zone:.2f} scale_smooth={scale_smooth:.2f} mu={mu_it:.3f} n_small_zone={n_small_zone_before}"
        )

    for inner_i in range(max(1, int(n_inner))):
        # Smooth term (local Laplacian); [Improvement 4] boundary: include out-of-zone neighbors with weight w_out
        dx_s = np.zeros_like(mesh.node_x)
        dy_s = np.zeros_like(mesh.node_y)
        for g in zone_nodes:
            gid = int(g)
            if gid not in internal_set:
                continue
            neigh = neighbors.get(gid, set())
            if not neigh:
                continue
            out_neigh = neighbors_out.get(gid, set())
            if not out_neigh:
                idxs = np.array(list(neigh), dtype=np.int64)
                dx_s[gid] = mesh.node_x[idxs].mean() - mesh.node_x[gid]
                dy_s[gid] = mesh.node_y[idxs].mean() - mesh.node_y[gid]
            else:
                idxs_in = np.array(list(neigh), dtype=np.int64)
                idxs_out = np.array(list(out_neigh), dtype=np.int64)
                bary_in_x = mesh.node_x[idxs_in].mean()
                bary_in_y = mesh.node_y[idxs_in].mean()
                bary_out_x = mesh.node_x[idxs_out].mean()
                bary_out_y = mesh.node_y[idxs_out].mean()
                bary_x = (bary_in_x * w_in + bary_out_x * w_out) / (w_in + w_out)
                bary_y = (bary_in_y * w_in + bary_out_y * w_out) / (w_in + w_out)
                dx_s[gid] = bary_x - mesh.node_x[gid]
                dy_s[gid] = bary_y - mesh.node_y[gid]

        # Ortho term: small corrections on the worst edges in the zone.
        # Skip when zone is already below threshold (e.g. zone only for small-link) to save cost.
        dx_o = np.zeros_like(mesh.node_x)
        dy_o = np.zeros_like(mesh.node_y)
        if edges_in_zone and max_cosphi_before > cosphi_threshold:
            cos_vals = np.abs(cosphi_abs[edges_in_zone])
            bad_mask = cos_vals > cosphi_threshold
            if np.any(bad_mask):
                bad_idx = np.where(bad_mask)[0]
                # Limit number of edges to stay local
                sort_loc = np.argsort(cos_vals[bad_idx])[::-1]
                top_loc = bad_idx[sort_loc[: min(3, sort_loc.size)]]
                for li in top_loc:
                    e = edges_in_zone[li]
                    k3, k4 = mesh.edge_nodes[e, :]
                    if k3 < 0 or k4 < 0:
                        continue
                    g3 = int(k3)
                    g4 = int(k4)
                    move3 = g3 in internal_set
                    move4 = g4 in internal_set
                    if not (move3 or move4):
                        continue
                    x3, y3 = mesh.node_x[g3], mesh.node_y[g3]
                    x4, y4 = mesh.node_x[g4], mesh.node_y[g4]
                    ex = x4 - x3
                    ey = y4 - y3
                    norm_e = np.hypot(ex, ey)
                    if norm_e < 1.0e-8:
                        continue
                    # Candidate directions (perpendicular + opposite)
                    px = -ey / norm_e
                    py = ex / norm_e
                    dirs = [(px, py), (-px, -py)]
                    best_improve = 0.0
                    best_dx3 = best_dy3 = 0.0
                    best_dx4 = best_dy4 = 0.0
                    base_val = float(np.abs(cosphi_abs[e]))
                    if not np.isfinite(base_val):
                        continue
                    w_excess = base_val - cosphi_threshold
                    if w_excess <= 0.0:
                        continue
                    step = alpha * w_excess
                    for (ux, uy) in dirs:
                        # Trial displacement: local copy of coordinates
                        trial_x = mesh.node_x.copy()
                        trial_y = mesh.node_y.copy()
                        if move3:
                            trial_x[g3] = x3 - step * ux
                            trial_y[g3] = y3 - step * uy
                        if move4:
                            trial_x[g4] = x4 + step * ux
                            trial_y[g4] = y4 + step * uy
                        _, _, cosphi_trial = compute_cosphi_abs_from_arrays(
                            trial_x,
                            trial_y,
                            mesh.face_nodes,
                            mesh.edge_nodes,
                            mesh.edge_faces,
                            use_file_centers=False,
                            use_circumcenter_3d=True,
                            edge_indices=np.array([e]),
                        )
                        new_val = float(np.abs(cosphi_trial[e]))
                        if not np.isfinite(new_val):
                            continue
                        improve = base_val - new_val
                        if improve > best_improve and new_val < base_val:
                            best_improve = improve
                            if move3:
                                best_dx3 = -step * ux
                                best_dy3 = -step * uy
                            if move4:
                                best_dx4 = step * ux
                                best_dy4 = step * uy
                    if best_improve > 0.0:
                        if move3:
                            dx_o[g3] += best_dx3
                            dy_o[g3] += best_dy3
                        if move4:
                            dx_o[g4] += best_dx4
                            dy_o[g4] += best_dy4

        # Small-link term [V3]: pymesh2d-style — move opposite vertices along circumcenter separation
        dx_small = np.zeros_like(mesh.node_x)
        dy_small = np.zeros_like(mesh.node_y)
        alpha_small = 0.05
        beta_small = 0.5
        max_small_edges_per_zone = 6
        step_small = min(alpha_small * 0.12, 0.004)
        # [V3-safe] Much less aggressive: only in fairly good zones, and smaller steps
        # - Only apply when the zone is already reasonably orthogonal (max_cosphi_before <= 0.30)
        # - Softer aggressive_factor
        # - Clamp step_meters tightly to avoid overshoot and cosphi blow-up
        if n_small_zone_before <= 1:
            aggressive_factor = 1.5
        elif n_small_zone_before <= 3:
            aggressive_factor = 1.2
        else:
            aggressive_factor = 1.0
        if (
            small_in_zone
            and zone_already_good
            and max_cosphi_before <= 0.30
            and (inner_i % 2 == 0)
        ):
            _, small_zone_current = compute_small_links_from_arrays(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                removesmalllinkstrsh=removesmalllinkstrsh,
                edge_indices=edges_in_zone_arr,
            )
            n_small_zone_current = len(small_zone_current)
            nface = mesh.face_nodes.shape[0]
            face_mask_zone = np.zeros(nface, dtype=bool)
            for fid in faces_zone:
                face_mask_zone[int(fid)] = True
            vert_deg = np.column_stack([mesh.node_x, mesh.node_y])
            circum_ll = _circumcenters_lonlat_ugrid(
                vert_deg, mesh.face_nodes, mesh.edge_faces, face_mask=face_mask_zone
            )
            vert_xy, _, _ = _lonlat_to_local_xy(mesh.node_x, mesh.node_y)
            tria = mesh.face_nodes[:, :3]
            valid_t = (tria[:, 0] >= 0) & (tria[:, 1] >= 0) & (tria[:, 2] >= 0)
            ba = np.zeros(nface, dtype=np.float64)
            ba[valid_t] = np.abs(_triarea_2d(vert_xy, tria[valid_t]))
            for li, e in enumerate(small_in_zone[:max_small_edges_per_zone]):
                f1, f2 = mesh.edge_faces[e, 0], mesh.edge_faces[e, 1]
                k3, k4 = mesh.edge_nodes[e, 0], mesh.edge_nodes[e, 1]
                if k3 < 0 or k4 < 0 or f1 < 0 or f2 < 0 or f1 >= nface or f2 >= nface:
                    continue
                f1, f2, k3, k4 = int(f1), int(f2), int(k3), int(k4)
                tri1 = mesh.face_nodes[f1, :3]
                tri2 = mesh.face_nodes[f2, :3]
                opp1 = next((int(v) for v in tri1 if v >= 0 and v != k3 and v != k4), None)
                opp2 = next((int(v) for v in tri2 if v >= 0 and v != k3 and v != k4), None)
                if opp1 is None or opp2 is None or opp1 == opp2:
                    continue
                move_opp1 = opp1 in internal_set
                move_opp2 = opp2 in internal_set
                if not (move_opp1 or move_opp2):
                    continue
                cc1, cc2 = circum_ll[f1], circum_ll[f2]
                if np.any(np.isnan(cc1)) or np.any(np.isnan(cc2)):
                    continue
                dx_m = _getdx(cc1[0], cc1[1], cc2[0], cc2[1], 1)
                dy_m = _getdy(cc1[0], cc1[1], cc2[0], cc2[1], 1)
                dxlink = np.sqrt(dx_m * dx_m + dy_m * dy_m)
                if dxlink < 1e-12:
                    continue
                sqrt_ba1 = np.sqrt(max(ba[f1], 1e-20))
                sqrt_ba2 = np.sqrt(max(ba[f2], 1e-20))
                dxlim = 0.9 * removesmalllinkstrsh * 0.5 * (sqrt_ba1 + sqrt_ba2)
                if dxlink >= dxlim:
                    continue
                needed_distance = (dxlim - dxlink) * aggressive_factor
                cc_diff_deg = np.array([cc2[0] - cc1[0], cc2[1] - cc1[1]], dtype=np.float64)
                max_step_meters = min(np.sqrt(max(ba[f1], 1e-20)), np.sqrt(max(ba[f2], 1e-20))) * 0.5
                # Conservative step: at most 25% of needed distance, and at most half the current link length
                step_meters = min(needed_distance * 0.25, max_step_meters, 0.5 * dxlink)
                if step_meters < 1e-12:
                    continue
                disp_deg = cc_diff_deg * (step_meters / dxlink)
                best_n_small = n_small_zone_current
                best_dopp1 = np.zeros(2)
                best_dopp2 = np.zeros(2)
                for sign in (1, -1):
                    trial_x = mesh.node_x.copy()
                    trial_y = mesh.node_y.copy()
                    d1 = (-sign * 0.5 * disp_deg) if move_opp1 else np.zeros(2)
                    d2 = (sign * 0.5 * disp_deg) if move_opp2 else np.zeros(2)
                    if move_opp1:
                        trial_x[opp1] = mesh.node_x[opp1] + d1[0]
                        trial_y[opp1] = mesh.node_y[opp1] + d1[1]
                    if move_opp2:
                        trial_x[opp2] = mesh.node_x[opp2] + d2[0]
                        trial_y[opp2] = mesh.node_y[opp2] + d2[1]
                    _, small_trial = compute_small_links_from_arrays(
                        trial_x, trial_y,
                        mesh.face_nodes, mesh.edge_nodes, mesh.edge_faces,
                        removesmalllinkstrsh=removesmalllinkstrsh,
                        edge_indices=edges_in_zone_arr,
                    )
                    if len(small_trial) < best_n_small:
                        best_n_small = len(small_trial)
                        best_dopp1 = d1
                        best_dopp2 = d2
                if best_n_small < n_small_zone_current:
                    if move_opp1:
                        dx_small[opp1] += best_dopp1[0]
                        dy_small[opp1] += best_dopp1[1]
                    if move_opp2:
                        dx_small[opp2] += best_dopp2[0]
                        dy_small[opp2] += best_dopp2[1]

        # Combine and update: smooth + ortho + small_link (scale_smooth dampens Laplacian for "good" zones)
        new_x = mesh.node_x.copy()
        new_y = mesh.node_y.copy()
        for g in zone_nodes:
            gid = int(g)
            if gid not in internal_set:
                continue
            w_node = dist_weight.get(gid, 1.0)
            dx_o_and_small = w_node * (dx_o[gid] + beta_small * dx_small[gid])
            dy_o_and_small = w_node * (dy_o[gid] + beta_small * dy_small[gid])
            dx = (1.0 - mu_it) * scale_smooth * dx_s[gid] + mu_it * dx_o_and_small
            dy = (1.0 - mu_it) * scale_smooth * dy_s[gid] + mu_it * dy_o_and_small
            new_x[gid] += relax_zone * dx
            new_y[gid] += relax_zone * dy
        mesh.node_x[:] = new_x
        mesh.node_y[:] = new_y

    # [Improvement 2] Line search: try factors 1.0, 0.5, 0.25 on total displacement to avoid rollback
    # [Improvement 5] For [good] zones: accept "no degradation" (max_ca <= threshold, n_small_za <= n_small_before)
    delta_x = mesh.node_x[zone_idx].copy() - x_old
    delta_y = mesh.node_y[zone_idx].copy() - y_old
    mesh.node_x[zone_idx] = x_old
    mesh.node_y[zone_idx] = y_old
    improved = False
    best_factor = 0.0
    max_cosphi_after = float("inf")
    min_cosphi_after = 0.0
    n_small_zone_after = n_small_zone_before
    for factor in (1.0, 0.5, 0.25):
        mesh.node_x[zone_idx] = x_old + factor * delta_x
        mesh.node_y[zone_idx] = y_old + factor * delta_y
        _, _, cosphi_after = compute_cosphi_abs_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            use_file_centers=False,
            use_circumcenter_3d=True,
            edge_indices=eval_edges_arr,
        )
        cos_zone_after = np.abs(cosphi_after[eval_edges_arr])
        mask_after = np.isfinite(cos_zone_after)
        if not np.any(mask_after):
            continue
        max_ca = float(np.max(cos_zone_after[mask_after]))
        min_ca = float(np.min(cos_zone_after[mask_after]))
        n_small_za = 0
        if small_edges_global is not None and small_edges_global.size > 0:
            _, small_zone_after_list = compute_small_links_from_arrays(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                removesmalllinkstrsh=removesmalllinkstrsh,
                edge_indices=edges_in_zone_arr,
            )
            n_small_za = len(small_zone_after_list)
        is_strict_improved = (
            max_ca < max_cosphi_before
            or (
                small_edges_global is not None
                and small_edges_global.size > 0
                and n_small_za < n_small_zone_before
                and max_ca <= cosphi_threshold
            )
        )
        if zone_already_good:
            # For [good]: accept only if we don't degrade orthogonality near the zone boundary too much.
            # This prevents "small-link only" actions from quietly increasing max|cosphi| in the crown.
            ortho_slack = 0.02
            ortho_cap = min(cosphi_threshold, max_cosphi_before + ortho_slack)
            is_acceptable = (max_ca <= ortho_cap) and (n_small_za <= n_small_zone_before)
            if is_acceptable and not improved:
                improved = True
                best_factor = factor
                max_cosphi_after = max_ca
                min_cosphi_after = min_ca
                n_small_zone_after = n_small_za
            elif is_acceptable and improved:
                # Prefer smaller max_ca, then smaller n_small_za, then larger factor
                if max_ca < max_cosphi_after or (
                    max_ca == max_cosphi_after
                    and (n_small_za < n_small_zone_after or (n_small_za == n_small_zone_after and factor > best_factor))
                ):
                    best_factor = factor
                    max_cosphi_after = max_ca
                    min_cosphi_after = min_ca
                    n_small_zone_after = n_small_za
        else:
            # [bad] zone: first strict improvement wins
            if is_strict_improved:
                improved = True
                best_factor = factor
                max_cosphi_after = max_ca
                min_cosphi_after = min_ca
                n_small_zone_after = n_small_za
                break
    if improved and zone_already_good:
        # Apply best factor (we may have tried several)
        mesh.node_x[zone_idx] = x_old + best_factor * delta_x
        mesh.node_y[zone_idx] = y_old + best_factor * delta_y
    elif improved and not zone_already_good:
        mesh.node_x[zone_idx] = x_old + best_factor * delta_x
        mesh.node_y[zone_idx] = y_old + best_factor * delta_y
    if not improved:
        mesh.node_x[zone_idx] = x_old
        mesh.node_y[zone_idx] = y_old
        if VERBOSE_ZONE_LOGS:
            print(
                f"    [ZONE] rollback [{tag}]: max_before={max_cosphi_before:.6f} "
                f"(line_search: all factors 1.0, 0.5, 0.25 failed)"
            )
    else:
        log_small = ""
        if small_edges_global is not None and small_edges_global.size > 0:
            log_small = f" n_small_zone={n_small_zone_before}->{n_small_zone_after}"
        if VERBOSE_ZONE_LOGS:
            print(
                f"    [ZONE] accept [{tag}]: max_before={max_cosphi_before:.6f} "
                f"max_after={max_cosphi_after:.6f} "
                f"min_after={min_cosphi_after:.6f} factor={best_factor:.2f}{log_small}"
            )
    return (improved, zone_already_good)


# ---------------------------------------------------------------------------
# Main orthogonalization loop by zones
# ---------------------------------------------------------------------------


def _log_ortho_parameters(
    cosphi_threshold: float,
    removesmalllinkstrsh: float,
    buffer_layers: int,
    max_global_iter: int,
    smooth_iter: float,
) -> None:
    """Print tunable parameters once for easy re-tuning; zone defaults are in apply_combined_ortho_smoother_to_zone."""
    print("[ORTHO] Parameters (tune in code or extend CLI):")
    print(
        f"  global: cosphi_threshold={cosphi_threshold} removesmalllinkstrsh={removesmalllinkstrsh} "
        f"buffer_layers={buffer_layers} max_global_iter={max_global_iter} smooth_iter={smooth_iter}"
    )
    print(
        "  zone: relax=0.2 mu_max=0.4 n_inner=smooth_iter  "
        "| when max_cosphi<=threshold: relax_zone=0.12 scale_smooth=0 (v2)"
    )
    print("  line_search=1.0,0.5,0.25 | [good] accept no-degradation (max_ca<=thr, n_small not increase)")
    print("  Laplacian out-of-zone weight=0.3")
    print(
        "  zone ortho: alpha=0.025 top_loc_edges=3  "
        "| zone small_link: step_small=0.004 beta_small=0.5 max_small_edges_per_zone=4"
    )
    print("  [good] = zone with max|cosphi|<=threshold (small-link only); [bad] = zone with cosphi>threshold")


def orthogonalize_netcdf(
    input_path: str,
    output_path: Optional[str] = None,
    cosphi_threshold: float = 0.49,
    removesmalllinkstrsh: float = 0.1,
    buffer_layers: int = 2,
    max_global_iter: int = 10,
    smooth_iter: int = 4,
    merge_small_links: bool = False,
) -> float:
    """
    Orthogonalize a *_net.nc* file by zones until max(|cosphi|) < `cosphi_threshold`
    (or iterations are exhausted).

    Parameters
    ----------
    input_path : str
        Input UGRID NetCDF file (mesh unmodified).
    output_path : str, optional
        Output NetCDF file. If None, suffix `_ortho` is added before extension.
    cosphi_threshold : float
        Maximum acceptable threshold for |cosphi| (default 0.49).
    removesmalllinkstrsh : float
        Small flow links threshold for circumcenter criterion (default 0.1).
    buffer_layers : int
        Topological radius (number of cell layers) around each problematic
        element (2 or 3 recommended).
    max_global_iter : int
        Maximum number of global iterations (recomputes cosphi after each
        pass over all zones).
    smooth_iter : int
        Number of local elliptic smoothing `smooth_delft` iterations per zone.
    merge_small_links : bool
        If True and small links remain after ortho, run pymesh2d merge_circumcenters
        on the output file to merge triangle pairs into quads (default False).

    Returns
    -------
    max_cosphi : float
        Final value of max(|cosphi|) on internal edges.
    """
    if output_path is None:
        if input_path.endswith(".nc"):
            output_path = input_path[:-3] + "_ortho.nc"
        else:
            output_path = input_path + "_ortho.nc"

    # Load UGRID (file is 1-based); convert to 0-based for internal use
    print("[ORTHO] Loading mesh...")
    node_x, node_y, fn_1b, en_1b, ef_1b, fx, fy = _load_ugrid(input_path)
    if ef_1b is None:
        ef_1b = _edge_faces_from_faces_edges(fn_1b, en_1b)
    face_nodes = _to_0b(fn_1b)
    edge_nodes = _to_0b(en_1b)
    edge_faces = _to_0b(ef_1b)

    mesh = MeshData(
        node_x=node_x.copy(),
        node_y=node_y.copy(),
        face_nodes=face_nodes,
        edge_nodes=edge_nodes,
        edge_faces=edge_faces,
    )

    n_faces = face_nodes.shape[0]
    face_neighbors = build_face_adjacency(mesh.edge_faces, n_faces)

    print("[ORTHO] Computing initial cosphi and small links...")
    _, _, cosphi_abs0 = compute_cosphi_abs_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        use_file_centers=False,
        use_circumcenter_3d=True,
    )
    n_small0, _ = compute_small_links_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        removesmalllinkstrsh=removesmalllinkstrsh,
    )
    mask0 = ~np.isnan(cosphi_abs0)
    if np.any(mask0):
        max0 = float(np.nanmax(cosphi_abs0[mask0]))
        nb_bad0 = int(np.count_nonzero(cosphi_abs0[mask0] > cosphi_threshold))
        print(
            f"[ORTHO] Initial state: max |cosphi| = {max0:.6f} "
            f"(threshold={cosphi_threshold:.3f}, edges > threshold = {nb_bad0}), "
            f"n_small_flow_links = {n_small0}"
        )
    else:
        print(
            f"[ORTHO] No valid internal edge found in initial state. "
            f"n_small_flow_links = {n_small0}"
        )

    # Log all tunable parameters once for easy re-tuning (edit defaults in code or add CLI later)
    _log_ortho_parameters(
        cosphi_threshold=cosphi_threshold,
        removesmalllinkstrsh=removesmalllinkstrsh,
        buffer_layers=buffer_layers,
        max_global_iter=max_global_iter,
        smooth_iter=smooth_iter,
    )

    # Single loop: process bad cosphi + small-link edges together (two-phase caused phase2 never reached)
    for it in range(max_global_iter):
        _, _, cosphi_abs = compute_cosphi_abs_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            use_file_centers=False,
            use_circumcenter_3d=True,
        )

        mask = ~np.isnan(cosphi_abs)
        if not np.any(mask):
            print(f"[ORTHO] Iter {it}: no valid internal edge, stop.")
            break

        max_cosphi = float(np.nanmax(cosphi_abs[mask]))
        nb_bad = int(np.count_nonzero(cosphi_abs[mask] > cosphi_threshold))
        n_small, small_edges_arr = compute_small_links_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
        )
        # [V3] Try to fix small links by edge flip (convex quad) before zone smoothing.
        # Guardrail: keep flips only if they don't worsen max|cosphi| too much.
        if n_small > 0:
            max_cosphi_before_flip = max_cosphi
            face_nodes_before_flip = mesh.face_nodes.copy()
            n_flipped = try_flip_small_flow_edges_ugrid(mesh, small_edges_arr, removesmalllinkstrsh)
            if n_flipped > 0:
                _, _, cosphi_abs_tmp = compute_cosphi_abs_from_arrays(
                    mesh.node_x,
                    mesh.node_y,
                    mesh.face_nodes,
                    mesh.edge_nodes,
                    mesh.edge_faces,
                    use_file_centers=False,
                    use_circumcenter_3d=True,
                )
                mask_tmp = ~np.isnan(cosphi_abs_tmp)
                max_cosphi_after_flip = float(np.nanmax(cosphi_abs_tmp[mask_tmp])) if np.any(mask_tmp) else max_cosphi_before_flip
                if max_cosphi_after_flip > max_cosphi_before_flip + 0.02:
                    # revert flips
                    mesh.face_nodes[:] = face_nodes_before_flip
                    print(
                        f"[ORTHO] Iter {it}: edge flips reverted "
                        f"(max|cosphi| {max_cosphi_before_flip:.6f} -> {max_cosphi_after_flip:.6f})"
                    )
                else:
                    cosphi_abs = cosphi_abs_tmp
                    mask = mask_tmp
                    max_cosphi = max_cosphi_after_flip
                    nb_bad = int(np.count_nonzero(cosphi_abs[mask] > cosphi_threshold)) if np.any(mask) else nb_bad
                    n_small, small_edges_arr = compute_small_links_from_arrays(
                        mesh.node_x,
                        mesh.node_y,
                        mesh.face_nodes,
                        mesh.edge_nodes,
                        mesh.edge_faces,
                        removesmalllinkstrsh=removesmalllinkstrsh,
                    )
                    print(f"[ORTHO] Iter {it}: edge flips = {n_flipped}, n_small_flow_links = {n_small}")

        print(
            f"[ORTHO] Iter {it}: max |cosphi| = {max_cosphi:.6f}, "
            f"edges > threshold = {nb_bad}, n_small_flow_links = {n_small}"
        )

        if max_cosphi <= cosphi_threshold and n_small == 0:
            print(
                f"[ORTHO] Criterion reached (max |cosphi| <= {cosphi_threshold:.3f}, "
                f"n_small_flow_links = 0) after {it} global iterations."
            )
            break

        bad_edges = np.where(
            (mask) & (cosphi_abs > cosphi_threshold)
        )[0]
        bad_set = set(bad_edges.tolist())
        sort_idx = np.argsort(cosphi_abs[bad_edges])[::-1] if bad_edges.size > 0 else np.array([], dtype=np.int64)
        bad_edges_sorted = bad_edges[sort_idx] if bad_edges.size > 0 else np.array([], dtype=np.int64)
        small_only = np.array(
            [e for e in small_edges_arr.tolist() if e not in bad_set],
            dtype=np.int64,
        )
        problematic_edges = np.concatenate([bad_edges_sorted, small_only]) if bad_edges_sorted.size > 0 else small_only

        if problematic_edges.size == 0:
            print(f"[ORTHO] Iter {it}: no problematic edge (cosphi or small link), stop.")
            break
        visited_faces_global: Set[int] = set()
        improved_zones = 0
        total_zones = 0
        accept_good = 0
        accept_bad = 0
        rollback_good = 0
        rollback_bad = 0

        for e in problematic_edges:
            f1, f2 = mesh.edge_faces[e, :]
            start_faces: List[int] = []
            if f1 >= 0:
                start_faces.append(int(f1))
            if f2 >= 0:
                start_faces.append(int(f2))
            if not start_faces:
                continue

            # Slightly enlarge zones for truly bad edges to avoid sharp transitions at the buffer boundary
            this_buffer = buffer_layers
            if e in bad_set:
                this_buffer = buffer_layers + 1
            faces_zone = bfs_faces(start_faces, face_neighbors, this_buffer)
            if faces_zone.issubset(visited_faces_global):
                continue
            total_zones += 1
            improved, zone_was_good = apply_combined_ortho_smoother_to_zone(
                mesh=mesh,
                faces_zone=faces_zone,
                cosphi_abs=cosphi_abs,
                cosphi_threshold=cosphi_threshold,
                it=it,
                max_global_iter=max_global_iter,
                n_inner=max(1, int(smooth_iter)),
                small_edges_global=small_edges_arr,
                removesmalllinkstrsh=removesmalllinkstrsh,
            )
            if improved:
                improved_zones += 1
                if zone_was_good:
                    accept_good += 1
                else:
                    accept_bad += 1
            else:
                if zone_was_good:
                    rollback_good += 1
                else:
                    rollback_bad += 1
            visited_faces_global.update(faces_zone)

        if total_zones > 0:
            frac_improved = improved_zones / float(total_zones)
        else:
            frac_improved = 0.0
        print(
            f"[ORTHO] Iter {it}: zones = {total_zones} "
            f"accept = {improved_zones} ({frac_improved:.2%}) [good={accept_good} bad={accept_bad}] "
            f"rollback = {rollback_good + rollback_bad} [good={rollback_good} bad={rollback_bad}] "
            f"distinct_faces = {len(visited_faces_global)}"
        )

    _, _, cosphi_abs_final = compute_cosphi_abs_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        use_file_centers=False,
        use_circumcenter_3d=True,
    )
    mask_final = ~np.isnan(cosphi_abs_final)
    if np.any(mask_final):
        max_cosphi_final = float(np.nanmax(cosphi_abs_final[mask_final]))
    else:
        max_cosphi_final = float("nan")
    n_small_final, _ = compute_small_links_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        removesmalllinkstrsh=removesmalllinkstrsh,
    )
    print(
        f"[ORTHO] Final small flow links (circumcenters too close) = "
        f"{n_small_final} (threshold={removesmalllinkstrsh})"
    )

    # -------------------------
    # Full reconstruction via adcirc2DFlowFM
    # -------------------------
    # NODE: (n_nodes, 3) -> x, y, z. Re-read z from input NetCDF if present.
    try:
        with Dataset(input_path, "r") as src:
            if "mesh2d_node_z" in src.variables:
                node_z = np.asarray(src["mesh2d_node_z"][:], dtype=np.float64).ravel()
            else:
                node_z = np.zeros_like(mesh.node_x, dtype=np.float64)
    except Exception:
        node_z = np.zeros_like(mesh.node_x, dtype=np.float64)

    NODE = np.column_stack(
        [
            mesh.node_x.astype(np.float64),
            mesh.node_y.astype(np.float64),
            node_z,
        ]
    )

    # EDGE: (n_elements, 3) -> 0-based triangles derived from UGRID face_nodes
    EDGE = _triangles_from_face_nodes(mesh.face_nodes)

    print(
        f"[ORTHO] Building UGRID with adcirc2DFlowFM: "
        f"{NODE.shape[0]} nodes, {EDGE.shape[0]} triangles"
    )
    ds_out = adcirc2DFlowFM(NODE=NODE, EDGE=EDGE)
    ds_out.to_netcdf(output_path)

    # [V3] Optional: merge remaining small-link pairs into quads (requires pymesh2d + xarray)
    if merge_small_links and n_small_final > 0:
        try:
            import xarray as xr
            from ..geomesh_util.merge_circumcenters import merge_circumcenters as _merge_cc
            ds = xr.open_dataset(output_path)
            ds_m = _merge_cc(ds, removesmalllinkstrsh=removesmalllinkstrsh)
            ds_m.to_netcdf(output_path)
            ds.close()
            print(f"[ORTHO] Merged remaining {n_small_final} small-link pairs into quads (output has tri+quad faces).")
        except Exception as exc:
            print(f"[ORTHO] merge_circumcenters skipped ({exc}); output is ortho mesh only.")

    return max_cosphi_final


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Local orthogonalization of a UGRID *_net.nc* mesh by zones "
            "(buffer in number of cells) using the |cosphi| metric from "
            "meshkernel_orthogonality.py."
        )
    )
    p.add_argument("netcdf_path", help="Input *_net.nc* file")
    p.add_argument(
        "-o",
        "--output",
        help="Output *_net.nc* file (default: suffix _ortho.nc)",
        default=None,
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.49,
        help="Maximum acceptable threshold for |cosphi| (default 0.49)",
    )
    p.add_argument(
        "--buffer",
        type=int,
        default=2,
        help="Topological radius in cells around a bad edge (2 or 3 recommended)",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=8,
        help="Maximum number of global orthogonalization iterations",
    )
    p.add_argument(
        "--smooth-iter",
        type=int,
        default=16,
        help="Number of local elliptic smoothing iterations per zone",
    )
    p.add_argument(
        "--merge-small-links",
        action="store_true",
        help="After ortho, merge remaining small-link triangle pairs into quads (pymesh2d)",
    )

    args = p.parse_args()

    max_cosphi = orthogonalize_netcdf(
        input_path=args.netcdf_path,
        output_path=args.output,
        cosphi_threshold=args.threshold,
        removesmalllinkstrsh=0.1,
        buffer_layers=args.buffer,
        max_global_iter=args.max_iter,
        smooth_iter=args.smooth_iter,
        merge_small_links=args.merge_small_links,
    )
    print(f"Orthogonalization done. Final max |cosphi| = {max_cosphi:.6f}")


if __name__ == "__main__":
    main()

