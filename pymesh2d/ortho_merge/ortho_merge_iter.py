"""
Iterative pipeline: orthogonalize <-> merge_circumcenters (with optional recovery).

Part of the ``pymesh2d`` package (no dependency on scripts outside this tree).
Used by :mod:`pymesh2d.smood` when ``use_backup_transition`` is True.

Intended process (Delft3D-FM / dual mesh)
-----------------------------------------
1. **Orthogonalize** on a triangulation of the current (mixed) faces **consistent with**
   ``merge_circumcenters`` / UGRID export: quads ``[a,v1,b,v2]`` are split on diagonal
   ``(v1,v2)`` (see ``geomesh_util.grd_util.triangulate_mixed_face_row_to_tris``), **not**
   fan-from-``a`` (diagonal ``(a,b)``), which skews the dual w.r.t. Delft3D-FM.
2. **Remove short flow links**: ``merge_circumcenters`` merges triangle pairs into quads.
3. Repeat; optional **recovery** cycles if ``max|cosφ|`` or small-link count still fails
   the dual criteria (see ``require_both_criteria``).
4. With ``require_both_criteria=True``, **raise** if criteria are still not met after main +
   recovery. Default **False** matches ``backup_ortho_merge_20260319_174424`` (no global dual
   check / no recovery). Set **True** for a strict guarantee on the same triangle proxy (slower).

Implementation note (package layout): the underlying numeric orthogonality/small-link code
(``meshkernel_orthogonalize_3.py`` and ``meshkernel_orthogonalize_3_tria.py``) lives in the same
directory as this module to keep the package root minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class OrthoMergeStats:
    outer_iter: int
    max_cosphi: float
    n_small_flow_links: int
    merged_this_iter: int
    n_zones_orthogonalized: int
    recovery: bool = False
    recovery_iter: int = 0


def _faces_from_face_nodes(face_nodes: np.ndarray) -> List[np.ndarray]:
    """
    Convert internal UGRID `face_nodes` (0-based, invalid = -1) to a list of faces.

    The returned faces contain only valid node indices (>=0) and keep original polygon
    size (3 or 4+ depending on input).
    """
    face_nodes = np.asarray(face_nodes, dtype=np.int64)
    faces: List[np.ndarray] = []
    for f in range(face_nodes.shape[0]):
        nodes = face_nodes[f, :]
        nodes = nodes[nodes >= 0]
        if nodes.size >= 3:
            faces.append(nodes.astype(np.int64, copy=True))
    return faces


def _triangulate_faces_for_ortho(vert_xy: np.ndarray, faces: Sequence[np.ndarray]) -> np.ndarray:
    """
    Triangle rows for ``orthogonalize_tria_mesh``, aligned with ``merge_circumcenters`` quads
    and NetCDF export: use :func:`~pymesh2d.geomesh_util.grd_util.triangulate_mixed_face_row_to_tris`
    so 4-node faces use diagonal ``(v1,v2)``, not fan-from-``a``.
    """
    from ..geomesh_util.grd_util import triangulate_mixed_face_row_to_tris

    vert_xy = np.asarray(vert_xy, dtype=np.float64)
    if vert_xy.ndim != 2 or vert_xy.shape[1] < 2:
        raise ValueError("vert_xy must have shape (N, 2) or (N, >=2) for x,y")
    xy = vert_xy[:, :2]
    out: List[Tuple[int, int, int]] = []
    for nodes in faces:
        n = np.asarray(nodes, dtype=np.int64).reshape(-1)
        if n.size < 3:
            continue
        out.extend(triangulate_mixed_face_row_to_tris(xy, n))
    if out:
        return np.asarray(out, dtype=np.int64)
    return np.empty((0, 3), dtype=np.int64)


def _face_nodes_raw_to_0b(face_nodes_raw: np.ndarray, start_index: int) -> np.ndarray:
    """UGRID face_nodes array to 0-based with -1 fill."""
    face_nodes_raw = np.asarray(face_nodes_raw, dtype=np.int64)
    if int(start_index) == 1:
        face_nodes = np.full_like(face_nodes_raw, -1)
        valid = face_nodes_raw > 0
        face_nodes[valid] = face_nodes_raw[valid] - 1
        return face_nodes
    return face_nodes_raw.copy()


def _fan_vert_tria_from_ds(
    ds_cur,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangle proxy of the mixed UGRID dataset, aligned with quad splitting
    used by the rest of this repository.

    Returns
    -------
    vert_xy : (N,2) float64
        Triangle-proxy vertex coordinates.
    tria : (T,3) int64
        Triangle connectivity of the proxy.
    tri_origin_face_id : (T,) int64
        Index of the original mixed-face row from which each triangle originates.
        (Triangles coming from the same quad share the same origin id.)
    quad_face_mask : (F,) bool
        Whether each original mixed-face row is a quad (len==4).
    """
    face_nodes_raw = np.asarray(ds_cur["mesh2d_face_nodes"].values, dtype=np.int64)
    start_index = int(ds_cur["mesh2d_face_nodes"].attrs.get("start_index", 1))
    face_nodes = _face_nodes_raw_to_0b(face_nodes_raw, start_index)
    faces = _faces_from_face_nodes(face_nodes)

    node_x = np.asarray(ds_cur["mesh2d_node_x"].values, dtype=np.float64)
    node_y = np.asarray(ds_cur["mesh2d_node_y"].values, dtype=np.float64)
    vert_xy = np.column_stack([node_x, node_y])
    xy = vert_xy[:, :2]

    from ..geomesh_util.grd_util import triangulate_mixed_face_row_to_tris

    quad_face_mask = np.asarray([(f.size == 4) for f in faces], dtype=bool)
    tris: List[Tuple[int, int, int]] = []
    tri_origin_face_id: List[int] = []

    for fid, nodes in enumerate(faces):
        tri_list = triangulate_mixed_face_row_to_tris(xy, np.asarray(nodes, dtype=np.int64))
        for tri in tri_list:
            tris.append((int(tri[0]), int(tri[1]), int(tri[2])))
            tri_origin_face_id.append(int(fid))

    tria = np.asarray(tris, dtype=np.int64)
    tri_origin_face_id = np.asarray(tri_origin_face_id, dtype=np.int64)

    return vert_xy, tria, tri_origin_face_id, quad_face_mask


def dual_criteria_on_fan_mesh(
    vert: np.ndarray,
    tria: np.ndarray,
    tri_origin_face_id: np.ndarray,
    quad_face_mask: np.ndarray,
    *,
    cosphi_threshold: float,
    removesmalllinkstrsh: float,
) -> Tuple[bool, float, int]:
    """
    Returns (ok, max_abs_cosphi, n_small_flow_links) using meshkernel definitions
    on the merge-consistent triangle mesh (same proxy as ortho).
    """
    from . import meshkernel_orthogonalize_3 as mk3
    from .meshkernel_orthogonalize_3_tria import _build_edges_from_tria

    vert = np.asarray(vert, dtype=np.float64)
    tria = np.asarray(tria, dtype=np.int64)
    if tria.size == 0:
        return True, 0.0, 0

    edge_nodes, edge_faces = _build_edges_from_tria(tria)
    _, _, cosphi_abs = mk3.compute_cosphi_abs_from_arrays(
        vert[:, 0],
        vert[:, 1],
        tria,
        edge_nodes,
        edge_faces,
        use_file_centers=False,
        use_circumcenter_3d=True,
    )
    mask = ~np.isnan(cosphi_abs)
    max_c = float(np.nanmax(cosphi_abs[mask])) if np.any(mask) else 0.0
    # MeshKernel small-flow-links should ignore edges internal to a quad in the
    # original mixed mesh. In the triangle-proxy, those correspond to the shared
    # diagonal between the two triangles coming from the same quad-face row.
    n_edges = edge_faces.shape[0]
    keep_edge_indices = np.arange(n_edges, dtype=np.int64)
    exclude_mask = np.zeros(n_edges, dtype=bool)
    for e in range(n_edges):
        f1 = int(edge_faces[e, 0])
        f2 = int(edge_faces[e, 1])
        if f1 < 0 or f2 < 0 or f1 == f2:
            continue
        o1 = int(tri_origin_face_id[f1])
        o2 = int(tri_origin_face_id[f2])
        if o1 == o2 and bool(quad_face_mask[o1]):
            exclude_mask[e] = True

    keep_edge_indices = keep_edge_indices[~exclude_mask]
    n_small, _ = mk3.compute_small_links_from_arrays(
        vert[:, 0],
        vert[:, 1],
        tria,
        edge_nodes,
        edge_faces,
        removesmalllinkstrsh=float(removesmalllinkstrsh),
        edge_indices=keep_edge_indices,
    )

    ok = (max_c <= float(cosphi_threshold) + 1.0e-9) and (int(n_small) == 0)
    return ok, max_c, int(n_small)


def ortho_merge_iterate_dataset(
    ds,
    *,
    outer_iter_max: int = 5,
    cosphi_threshold: float = 0.49,
    removesmalllinkstrsh: float = 0.11,
    buffer_layers: int = 2,
    max_global_iter: int = 6,
    smooth_iter: int = 16,
    enable_edge_flips: bool = True,
    stop_if_no_merge: bool = True,
    ortho_disable_smalllink_logic: bool = True,
    require_both_criteria: bool = False,
    max_recovery_iterations: int = 25,
    recovery_stagnation_break: int = 3,
    on_state: Optional[Callable[[OrthoMergeStats], None]] = None,
) -> tuple:
    """
    Iteratively apply (**orthogonalize → merge_circumcenters**) on a UGRID dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Delft3D-FM UGRID mesh dataset.
    outer_iter_max : int
        Number of outer iterations (ortho+merge cycles).
    cosphi_threshold, removesmalllinkstrsh, buffer_layers, max_global_iter, smooth_iter
        Passed to the triangle orthogonalizer (V3 wrapper).
    enable_edge_flips : bool
        Allow triangle edge flips during the ortho step.
    stop_if_no_merge : bool
        Stop outer loop when merge does not reduce number of faces (no merge applied).
    ortho_disable_smalllink_logic : bool
        If True, the ortho step is focused on orthogonality only (small-link handling
        disabled in ortho) and small-link removal is delegated to merge_circumcenters.
    require_both_criteria : bool
        If True, after the main loop (and optional recovery), require
        ``max|cosφ| <= cosphi_threshold`` and ``n_small_flow_links == 0`` on the
        triangle proxy (quad diagonal ``(v1,v2)``); otherwise raise ``RuntimeError``. **False** (default)
        skips that check and recovery — same idea as ``backup_ortho_merge_20260319_174424``.
    max_recovery_iterations : int
        Max extra ortho+merge cycles when criteria fail after the main loop.
        Ignored if ``require_both_criteria`` is False.
    recovery_stagnation_break : int
        Stop recovery early if ``(max|cosφ|, n_small)`` is unchanged for this many
        consecutive recovery cycles (0 = disable). Avoids paying for many no-op passes.

    Returns
    -------
    (ds_final, stats_list)
    """
    import xarray as xr

    from ..geomesh_util.merge_circumcenters import (
        _rebuild_ds_from_form,
        build_ugrid_arrays_mixed,
        merge_circumcenters,
    )

    from .meshkernel_orthogonalize_3_tria import orthogonalize_tria_mesh

    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset")

    stats: List[OrthoMergeStats] = []

    # Work on a copy to avoid mutating caller data
    ds_cur = ds.copy(deep=True)

    def _run_ortho_merge_cycle(ds_in):
        """One ortho (dual-consistent tris of mixed faces) + merge_circumcenters. Returns updated ds."""
        ds_before_outer = ds_in.copy(deep=True)
        node_x = np.asarray(ds_in["mesh2d_node_x"].values, dtype=np.float64)
        node_y = np.asarray(ds_in["mesh2d_node_y"].values, dtype=np.float64)
        node_z = (
            np.asarray(ds_in["mesh2d_node_z"].values, dtype=np.float64)
            if "mesh2d_node_z" in ds_in
            else np.zeros((node_x.shape[0],), dtype=np.float64)
        )
        vert = np.column_stack([node_x, node_y])
        face_nodes_raw = np.asarray(ds_in["mesh2d_face_nodes"].values, dtype=np.int64)
        start_index = int(ds_in["mesh2d_face_nodes"].attrs.get("start_index", 1))
        face_nodes = _face_nodes_raw_to_0b(face_nodes_raw, start_index)
        faces = _faces_from_face_nodes(face_nodes)
        tria_for_ortho = _triangulate_faces_for_ortho(vert, faces)

        ortho_smalllink_trsh = 1.0e-12 if ortho_disable_smalllink_logic else removesmalllinkstrsh
        ortho_res = orthogonalize_tria_mesh(
            vert,
            tria_for_ortho,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=ortho_smalllink_trsh,
            buffer_layers=buffer_layers,
            max_global_iter=max_global_iter,
            smooth_iter=smooth_iter,
            enable_edge_flips=(enable_edge_flips and (not ortho_disable_smalllink_logic)),
        )

        NODE = np.column_stack([ortho_res.vert[:, 0], ortho_res.vert[:, 1], node_z])
        ugrid_arrays = build_ugrid_arrays_mixed(NODE, faces)
        ds_after_ortho = _rebuild_ds_from_form(ds_in, ugrid_arrays)

        nfaces_before = int(
            ds_after_ortho.sizes.get("mesh2d_nFaces", ds_after_ortho["mesh2d_face_nodes"].shape[0])
        )
        ds_merged = merge_circumcenters(ds_after_ortho, removesmalllinkstrsh=removesmalllinkstrsh)
        nfaces_after = int(ds_merged.sizes.get("mesh2d_nFaces", ds_merged["mesh2d_face_nodes"].shape[0]))
        merged_this_iter = max(0, nfaces_before - nfaces_after)

        return ds_merged, ortho_res, merged_this_iter, ds_before_outer

    for outer in range(int(outer_iter_max)):
        ds_cur, ortho_res, merged_this_iter, ds_before_outer = _run_ortho_merge_cycle(ds_cur)

        # Guardrail: if no merge happened and orthogonality got worse than previous outer-iter,
        # revert this outer step and stop.
        if (
            merged_this_iter == 0
            and len(stats) > 0
            and float(ortho_res.max_cosphi) > (float(stats[-1].max_cosphi) + 1.0e-9)
        ):
            ds_cur = ds_before_outer
            stats.append(
                OrthoMergeStats(
                    outer_iter=outer,
                    max_cosphi=float(stats[-1].max_cosphi),
                    n_small_flow_links=int(stats[-1].n_small_flow_links),
                    merged_this_iter=0,
                    n_zones_orthogonalized=int(getattr(ortho_res, "n_zones_orthogonalized", 0)),
                )
            )
            if on_state is not None:
                on_state(stats[-1])
            break
        stats.append(
            OrthoMergeStats(
                outer_iter=outer,
                max_cosphi=float(ortho_res.max_cosphi),
                n_small_flow_links=int(ortho_res.n_small_flow_links),
                merged_this_iter=int(merged_this_iter),
                n_zones_orthogonalized=int(getattr(ortho_res, "n_zones_orthogonalized", 0)),
            )
        )
        if on_state is not None:
            on_state(stats[-1])

        if stop_if_no_merge and merged_this_iter == 0:
            break

    if bool(require_both_criteria):
        v_chk, t_chk, tri_origin_face_id, quad_face_mask = _fan_vert_tria_from_ds(ds_cur)
        ok, max_c, n_s = dual_criteria_on_fan_mesh(
            v_chk,
            t_chk,
            tri_origin_face_id,
            quad_face_mask,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=removesmalllinkstrsh,
        )
        max_rec = max(0, int(max_recovery_iterations))
        stall_need = int(recovery_stagnation_break)
        prev_metric_key = (round(max_c, 9), int(n_s))
        stall = 0
        r = 0
        while (not ok) and r < max_rec:
            ds_cur, ortho_res, merged_this_iter, ds_before_outer = _run_ortho_merge_cycle(ds_cur)
            # Recovery: no guardrail revert (keep trying); always log cycle.
            stats.append(
                OrthoMergeStats(
                    outer_iter=r,
                    max_cosphi=float(ortho_res.max_cosphi),
                    n_small_flow_links=int(ortho_res.n_small_flow_links),
                    merged_this_iter=int(merged_this_iter),
                    n_zones_orthogonalized=int(getattr(ortho_res, "n_zones_orthogonalized", 0)),
                    recovery=True,
                    recovery_iter=r,
                )
            )
            if on_state is not None:
                on_state(stats[-1])
            v_chk, t_chk, tri_origin_face_id, quad_face_mask = _fan_vert_tria_from_ds(ds_cur)
            ok, max_c, n_s = dual_criteria_on_fan_mesh(
                v_chk,
                t_chk,
                tri_origin_face_id,
                quad_face_mask,
                cosphi_threshold=cosphi_threshold,
                removesmalllinkstrsh=removesmalllinkstrsh,
            )
            r += 1
            if stall_need > 0:
                key = (round(max_c, 9), int(n_s))
                if key == prev_metric_key:
                    stall += 1
                    if stall >= stall_need:
                        break
                else:
                    stall = 0
                    prev_metric_key = key

        if not ok:
            raise RuntimeError(
                "ortho_merge_iterate_dataset: mesh still violates dual criteria after "
                f"{int(outer_iter_max)} main cycle(s) and {r} recovery cycle(s). "
                f"Required: max|cosφ| <= {cosphi_threshold} and n_small_flow_links == 0 "
                f"(triangle proxy). Got max|cosφ|={max_c:.6f}, n_small={n_s}. "
                "Increase max_recovery_iterations / outer_iter_max, relax thresholds, "
                "or improve the initial mesh."
            )

    return ds_cur, stats


def print_stats(stats: Sequence[OrthoMergeStats], *, print_header: bool = False) -> None:
    """Smooth-like ortho+merge summary block (one header + progressive rows)."""
    if print_header:
        print(" -------------------------------------------------------")
        print("      |STATE.|      |MAX|COS(PHI)| |N_SMALL| |N_ZONES|")
        print(" -------------------------------------------------------")

    for s in stats:
        if getattr(s, "recovery", False):
            head = f"recovery={s.recovery_iter}"
        else:
            # Special markers used by smood for start/end snapshots.
            if int(s.outer_iter) == -1:
                head = "initial"
            elif int(s.outer_iter) == -2:
                head = "final"
            else:
                head = f"outer={s.outer_iter}"

        print(
            f"    {head:<11}{s.max_cosphi:>13.6f}{int(s.n_small_flow_links):>14d}{int(s.n_zones_orthogonalized):>11d}",
            flush=True,
        )


def ortho_merge_iterate_tria(
    vert: np.ndarray,
    tria: np.ndarray,
    *,
    node_z: Optional[np.ndarray] = None,
    outer_iter_max: int = 5,
    cosphi_threshold: float = 0.49,
    removesmalllinkstrsh: float = 0.11,
    buffer_layers: int = 2,
    max_global_iter: int = 6,
    smooth_iter: int = 16,
    enable_edge_flips: bool = True,
    stop_if_no_merge: bool = True,
    ortho_disable_smalllink_logic: bool = True,
    require_both_criteria: bool = False,
    max_recovery_iterations: int = 25,
    recovery_stagnation_break: int = 3,
    on_state: Optional[Callable[[OrthoMergeStats], None]] = None,
) -> tuple:
    """
    Convenience wrapper that starts from a pure triangle mesh (vert, tria).

    Parameters
    ----------
    vert : (N,2) float array
        Node coordinates (lon/lat degrees, consistent with your UGRID usage).
    tria : (T,3) int array
        0-based triangle connectivity.
    node_z : (N,) optional
        If provided, will be stored as mesh2d_node_z. If None, zeros are used.

    Returns
    -------
    (vert_out, face_nodes_out, stats)
        - vert_out: (N,2) updated node coordinates
        - face_nodes_out: (F,4) int with fill -1 for triangles, quads have 4 nodes (0-based)
        - stats: list[OrthoMergeStats]
    """
    import xarray as xr

    from ..geomesh_util.grd_util import adcirc2DFlowFM

    vert = np.asarray(vert, dtype=np.float64)
    tria = np.asarray(tria, dtype=np.int64)
    if vert.ndim != 2 or vert.shape[1] != 2:
        raise ValueError("vert must have shape (N,2)")
    if tria.ndim != 2 or tria.shape[1] != 3:
        raise ValueError("tria must have shape (T,3)")

    if node_z is None:
        node_z = np.zeros((vert.shape[0],), dtype=np.float64)
    else:
        node_z = np.asarray(node_z, dtype=np.float64).reshape(-1)
        if node_z.shape[0] != vert.shape[0]:
            raise ValueError("node_z must have length N (same as vert)")

    NODE = np.column_stack([vert[:, 0], vert[:, 1], node_z])
    ds0 = adcirc2DFlowFM(NODE=NODE, EDGE=tria)
    if not isinstance(ds0, xr.Dataset):
        raise TypeError("adcirc2DFlowFM must return an xarray.Dataset in this repository")

    ds_final, stats = ortho_merge_iterate_dataset(
        ds0,
        outer_iter_max=outer_iter_max,
        cosphi_threshold=cosphi_threshold,
        removesmalllinkstrsh=removesmalllinkstrsh,
        buffer_layers=buffer_layers,
        max_global_iter=max_global_iter,
        smooth_iter=smooth_iter,
        enable_edge_flips=enable_edge_flips,
        stop_if_no_merge=stop_if_no_merge,
        ortho_disable_smalllink_logic=ortho_disable_smalllink_logic,
        require_both_criteria=require_both_criteria,
        max_recovery_iterations=max_recovery_iterations,
        recovery_stagnation_break=recovery_stagnation_break,
        on_state=on_state,
    )

    vert_out = np.column_stack(
        [
            np.asarray(ds_final["mesh2d_node_x"].values, dtype=np.float64),
            np.asarray(ds_final["mesh2d_node_y"].values, dtype=np.float64),
        ]
    )
    face_nodes_1b = np.asarray(ds_final["mesh2d_face_nodes"].values, dtype=np.int64)
    start_index = int(ds_final["mesh2d_face_nodes"].attrs.get("start_index", 1))
    fill = -1
    if start_index == 1:
        # Common convention: invalid nodes are stored as 0 for 1-based meshes.
        face_nodes_0b = np.where(face_nodes_1b > 0, face_nodes_1b - 1, fill)
    else:
        # Assume already 0-based with invalid encoded as -1.
        face_nodes_0b = face_nodes_1b.copy()

    return vert_out, face_nodes_0b, stats

