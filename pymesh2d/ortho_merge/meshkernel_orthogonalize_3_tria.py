"""
Triangle-mesh wrapper around `meshkernel_orthogonalize_3.py`.

This module provides an importable API that works directly with:
- `vert`: (N, 2) array of lon/lat (degrees) OR projected x/y (if you set `jsferic=0`).
- `tria`: (T, 3) array of 0-based triangle node indices.

It reuses the zone-based orthogonalization + small-link handling from
`meshkernel_orthogonalize_3.py`, but avoids any NetCDF I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

# Reuse the battle-tested numerics/logic from the sibling module in this package
from . import meshkernel_orthogonalize_3 as mk3


@dataclass
class TriaOrthoResult:
    vert: np.ndarray  # (N,2) float64
    tria: np.ndarray  # (T,3) int64 (unchanged topology unless edge flips are enabled)
    max_cosphi: float
    n_small_flow_links: int
    n_zones_orthogonalized: int


def _build_edges_from_tria(tria: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (edge_nodes, edge_faces) from triangles.

    Parameters
    ----------
    tria : (T,3) int array, 0-based.

    Returns
    -------
    edge_nodes : (E,2) int64
    edge_faces : (E,2) int64 (right face = -1 for boundary)
    """
    tria = np.asarray(tria, dtype=np.int64)
    if tria.ndim != 2 or tria.shape[1] != 3:
        raise ValueError("tria must be an array of shape (T,3) with 0-based indices")

    # Map undirected edge -> (n1,n2,eidx) and faces
    edge_map: Dict[Tuple[int, int], int] = {}
    edge_nodes: List[Tuple[int, int]] = []
    edge_faces = []  # list of [f_left, f_right]

    def _add_edge(a: int, b: int, f: int) -> None:
        i, j = (a, b) if a < b else (b, a)
        key = (i, j)
        if key in edge_map:
            eidx = edge_map[key]
            # fill second face slot if available
            if edge_faces[eidx][1] == -1:
                edge_faces[eidx][1] = f
        else:
            eidx = len(edge_nodes)
            edge_map[key] = eidx
            edge_nodes.append((i, j))
            edge_faces.append([f, -1])

    for f in range(tria.shape[0]):
        a, b, c = int(tria[f, 0]), int(tria[f, 1]), int(tria[f, 2])
        _add_edge(a, b, f)
        _add_edge(b, c, f)
        _add_edge(c, a, f)

    return np.asarray(edge_nodes, dtype=np.int64), np.asarray(edge_faces, dtype=np.int64)


def orthogonalize_tria_mesh(
    vert: np.ndarray,
    tria: np.ndarray,
    *,
    cosphi_threshold: float = 0.49,
    removesmalllinkstrsh: float = 0.1,
    buffer_layers: int = 2,
    max_global_iter: int = 8,
    smooth_iter: int = 16,
    enable_edge_flips: bool = True,
) -> TriaOrthoResult:
    """
    Orthogonalize a pure triangle mesh using the V3 zone logic.

    Notes
    -----
    - Topology changes can occur only if `enable_edge_flips=True`, and only via
      local edge flips inside convex quads (still triangles).
    - No merging into quads is performed here.
    - This uses the same cosphi/small-link definitions as `meshkernel_orthogonalize_3.py`.
    """
    vert = np.asarray(vert, dtype=np.float64)
    if vert.ndim != 2 or vert.shape[1] != 2:
        raise ValueError("vert must be an array of shape (N,2)")
    tria = np.asarray(tria, dtype=np.int64)

    face_nodes = tria.copy()
    edge_nodes, edge_faces = _build_edges_from_tria(tria)

    mesh = mk3.MeshData(
        node_x=vert[:, 0].copy(),
        node_y=vert[:, 1].copy(),
        face_nodes=face_nodes,
        edge_nodes=edge_nodes,
        edge_faces=edge_faces,
    )

    n_faces = face_nodes.shape[0]
    face_neighbors = mk3.build_face_adjacency(mesh.edge_faces, n_faces)

    # Count how many zone-orthogonalization passes we run. This is used for logging in
    # the ortho+merge pipeline (to replace the old merged-delta column in the log).
    n_zones_orthogonalized = 0

    # Global loop by zones (mirrors mk3.orthogonalize_netcdf without I/O)
    for it in range(int(max_global_iter)):
        _, _, cosphi_abs = mk3.compute_cosphi_abs_from_arrays(
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
            break

        max_cosphi = float(np.nanmax(cosphi_abs[mask]))
        bad_edges = np.where((mask) & (cosphi_abs > cosphi_threshold))[0]
        n_small, small_edges_arr = mk3.compute_small_links_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
        )

        # Optional: edge flips pre-pass (still triangles)
        # If no small links remain but orthogonality is still bad, also let
        # problematic edges participate so the mesh can keep improving.
        if enable_edge_flips:
            flip_candidates = small_edges_arr if n_small > 0 else bad_edges
            _ = mk3.try_flip_candidate_edges_ugrid(
                mesh,
                flip_candidates,
                removesmalllinkstrsh,
                max_cosphi_allowed=(cosphi_threshold if n_small == 0 else None),
            )
            # After flips, recompute edges & faces because topology changed
            # (face_nodes changed, but edge_nodes/edge_faces are now stale)
            mesh.edge_nodes, mesh.edge_faces = _build_edges_from_tria(mesh.face_nodes[:, :3])
            face_neighbors = mk3.build_face_adjacency(mesh.edge_faces, n_faces)
            _, _, cosphi_abs = mk3.compute_cosphi_abs_from_arrays(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                use_file_centers=False,
                use_circumcenter_3d=True,
            )
            mask = ~np.isnan(cosphi_abs)
            max_cosphi = float(np.nanmax(cosphi_abs[mask])) if np.any(mask) else max_cosphi
            n_small, small_edges_arr = mk3.compute_small_links_from_arrays(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                removesmalllinkstrsh=removesmalllinkstrsh,
            )

        if max_cosphi <= cosphi_threshold and n_small == 0:
            break

        bad_edges = np.where((mask) & (cosphi_abs > cosphi_threshold))[0]
        bad_set = set(int(e) for e in bad_edges.tolist())
        sort_idx = np.argsort(cosphi_abs[bad_edges])[::-1] if bad_edges.size > 0 else np.array([], dtype=np.int64)
        bad_edges_sorted = bad_edges[sort_idx] if bad_edges.size > 0 else np.array([], dtype=np.int64)
        small_only = np.array([e for e in small_edges_arr.tolist() if e not in bad_set], dtype=np.int64)
        problematic_edges = np.concatenate([bad_edges_sorted, small_only]) if bad_edges_sorted.size > 0 else small_only
        if problematic_edges.size == 0:
            break

        visited_faces_global: Set[int] = set()
        for e in problematic_edges:
            f1, f2 = mesh.edge_faces[e, :]
            start_faces: List[int] = []
            if f1 >= 0:
                start_faces.append(int(f1))
            if f2 >= 0:
                start_faces.append(int(f2))
            if not start_faces:
                continue

            this_buffer = buffer_layers + (1 if int(e) in bad_set else 0)
            faces_zone = mk3.bfs_faces(start_faces, face_neighbors, this_buffer)
            if faces_zone.issubset(visited_faces_global):
                continue

            mk3.apply_combined_ortho_smoother_to_zone(
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
            n_zones_orthogonalized += 1
            visited_faces_global.update(faces_zone)

    # Final metrics
    _, _, cosphi_abs_final = mk3.compute_cosphi_abs_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        use_file_centers=False,
        use_circumcenter_3d=True,
    )
    mask_final = ~np.isnan(cosphi_abs_final)
    max_final = float(np.nanmax(cosphi_abs_final[mask_final])) if np.any(mask_final) else float("nan")
    n_small_final, _ = mk3.compute_small_links_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        removesmalllinkstrsh=removesmalllinkstrsh,
    )

    vert_out = np.column_stack([mesh.node_x, mesh.node_y]).astype(np.float64, copy=False)
    tria_out = np.asarray(mesh.face_nodes[:, :3], dtype=np.int64)
    return TriaOrthoResult(
        vert=vert_out,
        tria=tria_out,
        max_cosphi=max_final,
        n_small_flow_links=int(n_small_final),
        n_zones_orthogonalized=int(n_zones_orthogonalized),
    )

