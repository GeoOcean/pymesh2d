"""
Small, dependency-free geometric/topological helpers shared across the
`ortho_merge` modules (previously duplicated verbatim in
`meshkernel_orthogonalize_3.py` and `meshkernel_orthogonalize_3_tria.py`).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def build_edges_from_tria(tria: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (edge_nodes, edge_faces) from 0-based triangles.

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

    edge_map: Dict[Tuple[int, int], int] = {}
    edge_nodes: List[Tuple[int, int]] = []
    edge_faces: List[List[int]] = []

    def _add_edge(a: int, b: int, f: int) -> None:
        i, j = (a, b) if a < b else (b, a)
        key = (i, j)
        if key in edge_map:
            eidx = edge_map[key]
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
