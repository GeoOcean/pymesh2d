import warnings

import pyproj

import numpy as np
from .geomesh_util.grd_util import triangulate_mixed_face_row_to_tris
from .mesh_util.tricon import tricon
from .geom_util.proj_util import get_local_utm_crs, reproject_node

warnings.filterwarnings('ignore', category=RuntimeWarning)

BACKUP_ORTHO_MERGE_SMALLLINK_THRESHOLD: float = 0.11
BACKUP_ORTHO_MERGE_REQUIRE_STRICT_DUAL: bool = False


def _signed_area_quad(vert, quad):
    """Signed area (doubled) of quadrilateral (a,b,c,d) for CCW check."""
    v = vert[quad]
    return (
        (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
        - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
        + (v[3, 0] - v[1, 0]) * (v[2, 1] - v[1, 1])
        - (v[2, 0] - v[1, 0]) * (v[3, 1] - v[1, 1])
    )


def _merge_small_links_quads_fan_tria_from_indices(
    tria,
    tnum,
    edge_cc,
    small_link_indices,
    vert_utm,
):
    """
    Merge each small-flow-link pair into one quad, then rebuild back to
    triangles by splitting the quad along the shared edge diagonal.
    """
    tria = np.asarray(tria, dtype=np.int64)
    tnum_1d = np.asarray(tnum, dtype=np.int64).reshape(-1)

    merged_tri = np.zeros(tria.shape[0], dtype=bool)
    quads = []  # list of (quad_nodes_4, part_idx)

    for idx in small_link_indices:
        e = edge_cc[int(idx)]
        v1, v2, t1, t2 = int(e[0]), int(e[1]), int(e[2]), int(e[3])

        if t2 < 0:
            continue
        if merged_tri[t1] or merged_tri[t2]:
            continue

        tri1 = tria[t1]
        tri2 = tria[t2]

        mask1 = (tri1 != v1) & (tri1 != v2)
        mask2 = (tri2 != v1) & (tri2 != v2)
        if not np.any(mask1) or not np.any(mask2):
            continue

        a = int(tri1[mask1][0])
        b = int(tri2[mask2][0])

        quad = np.array([a, v1, b, v2], dtype=np.int64)
        if _signed_area_quad(vert_utm, quad) < 0:
            quad = np.array([a, v2, b, v1], dtype=np.int64)

        part_idx = int(tnum_1d[t1])
        quads.append((quad, part_idx))
        merged_tri[t1] = True
        merged_tri[t2] = True

    new_tris = []
    new_parts = []

    for quad, part_idx in quads:
        # quad is `[a, v1, b, v2]` where (v1,v2) is the shared merged edge.
        # Fan triangulation around `a`:
        # (a, v1, b) and (a, b, v2)
        a = int(quad[0])
        v1 = int(quad[1])
        b = int(quad[2])
        v2 = int(quad[3])
        new_tris.append((a, v1, b))
        new_parts.append(part_idx)
        new_tris.append((a, b, v2))
        new_parts.append(part_idx)

    for ti in range(tria.shape[0]):
        if not merged_tri[ti]:
            new_tris.append(tuple(tria[ti].tolist()))
            new_parts.append(int(tnum_1d[ti]))

    tria_out = np.asarray(new_tris, dtype=np.int64)
    tnum_out = np.asarray(new_parts, dtype=np.int64).reshape(-1, 1)
    return tria_out, tnum_out, len(quads)


def _count_small_flow_links(tria, vert_lonlat, conn, removesmalllinkstrsh):
    """
    Compute number of small flow links using the same metric as meshkernel.

    This aligns with `meshkernel_orthogonalize_3.compute_small_links_from_arrays`,
    which is also what we use to validate `n_small` outside of `pymesh2d`.
    """
    from .ortho_merge import meshkernel_orthogonalize_3 as mk3

    tria = np.asarray(tria, dtype=np.int64)
    vert_lonlat = np.asarray(vert_lonlat, dtype=np.float64)
    if vert_lonlat.ndim != 2 or vert_lonlat.shape[1] != 2:
        raise ValueError("vert_lonlat must have shape (N,2) [lon,lat]")

    edge_cc, _ = tricon(tria, conn)
    edge_nodes = edge_cc[:, 0:2]
    edge_faces = edge_cc[:, 2:4]

    n_small, _ = mk3.compute_small_links_from_arrays(
        node_x=vert_lonlat[:, 0],
        node_y=vert_lonlat[:, 1],
        face_nodes=tria,
        edge_nodes=edge_nodes,
        edge_faces=edge_faces,
        removesmalllinkstrsh=removesmalllinkstrsh,
    )
    return int(n_small)


def _smood_ortho_merge_backup_pipeline(vert, conn, tria, tnum, opts):
    """
    Pipeline from ``backup_ortho_merge_20260319_174424``: repeated orthogonalize (mixed faces
    triangulated like export: quad diagonal ``(v1,v2)``, not fan-from-``a``)
    then ``merge_circumcenters``, same numerical defaults as that folder except
    ``smalllink_threshold`` defaults to ``BACKUP_ORTHO_MERGE_SMALLLINK_THRESHOLD`` (0.11).

    When ``require_both_criteria`` is False (default), behaviour matches the backup
    (no global dual check / no recovery). Set it True to require the fan-proxy dual
    criteria and run recovery cycles (see :mod:`pymesh2d.ortho_merge.ortho_merge_iter`).
    """
    from .ortho_merge.ortho_merge_iter import ortho_merge_iterate_tria, print_stats

    vert_in = np.asarray(vert, dtype=np.float64)
    tria_in = np.asarray(tria, dtype=np.int64)
    tnum_in = np.asarray(tnum, dtype=np.int64).reshape(-1)

    outer_iter_max = int(opts.get("iter", 4))
    outer_iter_max = max(1, min(outer_iter_max, 4))

    smalllink_trsh = float(
        opts.get("smalllink_threshold", BACKUP_ORTHO_MERGE_SMALLLINK_THRESHOLD)
    )
    require_strict = bool(
        opts.get("require_both_criteria", BACKUP_ORTHO_MERGE_REQUIRE_STRICT_DUAL)
    )

    do_log = not np.isinf(opts.get("disp", 4))

    # Initial snapshot (triangle proxy: 1 mixed-face row per input triangle).
    init_max_c = None
    init_n_small = None
    if do_log:
        cosphi_threshold = float(opts.get("orthogonality_threshold", 0.49))
        removesmalllinkstrsh = smalllink_trsh
        tri_origin_face_id = np.arange(tria_in.shape[0], dtype=np.int64)
        quad_face_mask = np.zeros(tria_in.shape[0], dtype=bool)

        from .ortho_merge.ortho_merge_iter import dual_criteria_on_fan_mesh, OrthoMergeStats

        _, init_max_c, init_n_small = dual_criteria_on_fan_mesh(
            np.asarray(vert_in, dtype=np.float64),
            tria_in,
            tri_origin_face_id,
            quad_face_mask,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=removesmalllinkstrsh,
        )
        print_stats(
            [
                OrthoMergeStats(
                    outer_iter=-1,
                    max_cosphi=float(init_max_c),
                    n_small_flow_links=int(init_n_small),
                    merged_this_iter=0,
                    n_zones_orthogonalized=0,
                )
            ],
            print_header=True,
        )

    def _on_state(s):
        if do_log:
            print_stats([s], print_header=False)

    # Keep output compact: disable verbose per-zone logs inside meshkernel orthogonalization.
    from .ortho_merge import meshkernel_orthogonalize_3 as mk3
    old_verbose = getattr(mk3, "VERBOSE_ZONE_LOGS", True)
    mk3.VERBOSE_ZONE_LOGS = False
    try:
        vert_out, face_nodes_0b, stats = ortho_merge_iterate_tria(
            vert_in,
            tria_in,
            node_z=None,
            outer_iter_max=outer_iter_max,
            cosphi_threshold=float(opts.get("orthogonality_threshold", 0.49)),
            removesmalllinkstrsh=smalllink_trsh,
            buffer_layers=int(opts.get("buffer_layers", 2)),
            max_global_iter=int(opts.get("max_global_iter", int(opts.get("inner_iter", 4)) + 2)),
            smooth_iter=int(opts.get("smooth_iter", int(opts.get("inner_iter", 4)) * 4)),
            enable_edge_flips=bool(opts.get("enable_edge_flips", True)),
            stop_if_no_merge=True,
            ortho_disable_smalllink_logic=True,
            require_both_criteria=require_strict,
            max_recovery_iterations=int(opts.get("max_recovery_iterations", 25)),
            recovery_stagnation_break=int(opts.get("recovery_stagnation_break", 3)),
            on_state=_on_state if do_log else None,
        )
    finally:
        mk3.VERBOSE_ZONE_LOGS = old_verbose

    # Final snapshot (triangle proxy built from mixed faces).
    if do_log:
        cosphi_threshold = float(opts.get("orthogonality_threshold", 0.49))
        removesmalllinkstrsh = smalllink_trsh

        from .ortho_merge.ortho_merge_iter import dual_criteria_on_fan_mesh, OrthoMergeStats

        face_nodes_0b_arr = np.asarray(face_nodes_0b, dtype=np.int64)
        vert_xy = np.asarray(vert_out, dtype=np.float64)

        tria_proxy = []
        tri_origin_face_id = []
        quad_face_mask = np.zeros(face_nodes_0b_arr.shape[0], dtype=bool)

        for fid, row in enumerate(face_nodes_0b_arr):
            nodes = row[row >= 0]
            if nodes.size < 3:
                continue
            quad_face_mask[fid] = nodes.size == 4
            for t in triangulate_mixed_face_row_to_tris(vert_xy, nodes):
                tria_proxy.append(t)
                tri_origin_face_id.append(fid)

        tria_proxy = np.asarray(tria_proxy, dtype=np.int64)
        tri_origin_face_id = np.asarray(tri_origin_face_id, dtype=np.int64)

        _, fin_max_c, fin_n_small = dual_criteria_on_fan_mesh(
            vert_xy,
            tria_proxy,
            tri_origin_face_id,
            quad_face_mask,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=removesmalllinkstrsh,
        )

        last_zones = int(getattr(stats[-1], "n_zones_orthogonalized", 0)) if stats else 0
        print_stats(
            [
                OrthoMergeStats(
                    outer_iter=-2,
                    max_cosphi=float(fin_max_c),
                    n_small_flow_links=int(fin_n_small),
                    merged_this_iter=0,
                    n_zones_orthogonalized=last_zones,
                )
            ],
            print_header=False,
        )

    # Optional: keep merged quads for NetCDF export (see ``preserve_merged_quads`` in ``code.py``).
    preserve_merged_quads = bool(opts.get("preserve_merged_quads", False))
    face_nodes_arr = np.asarray(face_nodes_0b, dtype=np.int64)

    if preserve_merged_quads:
        opts["_mixed_face_nodes_0b"] = face_nodes_arr.copy()
    else:
        opts.pop("_mixed_face_nodes_0b", None)

    # If we actually have quads and the caller asked to preserve them, return
    # ``face_nodes_0b`` directly as the 3rd output (`tria`), so callers can do:
    #   ds_out = adcirc2DFlowFM(NODE, tria)
    # without needing the separate `mixed_fn` conditional.
    #
    # For purely triangulated meshes (no quads), keep the historical return
    # type: triangle connectivity (T,3).
    valid_counts = np.sum(face_nodes_arr >= 0, axis=1)
    has_quads = bool(np.any(valid_counts == 4))
    if preserve_merged_quads and has_quads:
        tria_out = face_nodes_arr.copy()
        # `tnum` is not used by `adcirc2DFlowFM`, but keep the shape consistent
        # with the returned face rows.
        tnum_out = np.ones((tria_out.shape[0], 1), dtype=np.int64)
        return np.asarray(vert_out, dtype=np.float64), conn, tria_out, tnum_out

    # Triangle-only connectivity.
    # Quads from merge_circumcenters: split on diagonal (v1,v2), not fan-from-a.
    new_tris = []
    new_parts = []
    vert_xy = np.asarray(vert_out, dtype=np.float64)
    for row in face_nodes_arr:
        nodes = row[row >= 0]
        if nodes.size < 3:
            continue
        for t in triangulate_mixed_face_row_to_tris(vert_xy, nodes):
            new_tris.append(t)
            new_parts.append(1)

    tria_out = np.asarray(new_tris, dtype=np.int64)
    if tria_out.size == 0:
        tria_out = tria_in.copy()
        tnum_out = np.asarray(tnum, dtype=np.int64)
    else:
        tnum_out = np.asarray(new_parts, dtype=np.int64).reshape(-1, 1)

    return np.asarray(vert_out, dtype=np.float64), conn, tria_out, tnum_out


def _smood_v3_ortho_merge(vert, conn, tria, tnum, opts):
    """
    V3 orthogonalization + small-flow-link removal through iterated
    tri->quad merge (then fan triangulation back to triangles).
    """
    use_backup_transition = bool(opts.get("use_backup_transition", True))
    if use_backup_transition:
        return _smood_ortho_merge_backup_pipeline(vert, conn, tria, tnum, opts)

    from .ortho_merge.meshkernel_orthogonalize_3_tria import orthogonalize_tria_mesh

    from .ortho_merge import meshkernel_orthogonalize_3 as mk3

    vert_cur = np.asarray(vert, dtype=np.float64)
    tria_cur = np.asarray(tria, dtype=np.int64)
    tnum_cur = np.asarray(tnum, dtype=np.int64)
    conn_cur = np.asarray(conn, dtype=np.int64) if conn is not None else np.empty((0, 2), dtype=np.int64)

    removesmalllinkstrsh = float(opts.get("smalllink_threshold", 0.11))

    cosphi_threshold = float(opts.get("orthogonality_threshold", 0.49))
    # Keep Delft3D-ish target as a minimum for stability.
    cosphi_threshold = max(cosphi_threshold, 0.49)

    outer_iter_max = int(opts.get("iter", 4))
    # Hard cap to keep runtime bounded.
    outer_iter_max = max(1, min(outer_iter_max, 4))

    inner_iter = int(opts.get("inner_iter", 4))
    buffer_layers = int(opts.get("buffer_layers", 2))
    max_global_iter = int(opts.get("max_global_iter", inner_iter + 2))
    smooth_iter = int(opts.get("smooth_iter", inner_iter * 4))
    enable_edge_flips = bool(opts.get("enable_edge_flips", True))

    need_smalllinks = bool(opts.get("converge_on_smalllinks", True))
    need_ortho = bool(opts.get("converge_on_orthogonality", True))
    # Ensure we run at least 2 outer cycles when convergence is requested:
    # merge can temporarily worsen orthogonality, so we need a second
    # orthogonalization pass after small-link handling.
    if (need_smalllinks or need_ortho) and outer_iter_max < 2:
        outer_iter_max = 2

    # Stability: disable small-link-specific logic inside orthogonalization,
    # and let merge_circumcenters-style tri->quad replacement handle it.
    ortho_disable_smalllink_logic = True
    ortho_smalllinkstrsh = 1.0e-12 if ortho_disable_smalllink_logic else removesmalllinkstrsh

    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    utm_crs = get_local_utm_crs(crs_wgs84, x=vert_cur[:, 0], y=vert_cur[:, 1])

    prev_max_cosphi = None

    for _outer in range(outer_iter_max):
        vert_before_outer = vert_cur.copy()
        tria_before_outer = tria_cur.copy()
        tnum_before_outer = tnum_cur.copy()

        # 1) Orthogonalize (node movement).
        ortho_res = orthogonalize_tria_mesh(
            vert_cur,
            tria_cur,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=ortho_smalllinkstrsh,
            buffer_layers=buffer_layers,
            max_global_iter=max_global_iter,
            smooth_iter=smooth_iter,
            enable_edge_flips=enable_edge_flips,
        )
        vert_cur = ortho_res.vert
        tria_cur = ortho_res.tria
        max_cosphi = float(ortho_res.max_cosphi)

        # 2) Merge small links (tri->quad) and fan triangulate back to triangles.
        edge_cc, _ = tricon(tria_cur, conn_cur)
        edge_nodes = edge_cc[:, 0:2]
        edge_faces = edge_cc[:, 2:4]
        n_small_now, small_link_indices = mk3.compute_small_links_from_arrays(
            node_x=vert_cur[:, 0],
            node_y=vert_cur[:, 1],
            face_nodes=tria_cur,
            edge_nodes=edge_nodes,
            edge_faces=edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
        )

        if int(n_small_now) > 0:
            # Only reproject when we actually need to merge (signed quad area uses meters).
            vert_utm = reproject_node(vert_cur, crs_wgs84, utm_crs)
            tria_cur, tnum_cur, n_quads = _merge_small_links_quads_fan_tria_from_indices(
                tria_cur,
                tnum_cur,
                edge_cc,
                small_link_indices,
                vert_utm,
            )
        else:
            n_quads = 0

        # 3) Optional guardrail: if merge did nothing and orthogonality got worse, revert.
        if n_quads == 0 and prev_max_cosphi is not None and max_cosphi > (prev_max_cosphi + 1.0e-9):
            vert_cur = vert_before_outer
            tria_cur = tria_before_outer
            tnum_cur = tnum_before_outer
            break

        prev_max_cosphi = max_cosphi

        # 4) Stop conditions.
        if need_ortho and max_cosphi > cosphi_threshold:
            continue

        if need_smalllinks:
            n_small_after = _count_small_flow_links(
                tria_cur,
                vert_cur,
                conn_cur,
                removesmalllinkstrsh,
            )
            if n_small_after > 0:
                continue

        break

    return vert_cur, conn, tria_cur, tnum_cur


def smood(vert=None, conn=None, tria=None, tnum=None, opts=None, hfun=None, harg=[]):
    """
    Perform mesh smoothing with orthogonalization.

    This function combines orthogonalization (optimizing aspect ratios) with
    smoothing (optimizing internal angles) to improve mesh quality for flow
    simulations.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        XY coordinates of the vertices in the triangulation.
    conn : ndarray of shape (E, 2)
        Array of constrained edges.
    tria : ndarray of shape (T, 3)
        Array of triangles (vertex indices).
    tnum : ndarray of shape (T, 1)
        Array of part indices.
    opts : dict, optional
        Dictionary containing user-defined parameters:
        - 'vtol' : float, default = 1.0e-3
          Relative vertex movement tolerance.
        - 'iter' : int, default = 16
          Maximum number of outer iterations.
        - 'inner_iter' : int, default = 4
          Number of inner iterations per outer iteration.
        - 'ortho_factor' : float, default = 0.5
          Maximum orthogonalization to smoothing factor (0.0 = pure smoothing, 1.0 = pure orthogonalization).
          The actual factor starts small and increases progressively during iterations (like Delft3D's mu).
        - 'relaxation' : float, default = 0.75
          Relaxation factor for coordinate updates.
        - 'converge_on_smalllinks' : bool, default = True
          Used when ``use_backup_transition`` is **False** (inline V3 loop): keep outer cycles until
          small-link count drops, subject to ``iter``.
        - 'converge_on_orthogonality' : bool, default = True
          Same branch: continue while ``max|cosφ|`` exceeds ``orthogonality_threshold``.
        - 'orthogonality_threshold' : float, default = 0.49
          Passed to the V3 triangle orthogonalizer (max allowed |cos φ| on internal flow links).
        - 'allow_constraint_sliding' / 'allow_constraint_sliding_junctions' : bool
          Reserved for :func:`makeopt_smood` compatibility (not used by the default ortho-merge path).
        - 'max_final_smalllink_iter' / 'smalllink_iter_start' / 'smalllink_iter_freq' : int
          Reserved for option-schema compatibility with older scripts.
        - 'smalllink_threshold' : float, default = 0.11
          Threshold for small flow links (``removesmalllinkstrsh`` in merge / meshkernel checks).
          Same role as in ``backup_ortho_merge_20260319_174424`` (that folder used 0.1; default here 0.11).
        - 'use_backup_transition' : bool, default = True
          Use the ortho↔merge pipeline implemented as ``_smood_ortho_merge_backup_pipeline``
          (:mod:`pymesh2d.ortho_merge.ortho_merge_iter`). If False, falls back to the inline V3 ortho+merge loop.
        - 'require_both_criteria' : bool, default = False
          If True, after the main ortho-merge cycles, require dual criteria on the merge-consistent
          triangle proxy and run recovery (may raise ``RuntimeError``). **False** matches the backup
          snapshot (no global check). *Not* a threshold — use ``smalllink_threshold`` for 0.11.
        - 'max_recovery_iterations' : int, default = 25
          Extra ortho+merge cycles when ``require_both_criteria`` is True and checks fail.
        - 'recovery_stagnation_break' : int, default = 3
          Stop recovery when ``(max|cosφ|, n_small)`` is unchanged for this many consecutive
          recovery cycles (0 = disabled). Still raises if criteria are unmet.
        - 'preserve_merged_quads' : bool, default = False
          If True, store mixed face-node rows on ``opts['_mixed_face_nodes_0b']`` for UGRID export.
        - 'disp' : int or float, default = 4
          Display frequency for iteration progress. Set to `np.inf` for quiet execution.
    hfun : callable, optional
        Mesh-size function used for local edge-length control.
    harg : tuple, optional
        Additional arguments passed to the mesh-size function `hfun`.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Updated vertex coordinates after smoothing.
    conn : ndarray of shape (E, 2)
        Updated constrained edges.
    tria : ndarray of shape (T, 3)
        Updated triangle connectivity.
    tnum : ndarray of shape (T, 1)
        Updated part indices.

    Notes
    -----
    Default path delegates to :mod:`pymesh2d.ortho_merge.ortho_merge_iter` (orthogonalize on a
    merge-consistent triangle proxy, then ``merge_circumcenters``). See MeshKernel /
    Delft3D-FM references in :mod:`pymesh2d.ortho_merge.meshkernel_orthogonalize_3`.
    """

    if vert is None:
        vert = np.empty((0, 2))
    if conn is None:
        conn = np.empty((0, 2), dtype=int)
    if tria is None:
        tria = np.empty((0, 3), dtype=int)
    if tnum is None:
        tnum = np.empty((0, 1), dtype=int)
    if opts is None:
        opts = {}

    opts = makeopt_smood(opts)

    # ---------------------------------------------- default CONN
    if conn.size == 0:
        edge, _ = tricon(tria)
        ebnd = edge[:, 3] < 1  # use boundary edge
        conn = edge[ebnd, 0:2]

    # ---------------------------------------------- default TNUM
    if tnum.size == 0:
        tnum = np.ones((tria.shape[0], 1), dtype=int)

    # ---------------------------------------------- basic checks
    if not (
        isinstance(vert, np.ndarray)
        and isinstance(conn, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(tnum, np.ndarray)
        and isinstance(opts, dict)
    ):
        raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    nvrt = vert.shape[0]

    if np.min(conn[:, :2]) < 0 or np.max(conn[:, :2]) > nvrt:
        raise ValueError("smood:invalidInputs - Invalid CONN input array.")

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) > nvrt:
        raise ValueError("smood:invalidInputs - Invalid TRIA input array.")

    # ---------------------------------------------- output title
    if not np.isinf(opts["disp"]):
        print("\n Smooth triangulation for Delft3D-FM computation...\n")


    # Ortho ↔ merge (default ``use_backup_transition``) or inline V3 tria loop.
    return _smood_v3_ortho_merge(vert, conn, tria, tnum, opts)


def makeopt_smood(opts=None):
    """
    Initialize the options structure for the `smood` function.

    Parameters
    ----------
    opts : dict or None
        User-defined options dictionary. If None, a new dictionary is created.

    Returns
    -------
    opts : dict
        Options dictionary completed with default values for missing parameters.
    """
    if opts is None:
        opts = {}

    # --------------------------- ITER
    if "iter" not in opts:
        # Default pipeline: ortho <-> merge 4 outer cycles (matches prior `code.py` opts_smood).
        opts["iter"] = 4
    else:
        if not isinstance(opts["iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - Invalid OPT.ITER selection.")

    # --------------------------- INNER_ITER
    if "inner_iter" not in opts:
        opts["inner_iter"] = 4
    else:
        if not isinstance(opts["inner_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["inner_iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - Invalid OPT.INNER_ITER selection.")

    # --------------------------- ORTHO_FACTOR
    if "ortho_factor" not in opts:
        opts["ortho_factor"] = 0.5
    else:
        if not isinstance(opts["ortho_factor"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if not (0.0 <= opts["ortho_factor"] <= 1.0):
            raise ValueError("smood:invalidOptionValues - ORTHO_FACTOR must be in [0, 1].")

    # --------------------------- RELAXATION
    if "relaxation" not in opts:
        opts["relaxation"] = 0.75
    else:
        if not isinstance(opts["relaxation"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if not (0.0 < opts["relaxation"] <= 1.0):
            raise ValueError("smood:invalidOptionValues - RELAXATION must be in (0, 1].")

    # --------------------------- DISP
    if "disp" not in opts:
        opts["disp"] = 8
    else:
        if not isinstance(opts["disp"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["disp"] <= 0:
            raise ValueError("smood:invalidOptionValues - Invalid OPT.DISP selection.")

    # --------------------------- VTOL
    if "vtol" not in opts:
        opts["vtol"] = 1.0e-3
    else:
        if not isinstance(opts["vtol"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["vtol"] <= 0:
            raise ValueError("smood:invalidOptionValues - Invalid OPT.VTOL selection.")

    # --------------------------- DBUG
    if "dbug" not in opts:
        opts["dbug"] = False
    else:
        if not isinstance(opts["dbug"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    # --------------------------- USE_SMALLLINK
    if "use_smalllink" not in opts:
        opts["use_smalllink"] = True
    else:
        if not isinstance(opts["use_smalllink"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    # --------------------------- CONVERGE_ON_SMALLLINKS
    if "converge_on_smalllinks" not in opts:
        opts["converge_on_smalllinks"] = True  # Continue until no small links remain
    else:
        if not isinstance(opts["converge_on_smalllinks"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
    
    # --------------------------- CONVERGE_ON_ORTHOGONALITY
    if "converge_on_orthogonality" not in opts:
        opts["converge_on_orthogonality"] = True  # Continue until internal constrained edges have good orthogonality
    else:
        if not isinstance(opts["converge_on_orthogonality"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
    
    # --------------------------- ORTHOGONALITY_THRESHOLD
    if "orthogonality_threshold" not in opts:
        opts["orthogonality_threshold"] = 0.49  # max|cosphi|
    else:
        if not isinstance(opts["orthogonality_threshold"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if not (0.0 <= opts["orthogonality_threshold"] <= 1.0):
            raise ValueError("smood:invalidOptionValues - ORTHOGONALITY_THRESHOLD must be in [0, 1].")

    # --------------------------- SMALLLINK_THRESHOLD
    if "smalllink_threshold" not in opts:
        opts["smalllink_threshold"] = 0.11
    else:
        if not isinstance(opts["smalllink_threshold"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["smalllink_threshold"] = float(opts["smalllink_threshold"])

    # --------------------------- BUFFER_LAYERS
    if "buffer_layers" not in opts:
        opts["buffer_layers"] = 2
    else:
        if not isinstance(opts["buffer_layers"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["buffer_layers"] = int(opts["buffer_layers"])
        if opts["buffer_layers"] <= 0:
            raise ValueError("smood:invalidOptionValues - buffer_layers must be > 0.")

    # --------------------------- ENABLE_EDGE_FLIPS
    if "enable_edge_flips" not in opts:
        opts["enable_edge_flips"] = True
    else:
        if not isinstance(opts["enable_edge_flips"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    # --------------------------- MAX_GLOBAL_ITER / SMOOTH_ITER
    if "max_global_iter" not in opts:
        opts["max_global_iter"] = int(opts["inner_iter"]) + 2
    else:
        if not isinstance(opts["max_global_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["max_global_iter"] = int(opts["max_global_iter"])
        if opts["max_global_iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - max_global_iter must be > 0.")

    if "smooth_iter" not in opts:
        opts["smooth_iter"] = int(opts["inner_iter"]) * 4
    else:
        if not isinstance(opts["smooth_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["smooth_iter"] = int(opts["smooth_iter"])
        if opts["smooth_iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - smooth_iter must be > 0.")
    
    # --------------------------- ALLOW_CONSTRAINT_SLIDING_JUNCTIONS (check first)
    if "allow_constraint_sliding_junctions" not in opts:
        opts["allow_constraint_sliding_junctions"] = False  # Don't allow sliding at junctions by default
    else:
        if not isinstance(opts["allow_constraint_sliding_junctions"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
    
    # --------------------------- ALLOW_CONSTRAINT_SLIDING
    if "allow_constraint_sliding" not in opts:
        opts["allow_constraint_sliding"] = False  # Allow vertices on constraint lines to slide along lines
    else:
        if not isinstance(opts["allow_constraint_sliding"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
    
    # If allow_constraint_sliding_junctions is True, automatically enable allow_constraint_sliding
    if opts["allow_constraint_sliding_junctions"]:
        opts["allow_constraint_sliding"] = True

    # --------------------------- MAX_FINAL_SMALLLINK_ITER
    if "max_final_smalllink_iter" not in opts:
        opts["max_final_smalllink_iter"] = 15  # Increased default for better convergence
    else:
        if not isinstance(opts["max_final_smalllink_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["max_final_smalllink_iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - MAX_FINAL_SMALLLINK_ITER must be > 0.")
        opts["max_final_smalllink_iter"] = int(opts["max_final_smalllink_iter"])

    # --------------------------- SMALLLINK_ITER_START (automatic if not defined)
    if "smalllink_iter_start" not in opts:
        # Automatically set to half of iterations
        opts["smalllink_iter_start"] = int(opts["iter"]) // 2
    else:
        if not isinstance(opts["smalllink_iter_start"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["smalllink_iter_start"] < 0:
            raise ValueError("smood:invalidOptionValues - SMALLLINK_ITER_START must be >= 0.")
        opts["smalllink_iter_start"] = int(opts["smalllink_iter_start"])

    # --------------------------- SMALLLINK_ITER_FREQ (automatic if not defined)
    if "smalllink_iter_freq" not in opts:
        # Automatically set based on number of iterations
        # For fewer iterations: more frequent, for more iterations: less frequent
        n_iter = int(opts["iter"])
        if n_iter <= 10:
            opts["smalllink_iter_freq"] = 2
        elif n_iter <= 20:
            opts["smalllink_iter_freq"] = 3
        else:
            opts["smalllink_iter_freq"] = max(3, n_iter // 8)  # Approximately every 8th iteration
    else:
        if not isinstance(opts["smalllink_iter_freq"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["smalllink_iter_freq"] <= 0:
            raise ValueError("smood:invalidOptionValues - SMALLLINK_ITER_FREQ must be > 0.")
        opts["smalllink_iter_freq"] = int(opts["smalllink_iter_freq"])

    # --------------------------- USE_BACKUP_TRANSITION (ortho_merge_iterate_tria path)
    if "use_backup_transition" not in opts:
        opts["use_backup_transition"] = True
    else:
        if not isinstance(opts["use_backup_transition"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    # --------------------------- REQUIRE_BOTH_CRITERIA (fan-proxy dual check + recovery)
    if "require_both_criteria" not in opts:
        opts["require_both_criteria"] = BACKUP_ORTHO_MERGE_REQUIRE_STRICT_DUAL
    else:
        if not isinstance(opts["require_both_criteria"], bool):
            raise TypeError("smood:incorrectInputClass - require_both_criteria must be bool.")

    if "max_recovery_iterations" not in opts:
        opts["max_recovery_iterations"] = 25
    else:
        if not isinstance(opts["max_recovery_iterations"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["max_recovery_iterations"] = int(opts["max_recovery_iterations"])
        if opts["max_recovery_iterations"] < 0:
            raise ValueError("smood:invalidOptionValues - max_recovery_iterations must be >= 0.")

    if "recovery_stagnation_break" not in opts:
        opts["recovery_stagnation_break"] = 3
    else:
        if not isinstance(opts["recovery_stagnation_break"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["recovery_stagnation_break"] = int(opts["recovery_stagnation_break"])
        if opts["recovery_stagnation_break"] < 0:
            raise ValueError("smood:invalidOptionValues - recovery_stagnation_break must be >= 0.")

    if "preserve_merged_quads" not in opts:
        opts["preserve_merged_quads"] = True
    else:
        if not isinstance(opts["preserve_merged_quads"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    return opts
