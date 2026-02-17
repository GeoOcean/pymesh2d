"""
Mesh smoothing with orthogonalization.

This module implements a mesh smoothing algorithm with orthogonalization,
combining aspect ratio optimization with angle-based smoothing for improved
mesh quality in flow simulations.
"""

import time
import warnings

import numpy as np
from scipy.sparse import csr_matrix

from .mesh_cost.triscr import triscr
from .mesh_util.circo import fix_small_flow_links, small_flow_links
from .mesh_util.deltri import deltri
from .mesh_util.setset import setset
from .mesh_util.tricon import tricon

warnings.filterwarnings('ignore', category=RuntimeWarning)


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
        - 'use_smalllink' : bool, default = True
          Whether to apply fix_small_flow_links during iterations and at the end.
        - 'converge_on_smalllinks' : bool, default = True
          If True, continue iterating until no small flow links remain (or max iterations reached).
          If False, converge only based on vertex movement tolerance.
        - 'converge_on_orthogonality' : bool, default = True
          If True, continue iterating until internal constrained edges have acceptable orthogonality.
          If False, converge only based on vertex movement tolerance and small links.
        - 'orthogonality_threshold' : float, default = 0.3
          Maximum acceptable orthogonality value (|cos(angle)|) for internal constrained edges.
          Lower values are better (0.0 = perfect orthogonality, 1.0 = parallel).
          Edges with orthogonality > threshold are considered "poor" and prevent convergence.
        - 'allow_constraint_sliding' : bool, default = False
          If True, allow vertices on internal constrained edges to slide along their constraint lines
          to improve orthogonality. Vertices at junctions (on multiple constraint lines) remain fixed
          unless allow_constraint_sliding_junctions=True.
        - 'allow_constraint_sliding_junctions' : bool, default = False
          If True, automatically enables allow_constraint_sliding and also allows vertices at junctions
          (on multiple constraint lines) to slide along constraint lines. For junctions, the vertex is
          projected onto the closest constraint line. If False, only vertices on exactly one constraint
          line can slide (if allow_constraint_sliding=True).
        - 'max_final_smalllink_iter' : int, default = 10
          Maximum number of iterations for final small link correction phase.
        - 'smalllink_iter_start' : int, optional
          Iteration at which to start applying small flow link fixes.
          If not provided, automatically set to iter // 2 (halfway through iterations).
        - 'smalllink_iter_freq' : int, optional
          Frequency of small flow link fixes (every N iterations).
          If not provided, automatically set based on number of iterations:
          - iter <= 10: every 2 iterations
          - iter <= 20: every 3 iterations
          - iter > 20: approximately every iter/8 iterations
          Applied every iteration in the last 3 iterations regardless of frequency.
        - 'smalllink_threshold' : float, default = 0.11
          Threshold for small flow links (passed to fix_small_flow_links as removesmalllinkstrsh).
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
    This routine implements a Delft3D-inspired approach that:
    1. Computes orthogonalization weights based on edge aspect ratios
    2. Computes smoothing weights based on angle optimization
    3. Combines both contributions with a weighted factor
    4. Uses iterative refinement with relaxation for stability
    5. Optimized for performance using vectorized numpy operations
    
    Performance optimizations:
    - Weights are computed once per outer iteration (not per inner iteration)
    - Adaptive ortho_factor (mu) starts small and increases progressively
    - Small flow link fixes are applied conditionally (not every iteration)
    - Vectorized operations replace Python loops where possible
    
    The ortho_factor adaptively increases from a small initial value (0.01 * max)
    to the maximum specified value, similar to Delft3D's mu parameter.

    References
    ----------
    Inspired by MeshKernel's OrthogonalizationAndSmoothing implementation.
    See: https://github.com/Deltares/MeshKernel
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
        print(" -------------------------------------------------------")
        print("      |ITER.|          |MOVE(X)|          |LINK(X)|     ")
        print(" -------------------------------------------------------")

    # ---------------------------------------------- polygon bounds
    node = vert.copy()
    PSLG = conn.copy()
    pmax = int(np.max(tnum))
    part = [None for _ in range(pmax)]

    for ppos in range(pmax):
        tsel = tnum.flatten() == (ppos + 1)
        tcur = tria[tsel, :]
        ecur, tcur = tricon(tcur)
        ebnd = ecur[:, 3] == -1
        same, _ = setset(PSLG, ecur[ebnd, 0:2])
        part[ppos] = np.where(same)[0]

    # ---------------------------------------------- DO MESH ITER
    tnow = time.time()
    tcpu = {
        "full": 0.0,
        "ortho": 0.0,
        "smooth": 0.0,
        "solve": 0.0,
        "smalllink": 0.0,
    }

    ortho_factor_max = opts["ortho_factor"]
    ortho_factor = min(0.05, ortho_factor_max * 0.15)
    relaxation = opts["relaxation"]
    
    smalllink_iter_start = int(opts["smalllink_iter_start"])
    smalllink_iter_freq = int(opts["smalllink_iter_freq"])

    for outer_iter in range(int(opts["iter"])):
        # ---------------------------------------------- rebuild connectivity
        edge, tria_6col = tricon(tria, conn)
        nvrt = vert.shape[0]
        nedg = edge.shape[0]

        # Build vertex-edge incidence matrix
        IMAT = csr_matrix(
            (np.ones(nedg), (edge[:, 0], np.arange(nedg))), shape=(nvrt, nedg)
        )
        JMAT = csr_matrix(
            (np.ones(nedg), (edge[:, 1], np.arange(nedg))), shape=(nvrt, nedg)
        )
        EMAT = IMAT + JMAT
        vdeg = np.array(EMAT.sum(axis=1)).flatten()

        allow_constraint_sliding = opts.get("allow_constraint_sliding", False)
        allow_junctions = opts.get("allow_constraint_sliding_junctions", False)
        
        # If allow_constraint_sliding_junctions is True, automatically enable allow_constraint_sliding
        if allow_junctions:
            allow_constraint_sliding = True
        
        free_vertices = compute_free_vertices(nvrt, conn, part, allow_constraint_sliding)
        vold = vert.copy()
        oscr = triscr(vert, tria)

        # ---------------------------------------------- compute weights
        # Evaluate hfun first (needed for both orthogonalization and smoothing)
        hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
        
        ttic_ortho = time.time()
        ortho_weights, ortho_rhs = compute_orthogonalization_weights(
            vert, edge, tria, tria_6col, free_vertices, part, conn, hvrt
        )
        tcpu["ortho"] += time.time() - ttic_ortho

        ttic_smooth = time.time()
        smooth_weights = compute_smoothing_weights(
            vert, edge, EMAT, vdeg, hvrt
        )
        tcpu["smooth"] += time.time() - ttic_smooth

        # ---------------------------------------------- inner iterations
        smooth_factor = 1.0 - ortho_factor
        
        for inner_iter in range(opts["inner_iter"]):
            ttic_solve = time.time()

            vnew = combine_and_solve(
                vert,
                edge,
                EMAT,
                ortho_weights,
                ortho_rhs,
                smooth_weights,
                ortho_factor,
                smooth_factor,
                free_vertices,
                conn,
                part,
            )

            vnew = relaxation * vnew + (1.0 - relaxation) * vert

            # Project vertices on constraint lines
            if allow_constraint_sliding and conn is not None and len(conn) > 0 and part is not None:
                external_vertex_set = set()
                for p in part:
                    if p is not None and len(p) > 0:
                        part_edges = conn[p, :]
                        for e in part_edges:
                            external_vertex_set.add(int(e[0]))
                            external_vertex_set.add(int(e[1]))
                
                external_vertex_array = np.array(list(external_vertex_set))
                if len(external_vertex_array) > 0:
                    vnew[external_vertex_array, :] = vert[external_vertex_array, :]
                
                # Count constraint lines per vertex to identify junctions
                conn_sorted = np.sort(conn, axis=1)
                external_edge_set = set()
                for p in part:
                    if p is not None and len(p) > 0:
                        part_edges = conn[p, :]
                        part_edges_sorted = np.sort(part_edges, axis=1)
                        for e in part_edges_sorted:
                            external_edge_set.add(tuple(e))
                
                vertex_constraint_count = {}
                internal_constrained_vertices = set()
                for e in conn_sorted:
                    edge_tuple = tuple(e)
                    if edge_tuple not in external_edge_set:
                        v1_idx = int(e[0])
                        v2_idx = int(e[1])
                        internal_constrained_vertices.add(v1_idx)
                        internal_constrained_vertices.add(v2_idx)
                        vertex_constraint_count[v1_idx] = vertex_constraint_count.get(v1_idx, 0) + 1
                        vertex_constraint_count[v2_idx] = vertex_constraint_count.get(v2_idx, 0) + 1
                
                for v_idx in internal_constrained_vertices:
                    if v_idx < len(vnew) and v_idx not in external_vertex_set:
                        is_junction = vertex_constraint_count.get(v_idx, 0) > 1
                        if not is_junction or allow_junctions:
                            vnew[v_idx, :] = project_vertex_on_constraint_line(
                                v_idx, vnew[v_idx, :], conn, part, vert, allow_junctions
                            )
            else:
                # Fix all constrained vertices if sliding is not allowed
                if conn is not None and len(conn) > 0:
                    vnew[conn.flatten(), :] = vert[conn.flatten(), :]
            
            # Only fix vertices that are truly fixed (external boundaries or constrained without sliding)
            vnew[~free_vertices, :] = vert[~free_vertices, :]

            vert = vnew

            tcpu["solve"] += time.time() - ttic_solve

        # ---------------------------------------------- fix small flow links
        if opts.get("use_smalllink", True):
            should_fix = False
            
            if outer_iter >= smalllink_iter_start:
                iter_remaining = int(opts["iter"]) - outer_iter
                if iter_remaining <= 3:
                    should_fix = True
                elif outer_iter % smalllink_iter_freq == 0:
                    should_fix = True
            
            if should_fix:
                ttic_smalllink = time.time()
                try:
                    if "removesmalllinkstrsh" not in opts:
                        opts["removesmalllinkstrsh"] = opts.get("smalllink_threshold", 0.1)
                    
                    vert, conn, tria, tnum = fix_small_flow_links(
                        vert, conn, tria, tnum, node, PSLG, part, opts
                    )
                    vold = vert.copy()
                    edge, tria_6col = tricon(tria, conn)
                    nvrt = vert.shape[0]
                    nedg = edge.shape[0]
                    IMAT = csr_matrix(
                        (np.ones(nedg), (edge[:, 0], np.arange(nedg))), shape=(nvrt, nedg)
                    )
                    JMAT = csr_matrix(
                        (np.ones(nedg), (edge[:, 1], np.arange(nedg))), shape=(nvrt, nedg)
                    )
                    EMAT = IMAT + JMAT
                    vdeg = np.array(EMAT.sum(axis=1)).flatten()
                    free_vertices = compute_free_vertices(nvrt, conn, part, allow_constraint_sliding)
                    
                    # Recalculate hvrt after mesh modification
                    ttic_ortho_recomp = time.time()
                    hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
                    ortho_weights, ortho_rhs = compute_orthogonalization_weights(
                        vert, edge, tria, tria_6col, free_vertices, part, conn, hvrt
                    )
                    tcpu["ortho"] += time.time() - ttic_ortho_recomp
                    
                    ttic_smooth_recomp = time.time()
                    smooth_weights = compute_smoothing_weights(
                        vert, edge, EMAT, vdeg, hvrt
                    )
                    tcpu["smooth"] += time.time() - ttic_smooth_recomp
                    
                except Exception as e:
                    if opts.get("dbug", False):
                        print(f"Warning: fix_small_flow_links failed at iter {outer_iter}: {e}")
                tcpu["smalllink"] += time.time() - ttic_smalllink

        # ---------------------------------------------- update ortho_factor
        progress = (outer_iter + 1) / int(opts["iter"])
        
        if progress < 0.5:
            growth_factor = 2.5
            ortho_factor = min(growth_factor * ortho_factor, ortho_factor_max)
        else:
            target_factor = ortho_factor_max * (0.5 + 0.5 * (progress - 0.5) / 0.5)
            ortho_factor = min(max(ortho_factor * 1.2, target_factor), ortho_factor_max)

        # ---------------------------------------------- check convergence
        n_smalllinks = 0
        if opts.get("use_smalllink", True) and opts.get("converge_on_smalllinks", True):
            try:
                edge_check, tria_6col_check = tricon(tria, conn)
                removesmalllinkstrsh = opts.get("removesmalllinkstrsh", opts.get("smalllink_threshold", 0.1))
                n_smalllinks, _ = small_flow_links(
                    vert, tria, edge_check, removesmalllinkstrsh, conn, tria_6col_check
                )
            except Exception:
                n_smalllinks = -1
        
        n_poor_ortho = 0
        if opts.get("converge_on_orthogonality", True):
            try:
                ortho_threshold = opts.get("orthogonality_threshold", 0.3)
                n_poor_ortho = check_internal_constrained_orthogonality(
                    vert, edge, tria, conn, part, ortho_threshold
                )
            except Exception:
                n_poor_ortho = -1

        # ---------------------------------------------- test convergence
        # Recalculate hvrt if mesh was modified (vertex count changed)
        if hvrt.shape[0] != vert.shape[0]:
            hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
        
        if vert.shape[0] == vold.shape[0]:
            vdel = np.sum((vert - vold) ** 2, axis=1)
        else:
            n_common = min(vert.shape[0], vold.shape[0])
            vdel = np.sum((vert[:n_common, :] - vold[:n_common, :]) ** 2, axis=1)
            if vert.shape[0] > vold.shape[0]:
                vdel = np.concatenate([vdel, np.zeros(vert.shape[0] - vold.shape[0])])
            elif vold.shape[0] > vert.shape[0]:
                vdel = np.concatenate([vdel, np.zeros(vold.shape[0] - vert.shape[0])])

        # Ensure hvrt matches vdel shape
        hvrt_flat = hvrt.flatten()
        if hvrt_flat.shape[0] != vdel.shape[0]:
            # Recalculate if still mismatched
            hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
            hvrt_flat = hvrt.flatten()
        
        vdel_norm = vdel / (hvrt_flat ** 2 + np.finfo(float).eps)

        move = vdel_norm > opts["vtol"] ** 2
        nmov = np.count_nonzero(move)

        nscr = triscr(vert, tria)
        
        # Display small links count (or -1 if not computed)
        n_smalllinks_display = n_smalllinks if n_smalllinks >= 0 else -1

        if outer_iter % opts["disp"] == 0:
            print(f"{outer_iter:11d} {nmov:18d} {n_smalllinks_display:18d}")

        # ---------------------------------------------- loop convergence
        converged = nmov == 0
        if converged and opts.get("converge_on_smalllinks", True) and n_smalllinks > 0:
            converged = False
        if converged and opts.get("converge_on_orthogonality", True) and n_poor_ortho > 0:
            converged = False
            if outer_iter % 2 == 0:
                try:
                    if "removesmalllinkstrsh" not in opts:
                        opts["removesmalllinkstrsh"] = opts.get("smalllink_threshold", 0.11)
                    vert, conn, tria, tnum = fix_small_flow_links(
                        vert, conn, tria, tnum, node, PSLG, part, opts
                    )
                    vold = vert.copy()
                    edge, tria_6col = tricon(tria, conn)
                    nvrt = vert.shape[0]
                    nedg = edge.shape[0]
                    IMAT = csr_matrix(
                        (np.ones(nedg), (edge[:, 0], np.arange(nedg))), shape=(nvrt, nedg)
                    )
                    JMAT = csr_matrix(
                        (np.ones(nedg), (edge[:, 1], np.arange(nedg))), shape=(nvrt, nedg)
                    )
                    EMAT = IMAT + JMAT
                    vdeg = np.array(EMAT.sum(axis=1)).flatten()
                    free_vertices = compute_free_vertices(nvrt, conn, part, allow_constraint_sliding)
                    
                    # Recalculate hvrt after mesh modification
                    hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
                    ortho_weights, ortho_rhs = compute_orthogonalization_weights(
                        vert, edge, tria, tria_6col, free_vertices, part, conn, hvrt
                    )
                    smooth_weights = compute_smoothing_weights(
                        vert, edge, EMAT, vdeg, hvrt
                    )
                except Exception:
                    pass
        
        if converged:
            break

    # ---------------------------------------------- final small link fix
    if opts.get("use_smalllink", True):
        ttic_smalllink = time.time()
        try:
            if "removesmalllinkstrsh" not in opts:
                opts["removesmalllinkstrsh"] = opts.get("smalllink_threshold", 0.11)
            
            removesmalllinkstrsh = opts["removesmalllinkstrsh"]
            max_final_iter = opts.get("max_final_smalllink_iter", 15)
            
            edge_final, tria_6col_final = tricon(tria, conn)
            
            n_smalllinks_final, _ = small_flow_links(
                vert, tria, edge_final, removesmalllinkstrsh, conn, tria_6col_final
            )
            
            if n_smalllinks_final > 0 and opts.get("converge_on_smalllinks", True):
                if opts.get("dbug", False):
                    print(f"\nFinal small link correction: starting with {n_smalllinks_final} small links")
                
                vert_before_final = vert.copy()
                final_opts = opts.copy()
                
                if n_smalllinks_final <= 3:
                    final_opts["max_fix_iter"] = max(30, max_final_iter * 3)
                else:
                    final_opts["max_fix_iter"] = min(20, max_final_iter * 2)
                
                prev_nlinks = n_smalllinks_final
                no_improvement_count = 0
                max_no_improvement = 3 if n_smalllinks_final <= 3 else 2
                
                for final_iter in range(max_final_iter):
                    vert, conn, tria, tnum = fix_small_flow_links(
                        vert, conn, tria, tnum, node, PSLG, part, final_opts
                    )
                    
                    edge_final, tria_6col_final = tricon(tria, conn)
                    n_smalllinks_final, _ = small_flow_links(
                        vert, tria, edge_final, removesmalllinkstrsh, conn, tria_6col_final
                    )
                    
                    if n_smalllinks_final == 0:
                        break
                    elif n_smalllinks_final >= prev_nlinks:
                        no_improvement_count += 1
                        if no_improvement_count >= max_no_improvement and n_smalllinks_final > 1:
                            break
                    else:
                        no_improvement_count = 0
                        prev_nlinks = n_smalllinks_final
                    
                    if n_smalllinks_final == 1 and final_iter < max_final_iter - 1:
                        continue
                
                if opts.get("dbug", False):
                    n_vert_before = vert_before_final.shape[0]
                    n_vert_after = vert.shape[0]
                    print(f"Final fix_small_flow_links: {n_vert_before} -> {n_vert_after} vertices, "
                          f"{n_smalllinks_final} small links remaining after {final_iter + 1} iterations")
            elif n_smalllinks_final > 0:
                vert, conn, tria, tnum = fix_small_flow_links(
                    vert, conn, tria, tnum, node, PSLG, part, opts
                )
        except Exception as e:
            if opts.get("dbug", False):
                print(f"Warning: final fix_small_flow_links failed: {e}")
        tcpu["smalllink"] += time.time() - ttic_smalllink

    tcpu["full"] += time.time() - tnow

    if opts["dbug"]:
        print("\n Mesh smoothing timer...\n")
        print(f" FULL: {tcpu['full']:.6f}")
        print(f" ORTHO: {tcpu['ortho']:.6f}")
        print(f" SMOOTH: {tcpu['smooth']:.6f}")
        print(f" SOLVE: {tcpu['solve']:.6f}")
        print(f" SMALLLINK: {tcpu['smalllink']:.6f}\n")

    if not np.isinf(opts["disp"]):
        print("")

    return vert, conn, tria, tnum


def compute_free_vertices(nvrt, conn, part, allow_constraint_sliding=False):
    """
    Compute which vertices are free to move.
    
    Vertices on external boundaries (part) are always fixed.
    Vertices on internal constrained edges are free if allow_constraint_sliding=True.
    
    Parameters
    ----------
    nvrt : int
        Number of vertices.
    conn : ndarray of shape (E_conn, 2)
        All constrained edges (PSLG).
    part : list of ndarray
        List of edge indices in conn that define external boundaries.
    allow_constraint_sliding : bool, default = False
        If True, vertices on internal constrained edges are considered free (can slide along lines).
        If False, all constrained vertices are fixed.
    
    Returns
    -------
    free_vertices : ndarray of shape (nvrt,)
        Boolean array indicating free (movable) vertices.
    """
    free_vertices = np.ones(nvrt, dtype=bool)
    if part is not None and conn is not None and len(conn) > 0:
        # Always fix vertices on external boundaries
        external_vertex_set = set()
        for p in part:
            if p is not None and len(p) > 0:
                part_edges = conn[p, :]
                for e in part_edges:
                    external_vertex_set.add(int(e[0]))
                    external_vertex_set.add(int(e[1]))
        external_vertex_array = np.array(list(external_vertex_set))
        if len(external_vertex_array) > 0:
            free_vertices[external_vertex_array] = False
        
        # Handle internal constrained vertices based on allow_constraint_sliding
        if not allow_constraint_sliding:
            # Fix all constrained vertices if sliding is not allowed
            external_edge_set = set()
            for p in part:
                if p is not None and len(p) > 0:
                    part_edges = conn[p, :]
                    part_edges_sorted = np.sort(part_edges, axis=1)
                    for e in part_edges_sorted:
                        external_edge_set.add(tuple(e))
            
            conn_sorted = np.sort(conn, axis=1)
            internal_constrained_vertices = set()
            for e in conn_sorted:
                edge_tuple = tuple(e)
                if edge_tuple not in external_edge_set:
                    internal_constrained_vertices.add(int(e[0]))
                    internal_constrained_vertices.add(int(e[1]))
            
            internal_vertex_array = np.array(list(internal_constrained_vertices))
            if len(internal_vertex_array) > 0:
                free_vertices[internal_vertex_array] = False
    else:
        # If no part info, fix all constrained vertices if sliding is not allowed
        if not allow_constraint_sliding and conn is not None and len(conn) > 0:
            free_vertices[conn.flatten()] = False
    return free_vertices


def check_internal_constrained_orthogonality(vert, edge, tria, conn, part, ortho_threshold=0.3):
    """
    Check orthogonality of internal constrained edges (edges in conn but not in part).
    
    Returns the number of internal constrained edges with poor orthogonality.
    
    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Vertex coordinates.
    edge : ndarray of shape (E, 5)
        Edge connectivity from tricon.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    conn : ndarray of shape (E_conn, 2)
        All constrained edges (PSLG).
    part : list of ndarray
        List of edge indices in conn that define external boundaries.
    ortho_threshold : float, default = 0.3
        Maximum acceptable orthogonality value (|cos(angle)|).
        Lower values are better (0.0 = perfect orthogonality).
    
    Returns
    -------
    n_poor_ortho : int
        Number of internal constrained edges with orthogonality > threshold.
    """
    if conn is None or len(conn) == 0 or part is None:
        return 0
    
    external_edge_set = set()
    for p in part:
        if p is not None and len(p) > 0:
            part_edges = conn[p, :]
            part_edges_sorted = np.sort(part_edges, axis=1)
            for e in part_edges_sorted:
                external_edge_set.add(tuple(e))
    
    conn_sorted = np.sort(conn, axis=1)
    internal_constrained_set = set()
    for e in conn_sorted:
        edge_tuple = tuple(e)
        if edge_tuple not in external_edge_set:
            internal_constrained_set.add(edge_tuple)
    
    if len(internal_constrained_set) == 0:
        return 0
    
    edge_sorted = np.sort(edge[:, 0:2], axis=1)
    internal_edge_indices = []
    for i in range(edge.shape[0]):
        edge_tuple = tuple(edge_sorted[i, :])
        if edge_tuple in internal_constrained_set:
            if edge[i, 3] > 0:
                internal_edge_indices.append(i)
    
    if len(internal_edge_indices) == 0:
        return 0
    
    n_poor_ortho = 0
    for e_idx in internal_edge_indices:
        v1_idx = int(edge[e_idx, 0])
        v2_idx = int(edge[e_idx, 1])
        v1 = vert[v1_idx, :]
        v2 = vert[v2_idx, :]
        
        t1_idx = int(edge[e_idx, 2])
        t2_idx = int(edge[e_idx, 3])
        
        tri1_verts = tria[t1_idx, :]
        tri2_verts = tria[t2_idx, :]
        
        def circumcenter_triangle(p1, p2, p3):
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            dx3 = p3[0] - p1[0]
            dy3 = p3[1] - p1[1]
            den = dy2 * dx3 - dy3 * dx2
            if abs(den) < 1e-12:
                return (p1 + p2 + p3) / 3.0
            z = (dx2 * (dx2 - dx3) + dy2 * (dy2 - dy3)) / den
            return np.array([
                p1[0] + 0.5 * (dx3 - z * dy3),
                p1[1] + 0.5 * (dy3 + z * dx3)
            ])
        
        cc1 = circumcenter_triangle(vert[tri1_verts[0], :], vert[tri1_verts[1], :], vert[tri1_verts[2], :])
        cc2 = circumcenter_triangle(vert[tri2_verts[0], :], vert[tri2_verts[1], :], vert[tri2_verts[2], :])
        
        if not (np.all(np.isfinite(cc1)) and np.all(np.isfinite(cc2))):
            continue
        
        edge_vec = v2 - v1
        cc_vec = cc2 - cc1
        
        edge_len_sq = np.sum(edge_vec**2)
        cc_len_sq = np.sum(cc_vec**2)
        
        if edge_len_sq > 0 and cc_len_sq > 0:
            cosphi = np.abs(np.dot(edge_vec, cc_vec) / np.sqrt(edge_len_sq * cc_len_sq))
            if np.isfinite(cosphi) and cosphi > ortho_threshold:
                n_poor_ortho += 1
    
    return n_poor_ortho


def compute_orthogonalization_weights(vert, edge, tria, tria_6col, free_vertices, part=None, conn=None, hvrt=None):
    """
    Compute orthogonalization weights based on edge aspect ratios.

    This function computes weights that encourage orthogonal edges by
    optimizing aspect ratios, similar to Delft3D's orthogonalizer.
    Uses a simplified but efficient approach for performance.
    Weights are normalized by mesh-size function to respect hfun constraints.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Vertex coordinates.
    edge : ndarray of shape (E, 5)
        Edge connectivity from tricon.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    tria_6col : ndarray of shape (T, 6)
        Triangle-to-edge mapping from tricon.
    free_vertices : ndarray of shape (V,)
        Boolean array indicating free (movable) vertices.
    part : list of ndarray, optional
        List of edge indices in conn that define external boundaries.
        Each element is an array of edge indices in conn (PSLG).
    conn : ndarray of shape (E_conn, 2), optional
        Array of all constrained edges (PSLG). Used to identify external
        boundary edges when part is provided.
    hvrt : ndarray of shape (V,), optional
        Mesh-size function values at vertices. If provided, weights are
        normalized to respect mesh-size constraints.

    Returns
    -------
    weights : scipy.sparse matrix of shape (V, V)
        Sparse representation of orthogonalization weights.
    rhs : ndarray of shape (V, 2)
        Right-hand side contributions for orthogonalization.
    """
    nvrt = vert.shape[0]
    nedg = edge.shape[0]

    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))
    elen = np.maximum(elen, np.finfo(float).eps)

    evec_norm = evec / elen[:, None]
    aspect_ratios = np.ones(nedg, dtype=np.float64)
    
    # Normalize weights by mesh-size function if provided
    h_weights = np.ones(nedg, dtype=np.float64)
    if hvrt is not None:
        hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])
        hmid = np.maximum(hmid, np.finfo(float).eps)
        hmid = np.where(np.isfinite(hmid), hmid, np.finfo(float).eps)
        # Weight inversely proportional to target edge length
        # Longer target edges get lower weight (less orthogonalization)
        h_weights = 1.0 / (hmid + np.finfo(float).eps)
        h_weights = np.where(np.isfinite(h_weights), h_weights, 1.0)
        # Normalize to avoid scaling issues
        h_weights = h_weights / (np.mean(h_weights) + np.finfo(float).eps)

    has_two_tri = edge[:, 3] > 0
    if np.any(has_two_tri):
        t1_idx = edge[has_two_tri, 2].astype(int)
        t2_idx = edge[has_two_tri, 3].astype(int)

        tri1 = tria[t1_idx, :]
        tri2 = tria[t2_idx, :]

        v1_t1 = vert[tri1[:, 0], :]
        v2_t1 = vert[tri1[:, 1], :]
        v3_t1 = vert[tri1[:, 2], :]
        area1 = 0.5 * np.abs(
            (v2_t1[:, 0] - v1_t1[:, 0]) * (v3_t1[:, 1] - v1_t1[:, 1])
            - (v3_t1[:, 0] - v1_t1[:, 0]) * (v2_t1[:, 1] - v1_t1[:, 1])
        )

        v1_t2 = vert[tri2[:, 0], :]
        v2_t2 = vert[tri2[:, 1], :]
        v3_t2 = vert[tri2[:, 2], :]
        area2 = 0.5 * np.abs(
            (v2_t2[:, 0] - v1_t2[:, 0]) * (v3_t2[:, 1] - v1_t2[:, 1])
            - (v3_t2[:, 0] - v1_t2[:, 0]) * (v2_t2[:, 1] - v1_t2[:, 1])
        )

        edge_idx = np.where(has_two_tri)[0]
        tri1_edges = tria_6col[t1_idx, 3:6]
        tri2_edges = tria_6col[t2_idx, 3:6]

        tri1_sum_len = np.sum(elen[tri1_edges], axis=1)
        tri2_sum_len = np.sum(elen[tri2_edges], axis=1)
        
        tri1_avg_len = (tri1_sum_len - elen[edge_idx]) / 2.0
        tri2_avg_len = (tri2_sum_len - elen[edge_idx]) / 2.0
        
        tri1_avg_len = np.maximum(tri1_avg_len, np.finfo(float).eps)
        tri2_avg_len = np.maximum(tri2_avg_len, np.finfo(float).eps)

        area_ratio = np.minimum(area1, area2) / np.maximum(area1, area2 + np.finfo(float).eps)
        len_ratio = np.minimum(tri1_avg_len, tri2_avg_len) / np.maximum(
            tri1_avg_len, tri2_avg_len + np.finfo(float).eps
        )
        aspect_ratios[has_two_tri] = area_ratio * len_ratio
        aspect_ratios[has_two_tri] = np.where(
            np.isfinite(aspect_ratios[has_two_tri]), 
            aspect_ratios[has_two_tri], 
            1.0
        )
        # Apply mesh-size function weighting
        aspect_ratios[has_two_tri] = aspect_ratios[has_two_tri] * h_weights[has_two_tri]

    is_external_boundary = np.zeros(nedg, dtype=bool)
    if part is not None and conn is not None and len(conn) > 0:
        external_edge_set = set()
        for p in part:
            if p is not None and len(p) > 0:
                part_edges = conn[p, :]
                part_edges_sorted = np.sort(part_edges, axis=1)
                for e in part_edges_sorted:
                    external_edge_set.add(tuple(e))
        
        edge_sorted = np.sort(edge[:, 0:2], axis=1)
        for i in range(nedg):
            edge_tuple = tuple(edge_sorted[i, :])
            if edge_tuple in external_edge_set:
                is_external_boundary[i] = True
    
    aspect_ratios[is_external_boundary] = 0.0

    row_indices = np.concatenate([edge[:, 0], edge[:, 1]])
    col_indices = np.concatenate([edge[:, 1], edge[:, 0]])
    data = np.concatenate([aspect_ratios, aspect_ratios])

    weight_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nvrt, nvrt))

    row_sums = np.array(weight_matrix.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, np.finfo(float).eps)
    inv_row_sums = 1.0 / row_sums
    weight_matrix = weight_matrix.multiply(inv_row_sums[:, np.newaxis])

    rhs = np.zeros((nvrt, 2))

    return weight_matrix, rhs


def compute_smoothing_weights(vert, edge, EMAT, vdeg, hvrt):
    """
    Compute smoothing weights based on edge length deviation from target.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Vertex coordinates.
    edge : ndarray of shape (E, 5)
        Edge connectivity from tricon.
    EMAT : scipy.sparse matrix
        Vertex-edge incidence matrix.
    vdeg : ndarray of shape (V,)
        Vertex degrees.
    hvrt : ndarray of shape (V,)
        Mesh-size function values at vertices.

    Returns
    -------
    weights : ndarray of shape (E,)
        Smoothing weights for each edge.
    """
    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))

    hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])
    hmid = np.maximum(hmid, np.finfo(float).eps)
    
    # Ensure no NaN or inf in calculations
    hmid = np.where(np.isfinite(hmid), hmid, np.finfo(float).eps)

    scal = elen / hmid
    scal = np.maximum(scal, np.finfo(float).eps)
    scal = np.where(np.isfinite(scal), scal, 1.0)
    
    weights = 1.0 / scal
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = weights / (np.sum(weights) + np.finfo(float).eps)

    return weights


def project_vertex_on_constraint_line(vertex_idx, new_pos, conn, part, vert, allow_junctions=False):
    """
    Project a vertex position onto its constraint line(s) if it's on an internal constrained edge.
    
    For vertices at junctions (on multiple constraint lines), projects onto the closest line
    or the intersection if allow_junctions=True.
    
    Parameters
    ----------
    vertex_idx : int
        Index of the vertex.
    new_pos : ndarray of shape (2,)
        Proposed new position.
    conn : ndarray of shape (E_conn, 2)
        All constrained edges (PSLG).
    part : list of ndarray
        List of edge indices in conn that define external boundaries.
    vert : ndarray of shape (V, 2)
        Current vertex coordinates.
    allow_junctions : bool, default = False
        If True, allow projection for vertices at junctions (on multiple constraint lines).
        For junctions, projects onto the closest constraint line.
        If False, only project vertices on exactly one constraint line (not at junctions).
    
    Returns
    -------
    projected_pos : ndarray of shape (2,)
        Projected position on the constraint line, or new_pos if not on internal constraint or at junction.
    """
    if conn is None or len(conn) == 0 or part is None:
        return new_pos
    
    external_edge_set = set()
    for p in part:
        if p is not None and len(p) > 0:
            part_edges = conn[p, :]
            part_edges_sorted = np.sort(part_edges, axis=1)
            for e in part_edges_sorted:
                external_edge_set.add(tuple(e))
    
    conn_sorted = np.sort(conn, axis=1)
    constraint_lines = []
    for e in conn_sorted:
        edge_tuple = tuple(e)
        if edge_tuple not in external_edge_set:
            if int(e[0]) == vertex_idx or int(e[1]) == vertex_idx:
                constraint_lines.append((int(e[0]), int(e[1])))
    
    if len(constraint_lines) == 0:
        return new_pos
    
    if not allow_junctions and len(constraint_lines) > 1:
        return vert[vertex_idx, :]
    
    # For junctions with allow_junctions=True, project onto the closest line
    if len(constraint_lines) > 1:
        min_dist_sq = np.inf
        best_projected = new_pos
        
        for v1_idx, v2_idx in constraint_lines:
            v1 = vert[v1_idx, :]
            v2 = vert[v2_idx, :]
            
            line_dir = v2 - v1
            line_len_sq = np.sum(line_dir**2)
            
            if line_len_sq < np.finfo(float).eps:
                continue
            
            to_point = new_pos - v1
            t = np.dot(to_point, line_dir) / line_len_sq
            projected = v1 + t * line_dir
            
            dist_sq = np.sum((new_pos - projected)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_projected = projected
        
        return best_projected
    
    # Single constraint line
    v1_idx, v2_idx = constraint_lines[0]
    v1 = vert[v1_idx, :]
    v2 = vert[v2_idx, :]
    
    line_dir = v2 - v1
    line_len_sq = np.sum(line_dir**2)
    
    if line_len_sq < np.finfo(float).eps:
        return vert[vertex_idx, :]
    
    to_point = new_pos - v1
    t = np.dot(to_point, line_dir) / line_len_sq
    projected_pos = v1 + t * line_dir
    
    return projected_pos


def combine_and_solve(
    vert,
    edge,
    EMAT,
    ortho_weights,
    ortho_rhs,
    smooth_weights,
    ortho_factor,
    smooth_factor,
    free_vertices,
    conn=None,
    part=None,
):
    """
    Combine orthogonalization and smoothing contributions and solve for new positions.

    Optimized version using vectorized operations for performance.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Current vertex coordinates.
    edge : ndarray of shape (E, 5)
        Edge connectivity.
    EMAT : scipy.sparse matrix
        Vertex-edge incidence matrix.
    ortho_weights : scipy.sparse matrix
        Orthogonalization weight matrix.
    ortho_rhs : ndarray of shape (V, 2)
        Orthogonalization right-hand side.
    smooth_weights : ndarray of shape (E,)
        Smoothing weights.
    ortho_factor : float
        Weight for orthogonalization (0-1).
    smooth_factor : float
        Weight for smoothing (0-1).
    free_vertices : ndarray of shape (V,)
        Boolean array indicating free vertices.
    conn : ndarray of shape (E_conn, 2), optional
        All constrained edges (PSLG). Used for projecting vertices onto constraint lines.
    part : list of ndarray, optional
        List of edge indices in conn that define external boundaries.

    Returns
    -------
    vnew : ndarray of shape (V, 2)
        New vertex coordinates.
    """
    nvrt = vert.shape[0]
    nedg = edge.shape[0]

    IMAT = csr_matrix(
        (np.ones(nedg), (edge[:, 0], np.arange(nedg))), shape=(nvrt, nedg)
    )
    JMAT = csr_matrix(
        (np.ones(nedg), (edge[:, 1], np.arange(nedg))), shape=(nvrt, nedg)
    )
    EMAT = IMAT + JMAT

    emid = 0.5 * (vert[edge[:, 0], :] + vert[edge[:, 1], :])
    weighted_emid = smooth_weights[:, None] * emid

    smooth_contrib = IMAT.dot(weighted_emid) + JMAT.dot(weighted_emid)
    smooth_sum = EMAT.dot(smooth_weights)

    if hasattr(smooth_contrib, 'toarray'):
        smooth_contrib = np.asarray(smooth_contrib.toarray())
    else:
        smooth_contrib = np.asarray(smooth_contrib)
    
    if hasattr(smooth_sum, 'toarray'):
        smooth_sum = np.asarray(smooth_sum.toarray()).flatten()
    else:
        smooth_sum = np.asarray(smooth_sum).flatten()

    smooth_sum_safe = np.maximum(smooth_sum, np.finfo(float).eps)
    
    if smooth_contrib.shape[0] != nvrt:
        if smooth_contrib.size == nvrt * 2:
            smooth_contrib = smooth_contrib.reshape(nvrt, 2)
        else:
            smooth_contrib = smooth_contrib[:nvrt, :]
    
    if smooth_sum_safe.shape[0] != nvrt:
        smooth_sum_safe = smooth_sum_safe[:nvrt]
    
    smooth_contrib = smooth_contrib / smooth_sum_safe[:, None]
    smooth_contrib = np.where(np.isfinite(smooth_contrib), smooth_contrib, 0.0)

    ortho_contrib = np.zeros((nvrt, 2))
    if ortho_factor > 0:
        ortho_result = ortho_weights.dot(vert)
        if hasattr(ortho_result, 'toarray'):
            ortho_contrib = ortho_result.toarray()
        else:
            ortho_contrib = np.asarray(ortho_result)
        ortho_contrib = np.where(np.isfinite(ortho_contrib), ortho_contrib, 0.0)

    vnew = smooth_factor * smooth_contrib + ortho_factor * ortho_contrib
    vnew += ortho_factor * ortho_rhs
    vnew = np.where(np.isfinite(vnew), vnew, vert)
    vnew[~free_vertices, :] = vert[~free_vertices, :]

    return vnew


def evalhfn(vert, edge, EMAT, hfun=None, harg=[]):
    """
    Evaluate the mesh spacing function at mesh vertices.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        XY coordinates of the mesh vertices.
    edge : ndarray of shape (E, 2)
        Array of edge connections.
    EMAT : scipy.sparse matrix
        Vertex–edge incidence matrix.
    hfun : float, callable, or None
        Mesh-size function or constant spacing value.
    harg : tuple
        Additional arguments passed to the mesh-size function `hfun`.

    Returns
    -------
    hvrt : ndarray of shape (N,)
        Mesh-size function values evaluated at the vertices.
    """
    # Fast path for scalar hfun
    if hfun is not None and np.isscalar(hfun):
        return hfun * np.ones(vert.shape[0], dtype=np.float64)
    
    # Compute default (mean edge length) only if needed
    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))
    elen = np.where(np.isfinite(elen), elen, 0.0)
    
    # Use sparse matrix multiplication (more efficient)
    default_hvrt = np.ravel(EMAT.dot(elen))
    vdeg_sum = np.ravel(EMAT.sum(axis=1))
    default_hvrt = np.where(
        vdeg_sum > np.finfo(float).eps,
        default_hvrt / np.maximum(vdeg_sum, np.finfo(float).eps),
        np.inf
    )
    default_hvrt = np.where(np.isfinite(default_hvrt), default_hvrt, np.inf)

    if hfun is not None and callable(hfun):
        try:
            hvrt = np.asarray(hfun(vert, *harg)).flatten()
            if hvrt.size != vert.shape[0]:
                raise ValueError(
                    "smood:evalhfn - hfun must return one value per vertex, "
                    f"got size {hvrt.size} for {vert.shape[0]} vertices."
                )
            bad = ~np.isfinite(hvrt) | (hvrt <= 0)
            hvrt = np.where(bad, default_hvrt, hvrt)
            hvrt = np.where(np.isfinite(hvrt), hvrt, default_hvrt)
        except Exception:
            hvrt = default_hvrt.copy()
    else:
        hvrt = default_hvrt.copy()

    return hvrt


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
        opts["iter"] = 16
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
        opts["disp"] = 4
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
        opts["orthogonality_threshold"] = 0.3  # Maximum acceptable |cos(angle)| for internal constrained edges
    else:
        if not isinstance(opts["orthogonality_threshold"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if not (0.0 <= opts["orthogonality_threshold"] <= 1.0):
            raise ValueError("smood:invalidOptionValues - ORTHOGONALITY_THRESHOLD must be in [0, 1].")
    
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

    return opts
