import numpy as np

def tricon(tt, cc=None):
    """
    tricon: edge-centred connectivity for 2D triangulations.
    Python translation of Darren Engwirda's tricon.m (0-based indexing).

    Parameters
    ----------
    tt : (nt, 3) int array
        Triangulation connectivity (triangle vertices).
    cc : (nc, 2) int array, optional
        Constrained edges.

    Returns
    -------
    ee : (ne, 5) int array
        Edge list: [v1, v2, t1, t2, constraint_flag].
    tt : (nt, 6) int array
        Triangle list: [v1, v2, v3, e1, e2, e3].
    """

    if cc is None:
        cc = np.empty((0, 2), dtype=int)

    #---------------------------------------------- basic checks
    if not isinstance(tt, np.ndarray) or tt.ndim != 2 or tt.shape[1] != 3:
        raise ValueError("tricon:incorrectDimensions - tt must be (n,3) int array")

    if tt.min() < 0:
        raise ValueError("tricon:invalidInputs - indices must be >= 0 (0-based)")

    if cc.size > 0 and (not isinstance(cc, np.ndarray) or cc.ndim != 2 or cc.shape[1] != 2):
        raise ValueError("tricon:incorrectDimensions - cc must be (m,2) int array")

    nt = tt.shape[0]
    nc = cc.shape[0]

    #------------------------------ assemble non-unique edge set
    ee = np.zeros((nt * 3, 2), dtype=int)
    ee[0*nt:1*nt, :] = tt[:, [0, 1]]
    ee[1*nt:2*nt, :] = tt[:, [1, 2]]
    ee[2*nt:3*nt, :] = tt[:, [2, 0]]

    #------------------------------ unique edges and re-indexing
    ee_sorted = np.sort(ee, axis=1)
    ed = ee_sorted[:, 0] * (2**31) + ee_sorted[:, 1]
    _, iv, jv = np.unique(ed, return_index=True, return_inverse=True)
    ee_unique = ee_sorted[iv, :]

    #------------------- tria-to-edge indexing: 3 edges per tria
    tt_full = np.zeros((nt, 6), dtype=int)
    tt_full[:, :3] = tt
    tt_full[:, 3] = jv[0*nt:1*nt]
    tt_full[:, 4] = jv[1*nt:2*nt]
    tt_full[:, 5] = jv[2*nt:3*nt]

    #------------------- edge-to-tria indexing: 2 trias per edge
    ne = ee_unique.shape[0]
    ee_full = np.zeros((ne, 5), dtype=int)  # [v1,v2,t1,t2,constraint_flag]
    ee_full[:, :2] = ee_unique

    #ep = np.full(ne, 2, dtype=int)  # next write position (2→t1, then 3→t2)

    # for ti in range(nt):
    #     for ei in tt_full[ti, 3:6]:
    #         if ep[ei] <= 3:  # store up to two trias per edge
    #             ee_full[ei, ep[ei]] = ti
    #             ep[ei] += 1
    for ti in range(nt):
        for ei in tt_full[ti, 3:6]:
            if ee_full[ei, 2] == 0:
                ee_full[ei, 2] = ti
            elif ee_full[ei, 3] == 0:
                ee_full[ei, 3] = ti
    # marquer les bords explicitement
    ee_full[ee_full[:, 3] == 0, 3] = -1

    #------------------------------------ find constrained edges
    if cc.size > 0:
        cc_sorted = np.sort(cc, axis=1)
        cd = cc_sorted[:, 0] * (2**31) + cc_sorted[:, 1]
        constraint_flag = np.isin(ed[iv], cd).astype(int)
        ee_full[:, 4] = constraint_flag


    return ee_full, tt_full
