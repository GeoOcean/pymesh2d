import numpy as np
from .tricon import tricon

def bfstri(PSLG, tria, seed):
    """
    bfstri expand about a single seed triangle via BFS. The search terminates
    when constraining edges are encountered.
    SEEN[i] is True if the i-th triangle is found in the current expansion.

    Parameters
    ----------
    PSLG : (E,2) array or empty
        Constraining edges (can be empty).
    tria : (T,3) array
        Triangle connectivity (0-based indices).
    seed : (S,) array
        Indices of seed triangles (0-based).

    Returns
    -------
    seen : (T,) boolean array
        True if triangle was visited.
    """

    if not isinstance(tria, np.ndarray) or not isinstance(seed, np.ndarray):
        raise TypeError("bfstri:incorrectInputClass")

    if tria.ndim != 2 or tria.shape[1] != 3:
        raise ValueError("bfstri:incorrectDimensions")

    if PSLG is not None and PSLG.size > 0:
        if not isinstance(PSLG, np.ndarray):
            raise TypeError("bfstri:incorrectInputClass")
        if PSLG.ndim != 2 or PSLG.shape[1] != 2:
            raise ValueError("bfstri:incorrectDimensions")

    ntri = tria.shape[0]

    # tricon must return (edge, tria_adj)
    edge, tria_adj = tricon(tria, PSLG)

    seed = seed.ravel()
    list_buf = np.zeros(ntri, dtype=int)
    nlst = len(seed)
    list_buf[:nlst] = seed

    seen = np.zeros(ntri, dtype=bool)

    # BFS iterations
    while nlst >= 1:
        next_t = list_buf[nlst-1]
        nlst -= 1
        seen[next_t] = True

        # visit 1-ring neighbours
        for eadj in range(3):
            epos = tria_adj[next_t, eadj+3]

            # if edge not constrained
            if edge[epos, 4] == 0:
                if next_t != edge[epos, 2]:
                    tadj = edge[epos, 2]
                else:
                    tadj = edge[epos, 3]

                if tadj > 0 and not seen[tadj]:
                    seen[tadj] = True
                    nlst += 1
                    list_buf[nlst-1] = tadj

    return seen
