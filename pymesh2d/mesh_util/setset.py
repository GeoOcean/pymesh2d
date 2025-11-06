import numpy as np

def setset(iset, jset):
    """
    setset: fast variant of ismember for edge lists (0-based).
    Python translation of Darren Engwirda's setset.m.

    Parameters
    ----------
    iset : (I, 2) int array
        Query edge list.
    jset : (J, 2) int array
        Reference edge list.

    Returns
    -------
    same : (I,) bool array
        True if iset[k,:] is present in jset.
    sloc : (I,) int array
        Index in jset of each matching row (or -1 if not found).
    """

    #---------------------------------------------- basic checks
    if not (isinstance(iset, np.ndarray) and isinstance(jset, np.ndarray)):
        raise TypeError("setset: inputs must be numpy arrays")

    if iset.ndim != 2 or jset.ndim != 2:
        raise ValueError("setset: inputs must be 2D arrays")

    if iset.shape[1] != 2 or jset.shape[1] != 2:
        raise ValueError("setset: each row must define an edge (2 columns)")

    #---------------------------------------------- ensure v1 <= v2
    iset = np.sort(iset, axis=1)
    jset = np.sort(jset, axis=1)

    #---------------------------------------------- encode edges as 1D keys
    iset_keys = iset[:, 0] * (2**31) + iset[:, 1]
    jset_keys = jset[:, 0] * (2**31) + jset[:, 1]

    #---------------------------------------------- fast membership
    # equivalent to: [same, sloc] = ismember(iset, jset, 'rows')
    jdict = {val: idx for idx, val in enumerate(jset_keys)}
    sloc = np.array([jdict.get(val, -1) for val in iset_keys], dtype=int)
    same = sloc >= 0

    return same, sloc
