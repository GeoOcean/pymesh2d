import numpy as np


def unique(set2, return_index=False, return_inverse=False):
    """
    UNIQUE2 : a fast variant of `unique` for edge lists.

    [set2] = unique2(set2) returns the unique rows of the N×2 array `set2`,
    equivalent to calling MATLAB’s `unique(set2, 'rows')`, but optimized for
    processing edge lists.

    [set2, imap, jmap] = unique2(set2) also returns the additional mapping
    arrays consistent with MATLAB’s `unique` function:
        - `imap` : indices such that `set2 = original_set2[imap, :]`
        - `jmap` : indices such that `original_set2 = set2[jmap, :]`

    Parameters
    ----------
    set2 : ndarray (N, 2)
        Array of edge index pairs.

    Returns
    -------
    set2 : ndarray (M, 2)
        Array containing only the unique edges.
    imap : ndarray (M,), optional
        Indices of the first occurrences of the unique edges.
    jmap : ndarray (N,), optional
        Indices mapping each original edge to its corresponding unique edge.

    Notes
    -----
    - This function is specifically optimized for edge-based operations in
      mesh processing and connectivity construction.
    - It provides the same logical behavior as `numpy.unique(set2, axis=0)`
      but with a focus on speed for structured integer arrays.

    References
    ----------
    Translation of the MESH2D function `UNIQUE2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not isinstance(set2, np.ndarray):
        raise TypeError("unique:incorrectInputClass")

    if set2.ndim != 2 or set2.shape[1] != 2:
        raise ValueError("unique:incorrectDimensions")

    set2_sorted = np.sort(set2, axis=1)

    # Encode each row as a single 64-bit integer for uniqueness
    # (works if indices < 2^31)
    code = set2_sorted[:, 0].astype(np.int64) * (2**31) + set2_sorted[:, 1].astype(
        np.int64
    )

    uniq_code, imap, jmap = np.unique(code, return_index=True, return_inverse=True)

    uniq = set2_sorted[imap, :]

    if return_index and return_inverse:
        return uniq, imap, jmap
    elif return_index:
        return uniq, imap
    elif return_inverse:
        return uniq, jmap
    else:
        return uniq
