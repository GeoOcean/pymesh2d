import numpy as np

def unique(set2, return_index=False, return_inverse=False):
    """
    unique a (much) faster variant of UNIQUE for edge lists.
    
    Parameters
    ----------
    set2 : (N,2) ndarray of int
        Input edge list.
    return_index : bool, optional
        If True, return the indices of the unique values (imap).
    return_inverse : bool, optional
        If True, return the indices to reconstruct the input array (jmap).
    
    Returns
    -------
    uniq : (M,2) ndarray
        Unique edges, sorted in each row.
    imap : (M,) ndarray of int, optional
        Indices such that uniq = set2[imap].
    jmap : (N,) ndarray of int, optional
        Indices such that set2 = uniq[jmap].
    """

    if not isinstance(set2, np.ndarray):
        raise TypeError("unique:incorrectInputClass")

    if set2.ndim != 2 or set2.shape[1] != 2:
        raise ValueError("unique:incorrectDimensions")

    # Sort rows (like MATLAB's sort(...,2))
    set2_sorted = np.sort(set2, axis=1)

    # Encode each row as a single 64-bit integer for uniqueness
    # (works if indices < 2^31)
    code = set2_sorted[:,0].astype(np.int64) * (2**31) + set2_sorted[:,1].astype(np.int64)

    uniq_code, imap, jmap = np.unique(code, return_index=True, return_inverse=True)

    uniq = set2_sorted[imap,:]

    if return_index and return_inverse:
        return uniq, imap, jmap
    elif return_index:
        return uniq, imap
    elif return_inverse:
        return uniq, jmap
    else:
        return uniq
