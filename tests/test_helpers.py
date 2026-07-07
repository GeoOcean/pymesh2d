"""
Helper functions for test reference data management.
"""
import os
import numpy as np
from scipy.spatial import cKDTree


def save_reference_data(vert, tria, demo_num, suffix=""):
    """
    Save vertex and triangle data to ASCII reference files.
    
    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        Vertex coordinates.
    TRIA : ndarray of shape (T, 3)
        Triangle vertex indices.
    DEMO_NUM : int
        Demo number (0-10).
    SUFFIX : str, optional
        Optional suffix for multiple outputs per demo (e.g., "_1", "_2").
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(test_dir, "reference_data")
    os.makedirs(ref_dir, exist_ok=True)
    
    # Save vertices
    vert_file = os.path.join(ref_dir, f"demo{demo_num}_vert{suffix}.txt")
    np.savetxt(vert_file, vert, fmt="%.15e", delimiter=" ")
    
    # Save triangles
    tria_file = os.path.join(ref_dir, f"demo{demo_num}_tria{suffix}.txt")
    np.savetxt(tria_file, tria, fmt="%d", delimiter=" ")
    
    return vert_file, tria_file


def load_reference_data(demo_num, suffix=""):
    """
    Load vertex and triangle data from ASCII reference files.
    
    Parameters
    ----------
    DEMO_NUM : int
        Demo number (0-10).
    SUFFIX : str, optional
        Optional suffix for multiple outputs per demo.
    
    Returns
    -------
    VERT : ndarray of shape (V, 2)
        Vertex coordinates.
    TRIA : ndarray of shape (T, 3)
        Triangle vertex indices.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(test_dir, "reference_data")
    
    # Load vertices
    vert_file = os.path.join(ref_dir, f"demo{demo_num}_vert{suffix}.txt")
    if not os.path.exists(vert_file):
        raise FileNotFoundError(f"Reference file not found: {vert_file}")
    vert = np.loadtxt(vert_file)
    
    # Load triangles
    tria_file = os.path.join(ref_dir, f"demo{demo_num}_tria{suffix}.txt")
    if not os.path.exists(tria_file):
        raise FileNotFoundError(f"Reference file not found: {tria_file}")
    tria = np.loadtxt(tria_file, dtype=int)
    
    return vert, tria


def compare_meshes(vert1, tria1, vert2, tria2, rtol=1e-10, atol=1e-12):
    """
    Compare two meshes for equality.
    
    Parameters
    ----------
    VERT1, VERT2 : ndarray of shape (V, 2)
        Vertex coordinates to compare.
    TRIA1, TRIA2 : ndarray of shape (T, 3)
        Triangle vertex indices to compare.
    RTOL : float, optional
        Relative tolerance for vertex comparison.
    ATOL : float, optional
        Absolute tolerance for vertex comparison.
    
    Returns
    -------
    IS_EQUAL : bool
        True if meshes are equal.
    MESSAGE : str
        Description of differences if not equal.
    """
    # Check vertex count
    if vert1.shape[0] != vert2.shape[0]:
        return False, f"Vertex count mismatch: {vert1.shape[0]} vs {vert2.shape[0]}"
    
    if vert1.shape[1] != vert2.shape[1]:
        return False, f"Vertex dimension mismatch: {vert1.shape[1]} vs {vert2.shape[1]}"
    
    # Check triangle count
    if tria1.shape[0] != tria2.shape[0]:
        return False, f"Triangle count mismatch: {tria1.shape[0]} vs {tria2.shape[0]}"
    
    if tria1.shape[1] != tria2.shape[1]:
        return False, f"Triangle dimension mismatch: {tria1.shape[1]} vs {tria2.shape[1]}"
    
    # Compare vertices via nearest-neighbor matching (robust to insertion-order
    # differences and to floating-point tie-breaks on structured/grid-like point
    # sets, where a lexicographic sort can mis-pair near-duplicate coordinates
    # after a sub-epsilon perturbation).
    tree2 = cKDTree(vert2)
    dist, nn2_of_1 = tree2.query(vert1, k=1)

    max_diff = float(np.max(dist)) if dist.size else 0.0
    tol = atol + rtol * np.max(np.abs(vert2)) if vert2.size else atol
    if max_diff > tol:
        return False, f"Vertex coordinates differ (max diff: {max_diff:.2e})"

    # Nearest-neighbor matching must be a bijection (each vert2 point used once).
    if len(np.unique(nn2_of_1)) != vert2.shape[0]:
        return False, "Vertex matching is not a bijection (duplicate/ambiguous points)"

    vert1_to_vert2 = nn2_of_1
    
    # Remap triangle indices from vert1 to vert2
    tria1_remapped = vert1_to_vert2[tria1]
    
    # Normalize triangles (sort indices within each triangle)
    tria1_norm = np.sort(tria1_remapped, axis=1)
    tria2_norm = np.sort(tria2, axis=1)
    
    # Sort triangles for comparison
    tria1_sorted = tria1_norm[np.lexsort((tria1_norm[:, 2], tria1_norm[:, 1], tria1_norm[:, 0]))]
    tria2_sorted = tria2_norm[np.lexsort((tria2_norm[:, 2], tria2_norm[:, 1], tria2_norm[:, 0]))]
    
    if not np.array_equal(tria1_sorted, tria2_sorted):
        return False, "Triangle connectivity differs"
    
    return True, "Meshes are equal"

