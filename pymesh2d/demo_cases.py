"""
Shared geometry builders for the `tridemo` demonstration problems.

These pure, plot-free builders are consumed by three places that used to
each define the same node/edge/hfun data independently: the interactive
`tridemo` module, the pytest regression suite, and the reference-data
generation script. Keeping a single source of truth here avoids the three
copies drifting apart.
"""
import importlib.resources

import numpy as np


def demo_data_path(name):
    """
    Return the path to a bundled `.msh` file under `pymesh2d/poly_data/`.

    Parameters
    ----------
    name : str
        File name, e.g. ``"lake.msh"``.

    Returns
    -------
    str
        Absolute path to the requested file.
    """
    return str(importlib.resources.files("pymesh2d") / "poly_data" / name)


def square_with_hole_geometry():
    """
    Build the DEMO-0 geometry: a square domain with a square hole cut from
    its centre.

    Returns
    -------
    node : ndarray of shape (8, 2)
    edge : ndarray of shape (8, 2)
    """
    node = np.array(
        [
            [0, 0],  # outer square
            [9, 0],
            [9, 9],
            [0, 9],
            [4, 4],  # inner square
            [5, 4],
            [5, 5],
            [4, 5],
        ]
    )
    edge = (
        np.array(
            [
                [1, 2],  # outer square
                [2, 3],
                [3, 4],
                [4, 1],
                [5, 6],  # inner square
                [6, 7],
                [7, 8],
                [8, 5],
            ]
        )
        - 1
    )
    return node, edge


def multi_part_geometry():
    """
    Build the DEMO-5 geometry: an outer square, an inner square, and a
    circular hole, assembled as a three-part PSLG.

    Returns
    -------
    node : ndarray of shape (N, 2)
    edge : ndarray of shape (N, 2)
        Edge list (part tags already stripped out).
    part : list of ndarray
        Per-part edge index arrays, suitable for `refine`/`smooth`.
    """
    nod1 = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    edg1 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    tag1 = np.zeros((edg1.shape[0], 1), dtype=int)

    nod2 = np.array([[0.1, 0.0], [0.8, 0.0], [0.8, 0.8], [0.1, 0.8]])
    edg2 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    tag2 = np.ones((edg2.shape[0], 1), dtype=int)

    adel = 2.0 * np.pi / 64.0
    amin = 0.0
    amax = 2.0 * np.pi - adel
    ang = np.arange(amin, amax + adel / 2, adel)
    xcir = 0.33 * np.cos(ang) - 0.33
    ycir = 0.33 * np.sin(ang) - 0.25
    ncir = np.column_stack((xcir, ycir))

    numc = ncir.shape[0]
    ecir = np.column_stack((np.arange(numc - 1), np.arange(1, numc)))
    ecir = np.vstack((ecir, [numc - 1, 0]))
    tagc = np.full((ecir.shape[0], 1), 2, dtype=int)

    edg2 = edg2 + nod1.shape[0]
    edge = np.vstack((np.hstack((edg1, tag1)), np.hstack((edg2, tag2))))
    node = np.vstack((nod1, nod2))

    ecir = ecir + node.shape[0]
    edge = np.vstack((edge, np.hstack((ecir, tagc))))
    node = np.vstack((node, ncir))

    edge_tag = edge[:, 2].astype(int)
    part = [
        np.where((edge_tag == 0) | (edge_tag == 1) | (edge_tag == 2))[0],
        np.where(edge_tag == 1)[0],
        np.where(edge_tag == 2)[0],
    ]
    edge = edge[:, :2].astype(int)

    return node, edge, part


def internal_constraint_geometry():
    """
    Build the DEMO-6 geometry: a square domain with a "star" of internal
    constraint edges radiating from its centre.

    The geometry is split into "exterior" and "interior" components via
    the `part` argument: `part[0]` defines the exterior boundary, while the
    star edges are left unreferenced and are imposed as isolated internal
    constraints.

    Returns
    -------
    node : ndarray of shape (19, 2)
    edge : ndarray of shape (18, 2)
    part : list of ndarray
    """
    node = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [0.2, 0.7],
            [0.6, 0.2],
            [0.4, 0.8],
            [0.0, 0.5],
            [-0.7, 0.3],
            [-0.1, 0.1],
            [-0.6, 0.5],
            [-0.9, -0.8],
            [-0.6, -0.7],
            [-0.3, -0.6],
            [0.0, -0.5],
            [0.3, -0.4],
            [-0.3, 0.4],
            [-0.1, 0.3],
        ]
    )
    edge = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [4, 6],
            [4, 7],
            [4, 8],
            [4, 9],
            [4, 10],
            [4, 11],
            [4, 12],
            [4, 13],
            [4, 14],
            [4, 15],
            [4, 16],
            [4, 17],
            [4, 18],
        ]
    )
    part = [np.array([0, 1, 2, 3])]
    return node, edge, part


def circle_in_box_geometry():
    """
    Build the DEMO-8 geometry: a rectangular box with a small circular hole,
    used to exercise a user-defined mesh-size function.

    Returns
    -------
    node : ndarray of shape (N, 2)
    edge : ndarray of shape (N, 2)
    """
    node = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [-1.0, 1.0]])
    edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    adel = 2.0 * np.pi / 64.0
    amin = 0.0 * np.pi
    amax = 2.0 * np.pi - adel
    angles = np.arange(amin, amax + adel, adel)
    xcir = 0.20 * np.cos(angles)
    ycir = 0.20 * np.sin(angles)
    ncir = np.column_stack([xcir, ycir])
    numc = ncir.shape[0]

    ecir = np.zeros((numc, 2), dtype=int)
    ecir[:, 0] = np.arange(numc)
    ecir[:, 1] = np.roll(ecir[:, 0], -1)
    ecir = ecir + node.shape[0]

    edge = np.vstack([edge, ecir])
    node = np.vstack([node, ncir])
    return node, edge


def hfun8(test):
    """
    User-defined mesh-size function used by DEMO-8: a Gaussian dip in cell
    size centred on the origin.

    Parameters
    ----------
    test : ndarray of shape (N, 2)
        Coordinates (x, y) at which the mesh-size function is evaluated.

    Returns
    -------
    hfun : ndarray of shape (N,)
        Mesh-size values at the input points.
    """
    hmax = 0.05
    hmin = 0.01
    xmid = 0.0
    ymid = 0.0
    hcir = np.exp(-0.5 * (test[:, 0] - xmid) ** 2 - 2.0 * (test[:, 1] - ymid) ** 2)
    return hmax - (hmax - hmin) * hcir
