"""
Regression tests for the tridemo geometries: each case rebuilds a demo mesh
and compares it against a pinned reference stored in tests/reference_data/.
"""
import os

import numpy as np
import pytest
from pymesh2d.demo_cases import (
    circle_in_box_geometry,
    demo_data_path,
    hfun8,
    internal_constraint_geometry,
    multi_part_geometry,
    square_with_hole_geometry,
)
from pymesh2d.hfun_util.lfshfn import lfshfn
from pymesh2d.hfun_util.trihfn import trihfn
from pymesh2d.mesh_util.idxtri import idxtri
from pymesh2d.mesh_util.tridiv import tridiv
from pymesh2d.refine import refine
from pymesh2d.smooth import smooth
from pymesh2d.triread import triread

from tests.test_helpers import compare_meshes, load_reference_data


def _demo0_1():
    node, edge = square_with_hole_geometry()
    vert, etri, tria, tnum = refine(node, edge, [], {})
    return vert, tria[:, 0:3]


def _demo0_2():
    node, edge = square_with_hole_geometry()
    vert, etri, tria, tnum = refine(node, edge, [], {}, 0.5)
    return vert, tria[:, 0:3]


def _demo1_1():
    node, edge, _, _ = triread(demo_data_path("lake.msh"))
    opts = {"kind": "delaunay", "rho2": 1.50}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    return vert, tria[:, 0:3]


def _demo1_2():
    node, edge, _, _ = triread(demo_data_path("lake.msh"))
    opts = {"kind": "delaunay", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    return vert, tria[:, 0:3]


def _demo2_1():
    node, edge, _, _ = triread(demo_data_path("lake.msh"))
    opts = {"kind": "delaunay", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    return vert, tria[:, 0:3]


def _demo2_2():
    node, edge, _, _ = triread(demo_data_path("lake.msh"))
    opts = {"kind": "delfront", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    return vert, tria[:, 0:3]


def _demo3():
    node, edge, _, _ = triread(demo_data_path("airfoil.msh"))
    olfs = {"dhdx": 0.15}
    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)
    vert, etri, tria, tnum = refine(node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs)
    return vert, tria[:, 0:3]


def _demo4_1():
    node, edge, _, _ = triread(demo_data_path("airfoil.msh"))
    olfs = {"dhdx": 0.15}
    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)
    vert, etri, tria, tnum = refine(node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs)
    return vert, tria[:, 0:3]


def _demo4_2():
    node, edge, _, _ = triread(demo_data_path("airfoil.msh"))
    olfs = {"dhdx": 0.15}
    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)
    vert, etri, tria, tnum = refine(node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs)
    vnew, enew, tnew, tnum = smooth(vert, etri, tria, tnum)
    return vnew, tnew[:, 0:3]


def _demo5():
    node, edge, part = multi_part_geometry()
    hmax = 0.045
    vlfs, tlfs, hlfs = lfshfn(node, edge, part)
    hlfs = np.minimum(hmax, hlfs)
    slfs = idxtri(vlfs, tlfs)
    vert, etri, tria, tnum = refine(node, edge, part, {}, trihfn, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    return vert, tria[:, 0:3]


def _demo6():
    node, edge, part = internal_constraint_geometry()
    hmax = 0.175
    opts = {"kind": "delaunay"}
    vert, etri, tria, tnum = refine(node, edge, part, opts, hmax)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    return vert, tria[:, 0:3]


def _demo7_1():
    node, edge, _, _ = triread(demo_data_path("channel.msh"))
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    pmax, pmin = np.max(node, axis=0), np.min(node, axis=0)
    hmax = np.mean(pmax - pmin) / 17.0
    hlfs = np.minimum(hmax, hlfs)
    vert, etri, tria, tnum = refine(node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    return vert, tria[:, 0:3]


def _demo7_2():
    node, edge, _, _ = triread(demo_data_path("channel.msh"))
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    pmax, pmin = np.max(node, axis=0), np.min(node, axis=0)
    hmax = np.mean(pmax - pmin) / 17.0
    hlfs = np.minimum(hmax, hlfs)
    vert, etri, tria, tnum = refine(node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    vnew, enew, tnew, tnum = tridiv(vert, etri, tria, tnum)
    vnew, enew, tnew, tnum = smooth(vnew, enew, tnew, tnum)
    return vnew, tnew[:, 0:3]


def _demo8():
    node, edge = circle_in_box_geometry()
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun8)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    return vert, tria[:, 0:3]


def _demo9():
    node, edge, _, _ = triread(demo_data_path("islands.msh"))
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    vert, etri, tria, tnum = refine(node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    return vert, tria[:, 0:3]


def _demo10():
    node, edge, _, _ = triread(demo_data_path("river.msh"))
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    vert, etri, tria, tnum = refine(node, edge, [], {}, trihfn, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    return vert, tria[:, 0:3]


# Each entry is (test id, builder, reference demo number, reference suffix).
DEMO_CASES = [
    ("demo0_1", _demo0_1, 0, "_1"),
    ("demo0_2", _demo0_2, 0, "_2"),
    ("demo1_1", _demo1_1, 1, "_1"),
    ("demo1_2", _demo1_2, 1, "_2"),
    ("demo2_1", _demo2_1, 2, "_1"),
    ("demo2_2", _demo2_2, 2, "_2"),
    ("demo3", _demo3, 3, ""),
    ("demo4_1", _demo4_1, 4, "_1"),
    ("demo4_2", _demo4_2, 4, "_2"),
    ("demo5", _demo5, 5, ""),
    ("demo6", _demo6, 6, ""),
    ("demo7_1", _demo7_1, 7, "_1"),
    ("demo7_2", _demo7_2, 7, "_2"),
    ("demo8", _demo8, 8, ""),
    ("demo9", _demo9, 9, ""),
    ("demo10", _demo10, 10, ""),
]


@pytest.mark.parametrize(
    "builder, demo_num, suffix",
    [case[1:] for case in DEMO_CASES],
    ids=[case[0] for case in DEMO_CASES],
)
def test_demo_matches_reference(builder, demo_num, suffix):
    """Rebuild a demo mesh and compare it against its pinned reference."""
    ref_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data")
    vert_file = os.path.join(ref_dir, f"demo{demo_num}_vert{suffix}.txt")
    if not os.path.exists(vert_file):
        pytest.skip(
            f"Reference file {vert_file} not committed (large-mesh demos are "
            "excluded from the repo, see .gitignore); run "
            "tests/generate_references.py locally to regenerate it."
        )

    vert, tria = builder()
    vert_ref, tria_ref = load_reference_data(demo_num, suffix)
    is_equal, message = compare_meshes(vert, tria, vert_ref, tria_ref)
    assert is_equal, message
