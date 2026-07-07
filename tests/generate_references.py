"""
Generate reference data files from the tridemo geometries.
Run this script to create reference files for all demos.
"""
import numpy as np
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

from tests.test_helpers import save_reference_data


def run_demo0():
    """DEMO0: Simple square domain with square hole."""
    node, edge = square_with_hole_geometry()

    opts = {}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 0, "_1")

    hfun = 0.5
    vert, etri, tria, tnum = refine(node, edge, [], opts, hfun)
    save_reference_data(vert, tria[:, 0:3], 0, "_2")


def run_demo1():
    """DEMO1: Impact of RHO2 threshold."""
    node, edge, _, _ = triread(demo_data_path("lake.msh"))

    opts = {"kind": "delaunay", "rho2": 1.50}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 1, "_1")

    opts = {"kind": "delaunay", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 1, "_2")


def run_demo2():
    """DEMO2: Impact of refinement KIND."""
    node, edge, _, _ = triread(demo_data_path("lake.msh"))

    opts = {"kind": "delaunay", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 2, "_1")

    opts = {"kind": "delfront", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 2, "_2")


def run_demo3():
    """DEMO3: User-defined mesh-size constraints."""
    node, edge, _, _ = triread(demo_data_path("airfoil.msh"))

    olfs = {"dhdx": 0.15}
    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    save_reference_data(vert, tria[:, 0:3], 3)


def run_demo4():
    """DEMO4: Hill-climbing mesh optimization."""
    node, edge, _, _ = triread(demo_data_path("airfoil.msh"))

    olfs = {"dhdx": 0.15}
    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    save_reference_data(vert, tria[:, 0:3], 4, "_1")

    vnew, enew, tnew, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vnew, tnew[:, 0:3], 4, "_2")


def run_demo5():
    """DEMO5: Multi-part geometry."""
    node, edge, part = multi_part_geometry()

    hmax = 0.045
    vlfs, tlfs, hlfs = lfshfn(node, edge, part)
    hlfs = np.minimum(hmax, hlfs)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn

    vert, etri, tria, tnum = refine(node, edge, part, {}, hfun, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 5)


def run_demo6():
    """DEMO6: Internal constraints."""
    node, edge, part = internal_constraint_geometry()

    hmax = 0.175
    opts = {"kind": "delaunay"}
    vert, etri, tria, tnum = refine(node, edge, part, opts, hmax)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 6)


def run_demo7():
    """DEMO7: Quadtree-type refinement."""
    node, edge, _, _ = triread(demo_data_path("channel.msh"))

    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    pmax = np.max(node, axis=0)
    pmin = np.min(node, axis=0)
    hmax = np.mean(pmax - pmin) / 17.0
    hlfs = np.minimum(hmax, hlfs)
    hfun = trihfn

    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 7, "_1")

    vnew, enew, tnew, tnum = tridiv(vert, etri, tria, tnum)
    vnew, enew, tnew, tnum = smooth(vnew, enew, tnew, tnum)
    save_reference_data(vnew, tnew[:, 0:3], 7, "_2")


def run_demo8():
    """DEMO8: User-defined mesh-size function."""
    node, edge = circle_in_box_geometry()

    hfun = hfun8
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 8)


def run_demo9():
    """DEMO9: Large-scale problem."""
    node, edge, _, _ = triread(demo_data_path("islands.msh"))

    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 9)


def run_demo10():
    """DEMO10: Medium-scale problem."""
    node, edge, _, _ = triread(demo_data_path("river.msh"))

    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 10)


def main():
    """Generate all reference files."""
    print("Generating reference data files...")

    demos = [
        (0, run_demo0), (1, run_demo1), (2, run_demo2), (3, run_demo3),
        (4, run_demo4), (5, run_demo5), (6, run_demo6), (7, run_demo7),
        (8, run_demo8), (9, run_demo9), (10, run_demo10),
    ]

    for demo_num, demo_func in demos:
        print(f"Running demo {demo_num}...")
        try:
            demo_func()
            print(f"  Demo {demo_num} completed successfully")
        except Exception as e:
            print(f"  Demo {demo_num} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nReference data generation complete!")


if __name__ == "__main__":
    main()
