"""
Generate reference data files from tridemo functions.
Run this script to create reference files for all demos.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pymesh2d.hfun_util.lfshfn import lfshfn
from pymesh2d.hfun_util.trihfn import trihfn
from pymesh2d.initmsh import initmsh
from pymesh2d.mesh_util.idxtri import idxtri
from pymesh2d.mesh_util.tridiv import tridiv
from pymesh2d.refine import refine
from pymesh2d.smooth import smooth
from pymesh2d.triread import triread

from tests.test_helpers import save_reference_data


def hfun8(test):
    """User-defined mesh-size function for DEMO-8."""
    hmax = 0.05
    hmin = 0.01
    xmid = 0.0
    ymid = 0.0
    hcir = np.exp(-0.5 * (test[:, 0] - xmid) ** 2 - 2.0 * (test[:, 1] - ymid) ** 2)
    hfun = hmax - (hmax - hmin) * hcir
    return hfun


def run_demo0():
    """DEMO0: Simple square domain with square hole."""
    node = np.array([
        [0, 0], [9, 0], [9, 9], [0, 9],  # outer square
        [4, 4], [5, 4], [5, 5], [4, 5],  # inner square
    ])
    edge = np.array([
        [1, 2], [2, 3], [3, 4], [4, 1],  # outer square
        [5, 6], [6, 7], [7, 8], [8, 5],  # inner square
    ]) - 1
    
    opts = {}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 0, "_1")
    
    hfun = 0.5
    vert, etri, tria, tnum = refine(node, edge, [], opts, hfun)
    save_reference_data(vert, tria[:, 0:3], 0, "_2")


def run_demo1():
    """DEMO1: Impact of RHO2 threshold."""
    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "lake.msh")
    node, edge, _, _ = triread(meshfile)
    
    opts = {"kind": "delaunay", "rho2": 1.50}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 1, "_1")
    
    opts = {"kind": "delaunay", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 1, "_2")


def run_demo2():
    """DEMO2: Impact of refinement KIND."""
    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "lake.msh")
    node, edge, _, _ = triread(meshfile)
    
    opts = {"kind": "delaunay", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 2, "_1")
    
    opts = {"kind": "delfront", "rho2": 1.00}
    vert, etri, tria, tnum = refine(node, edge, [], opts)
    save_reference_data(vert, tria[:, 0:3], 2, "_2")


def run_demo3():
    """DEMO3: User-defined mesh-size constraints."""
    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "airfoil.msh")
    node, edge, _, _ = triread(meshfile)
    
    olfs = {"dhdx": 0.15}
    vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    save_reference_data(vert, tria[:, 0:3], 3)


def run_demo4():
    """DEMO4: Hill-climbing mesh optimization."""
    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "airfoil.msh")
    node, edge, _, _ = triread(meshfile)
    
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
    node = np.array([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
        [0.0, 0.0], [0.2, 0.7], [0.6, 0.2], [0.4, 0.8], [0.0, 0.5],
        [-0.7, 0.3], [-0.1, 0.1], [-0.6, 0.5], [-0.9, -0.8],
        [-0.6, -0.7], [-0.3, -0.6], [0.0, -0.5], [0.3, -0.4],
        [-0.3, 0.4], [-0.1, 0.3],
    ])
    edge = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11],
        [4, 12], [4, 13], [4, 14], [4, 15], [4, 16], [4, 17], [4, 18],
    ])
    part = [np.array([0, 1, 2, 3])]
    
    hmax = 0.175
    opts = {"kind": "delaunay"}
    vert, etri, tria, tnum = refine(node, edge, part, opts, hmax)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 6)


def run_demo7():
    """DEMO7: Quadtree-type refinement."""
    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "channel.msh")
    node, edge, _, _ = triread(meshfile)
    
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
    
    hfun = hfun8
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 8)


def run_demo9():
    """DEMO9: Large-scale problem."""
    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "islands.msh")
    node, edge, _, _ = triread(meshfile)
    
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 9)


def run_demo10():
    """DEMO10: Medium-scale problem."""
    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "river.msh")
    node, edge, _, _ = triread(meshfile)
    
    vlfs, tlfs, hlfs = lfshfn(node, edge)
    slfs = idxtri(vlfs, tlfs)
    hfun = trihfn
    vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
    vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
    save_reference_data(vert, tria[:, 0:3], 10)


def main():
    """Generate all reference files."""
    print("Generating reference data files...")
    initmsh()
    
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


