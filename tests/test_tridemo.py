"""
Unit tests for tridemo functions.
Tests compare generated meshes against reference data.
"""
import os
import sys
import unittest

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

from tests.test_helpers import load_reference_data, compare_meshes


def hfun8(test):
    """User-defined mesh-size function for DEMO-8."""
    hmax = 0.05
    hmin = 0.01
    xmid = 0.0
    ymid = 0.0
    hcir = np.exp(-0.5 * (test[:, 0] - xmid) ** 2 - 2.0 * (test[:, 1] - ymid) ** 2)
    hfun = hmax - (hmax - hmin) * hcir
    return hfun


class TestTridemo(unittest.TestCase):
    """Test cases for tridemo functions."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize mesh system once for all tests."""
        initmsh()
    
    def test_demo0_1(self):
        """Test DEMO0 first output (basic refine)."""
        node = np.array([
            [0, 0], [9, 0], [9, 9], [0, 9],
            [4, 4], [5, 4], [5, 5], [4, 5],
        ])
        edge = np.array([
            [1, 2], [2, 3], [3, 4], [4, 1],
            [5, 6], [6, 7], [7, 8], [8, 5],
        ]) - 1
        
        opts = {}
        vert, etri, tria, tnum = refine(node, edge, [], opts)
        
        vert_ref, tria_ref = load_reference_data(0, "_1")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo0_2(self):
        """Test DEMO0 second output (refine with hfun)."""
        node = np.array([
            [0, 0], [9, 0], [9, 9], [0, 9],
            [4, 4], [5, 4], [5, 5], [4, 5],
        ])
        edge = np.array([
            [1, 2], [2, 3], [3, 4], [4, 1],
            [5, 6], [6, 7], [7, 8], [8, 5],
        ]) - 1
        
        opts = {}
        hfun = 0.5
        vert, etri, tria, tnum = refine(node, edge, [], opts, hfun)
        
        vert_ref, tria_ref = load_reference_data(0, "_2")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo1_1(self):
        """Test DEMO1 first output (RHO2=1.50)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "lake.msh")
        node, edge, _, _ = triread(meshfile)
        
        opts = {"kind": "delaunay", "rho2": 1.50}
        vert, etri, tria, tnum = refine(node, edge, [], opts)
        
        vert_ref, tria_ref = load_reference_data(1, "_1")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo1_2(self):
        """Test DEMO1 second output (RHO2=1.00)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "lake.msh")
        node, edge, _, _ = triread(meshfile)
        
        opts = {"kind": "delaunay", "rho2": 1.00}
        vert, etri, tria, tnum = refine(node, edge, [], opts)
        
        vert_ref, tria_ref = load_reference_data(1, "_2")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo2_1(self):
        """Test DEMO2 first output (KIND=delaunay)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "lake.msh")
        node, edge, _, _ = triread(meshfile)
        
        opts = {"kind": "delaunay", "rho2": 1.00}
        vert, etri, tria, tnum = refine(node, edge, [], opts)
        
        vert_ref, tria_ref = load_reference_data(2, "_1")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo2_2(self):
        """Test DEMO2 second output (KIND=delfront)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "lake.msh")
        node, edge, _, _ = triread(meshfile)
        
        opts = {"kind": "delfront", "rho2": 1.00}
        vert, etri, tria, tnum = refine(node, edge, [], opts)
        
        vert_ref, tria_ref = load_reference_data(2, "_2")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo3(self):
        """Test DEMO3 (mesh-size constraints)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "airfoil.msh")
        node, edge, _, _ = triread(meshfile)
        
        olfs = {"dhdx": 0.15}
        vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
        slfs = idxtri(vlfs, tlfs)
        hfun = trihfn
        vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
        
        vert_ref, tria_ref = load_reference_data(3)
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo4_1(self):
        """Test DEMO4 first output (before smooth)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "airfoil.msh")
        node, edge, _, _ = triread(meshfile)
        
        olfs = {"dhdx": 0.15}
        vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
        slfs = idxtri(vlfs, tlfs)
        hfun = trihfn
        vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
        
        vert_ref, tria_ref = load_reference_data(4, "_1")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo4_2(self):
        """Test DEMO4 second output (after smooth)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "airfoil.msh")
        node, edge, _, _ = triread(meshfile)
        
        olfs = {"dhdx": 0.15}
        vlfs, tlfs, hlfs = lfshfn(node, edge, [], olfs)
        slfs = idxtri(vlfs, tlfs)
        hfun = trihfn
        vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
        vnew, enew, tnew, tnum = smooth(vert, etri, tria, tnum)
        
        vert_ref, tria_ref = load_reference_data(4, "_2")
        is_equal, message = compare_meshes(vnew, tnew[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo5(self):
        """Test DEMO5 (multi-part geometry)."""
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
        
        vert_ref, tria_ref = load_reference_data(5)
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo6(self):
        """Test DEMO6 (internal constraints)."""
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
        
        vert_ref, tria_ref = load_reference_data(6)
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo7_1(self):
        """Test DEMO7 first output (before tridiv)."""
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
        
        vert_ref, tria_ref = load_reference_data(7, "_1")
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo7_2(self):
        """Test DEMO7 second output (after tridiv)."""
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
        vnew, enew, tnew, tnum = tridiv(vert, etri, tria, tnum)
        vnew, enew, tnew, tnum = smooth(vnew, enew, tnew, tnum)
        
        vert_ref, tria_ref = load_reference_data(7, "_2")
        is_equal, message = compare_meshes(vnew, tnew[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo8(self):
        """Test DEMO8 (user-defined mesh-size function)."""
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
        
        vert_ref, tria_ref = load_reference_data(8)
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo9(self):
        """Test DEMO9 (large-scale problem)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "islands.msh")
        node, edge, _, _ = triread(meshfile)
        
        vlfs, tlfs, hlfs = lfshfn(node, edge)
        slfs = idxtri(vlfs, tlfs)
        hfun = trihfn
        vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
        vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
        
        vert_ref, tria_ref = load_reference_data(9)
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)
    
    def test_demo10(self):
        """Test DEMO10 (medium-scale problem)."""
        filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        meshfile = os.path.join(filepath, "pymesh2d", "poly_data", "river.msh")
        node, edge, _, _ = triread(meshfile)
        
        vlfs, tlfs, hlfs = lfshfn(node, edge)
        slfs = idxtri(vlfs, tlfs)
        hfun = trihfn
        vert, etri, tria, tnum = refine(node, edge, [], {}, hfun, vlfs, tlfs, slfs, hlfs)
        vert, etri, tria, tnum = smooth(vert, etri, tria, tnum)
        
        vert_ref, tria_ref = load_reference_data(10)
        is_equal, message = compare_meshes(vert, tria[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)


if __name__ == "__main__":
    unittest.main()


