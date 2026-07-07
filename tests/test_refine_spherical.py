"""
Tests for the spherical (lon/lat) refine workflow (``opts['spherical']=True``).

The planar/UTM path is pinned by the demo-based regression suite; these tests
cover the spherical geometry primitives and the end-to-end spherical mode:
vertices stay in lon/lat degrees while the mesh-size function is honoured in
great-circle metres.
"""

import unittest

import numpy as np

from pymesh2d.constants import DEG2RAD, EARTH_RADIUS
from pymesh2d.geom_util.sphere import (
    cdtbal1_sph,
    findball_sph,
    sphdist,
    sphmid,
    sphmove,
    tribal2_sph,
)
from pymesh2d.refine import refine


class TestSpherePrimitives(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.a = np.column_stack([rng.uniform(-12, -2, 40), rng.uniform(35, 42, 40)])
        self.b = self.a + rng.uniform(-0.5, 0.5, (40, 2))
        self.rng = rng

    def test_sphdist_matches_haversine(self):
        lon1, lat1 = self.a[:, 0] * DEG2RAD, self.a[:, 1] * DEG2RAD
        lon2, lat2 = self.b[:, 0] * DEG2RAD, self.b[:, 1] * DEG2RAD
        hav = (
            2.0
            * EARTH_RADIUS
            * np.arcsin(
                np.sqrt(
                    np.sin((lat2 - lat1) / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
                )
            )
        )
        np.testing.assert_allclose(sphdist(self.a, self.b), hav, rtol=1e-12)

    def test_sphmove_places_at_requested_distance_on_geodesic(self):
        dab = sphdist(self.a, self.b)
        d = 0.7 * dab
        m = sphmove(self.a, self.b, d)
        np.testing.assert_allclose(sphdist(self.a, m), d, atol=1e-6)
        # betweenness: on the geodesic, distances add up
        np.testing.assert_allclose(
            sphdist(self.a, m) + sphdist(m, self.b), dab, atol=1e-6
        )

    def test_sphmid_is_equidistant(self):
        mid = sphmid(self.a, self.b)
        np.testing.assert_allclose(
            sphdist(mid, self.a), sphdist(mid, self.b), atol=1e-6
        )

    def test_tribal2_sph_centres_are_equidistant(self):
        # Mesh-scale triangles (a few km) about several base points: the local
        # tangent-plane circumcircle is equidistant (great-circle) from the
        # three vertices and its radius matches, to tangent-plane accuracy.
        rng = self.rng
        base = np.column_stack([rng.uniform(-12, -2, 40), rng.uniform(35, 42, 40)])
        p = np.vstack(
            [
                base,
                base + rng.uniform(-0.05, 0.05, (40, 2)),
                base + rng.uniform(-0.05, 0.05, (40, 2)),
            ]
        )
        tt = np.column_stack([np.arange(40), 40 + np.arange(40), 80 + np.arange(40)])
        cc = tribal2_sph(p, tt)
        r0 = sphdist(cc[:, :2], p[tt[:, 0]])
        r1 = sphdist(cc[:, :2], p[tt[:, 1]])
        r2 = sphdist(cc[:, :2], p[tt[:, 2]])
        np.testing.assert_allclose(r0, r1, rtol=1e-3)
        np.testing.assert_allclose(r0, r2, rtol=1e-3)
        # radius**2 is the mean of the three squared radii (as planar pwrbal2),
        # i.e. the RMS radius -- equal to each only for an equilateral triangle.
        np.testing.assert_allclose(np.sqrt(cc[:, 2]), r0, rtol=5e-3)

    def test_cdtbal1_sph_diametric_balls(self):
        ee = np.arange(40).reshape(20, 2)
        pp = np.vstack([self.a[:20], self.b[:20]])
        ee = np.column_stack([np.arange(20), 20 + np.arange(20)])
        bb = cdtbal1_sph(pp, ee)
        # centre equidistant from endpoints, radius = half length
        d0 = sphdist(bb[:, :2], pp[ee[:, 0]])
        d1 = sphdist(bb[:, :2], pp[ee[:, 1]])
        np.testing.assert_allclose(d0, d1, atol=1e-6)
        np.testing.assert_allclose(np.sqrt(bb[:, 2]) * 2.0, d0 + d1, atol=1e-6)

    def test_findball_sph_matches_bruteforce(self):
        rng = self.rng
        balls = np.column_stack(
            [self.a[:, 0], self.a[:, 1], rng.uniform(1e3, 5e4, 40) ** 2]
        )
        pts = np.column_stack([rng.uniform(-12, -2, 100), rng.uniform(35, 42, 100)])
        vp, vi, _ = findball_sph(balls, pts)
        got = set()
        for ii in range(vp.shape[0]):
            for ip in range(vp[ii, 0], vp[ii, 1] + 1):
                got.add((ii, int(vi[ip])))
        want = set()
        for ii in range(pts.shape[0]):
            dd = sphdist(np.tile(pts[ii], (40, 1)), balls[:, :2]) ** 2
            for jj in np.where(dd <= balls[:, 2])[0]:
                want.add((ii, int(jj)))
        self.assertEqual(got, want)


class TestRefineSpherical(unittest.TestCase):
    def test_spherical_refine_honours_metric_hfun(self):
        """
        Mesh a lon/lat square with a constant mesh size in metres: edge
        lengths (great-circle) must conform to hfun everywhere, independent
        of latitude.
        """
        node = np.array([[-4.0, 42.8], [-3.0, 42.8], [-3.0, 43.6], [-4.0, 43.6]])
        edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        h_m = 4000.0  # metres

        vert, conn, tria, tnum = refine(
            node, edge, [], {"disp": np.inf, "spherical": True}, h_m
        )

        self.assertGreater(tria.shape[0], 100)
        self.assertFalse(np.any(np.isnan(vert)))

        ee = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])
        ee.sort(axis=1)
        ee = np.unique(ee, axis=0)
        ll = sphdist(vert[ee[:, 0]], vert[ee[:, 1]])
        ratio = ll / h_m
        # sizes must be driven by the metric hfun (same envelope the planar
        # algorithm guarantees for planar lengths)
        self.assertLess(np.median(ratio), 1.35)
        self.assertLess(np.percentile(ratio, 99), 1.55)
        self.assertGreater(np.median(ratio), 0.5)

    def test_spherical_default_off_matches_planar(self):
        """opts without 'spherical' must run the (pinned) planar code path."""
        node = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        v1, c1, t1, n1 = refine(node, edge, [], {"disp": np.inf}, 0.2)
        v2, c2, t2, n2 = refine(
            node, edge, [], {"disp": np.inf, "spherical": False}, 0.2
        )
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(t1, t2)


if __name__ == "__main__":
    unittest.main()
