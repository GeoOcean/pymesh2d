"""
Regression smoke test for `smood` (orthogonalization + small-flow-link merge).

`smood` has no coverage in the `tridemo`-based demo suite. This test pins
down its numerical output on a small, fast lon/lat mesh so that the planned
deduplication of `pymesh2d.ortho_merge` / `pymesh2d.smood` internals cannot
silently change results. Run `tests/generate_smood_reference.py` to
(re)generate the reference after an intentional behaviour change.
"""
import unittest

import numpy as np

from pymesh2d.smood import smood

from tests.smood_case import build_smood_input, SMOOD_OPTS
from tests.test_helpers import compare_meshes, load_reference_data


class TestSmood(unittest.TestCase):
    def test_smood_matches_reference(self):
        vert, conn, tria, tnum = build_smood_input()
        vert_s, conn_s, tria_s, tnum_s = smood(vert, conn, tria, tnum, dict(SMOOD_OPTS))

        vert_ref, tria_ref = load_reference_data("_smood")
        is_equal, message = compare_meshes(vert_s, tria_s[:, 0:3], vert_ref, tria_ref)
        self.assertTrue(is_equal, message)

    def test_smood_output_is_a_valid_mesh(self):
        """Basic sanity checks independent of the pinned reference."""
        vert, conn, tria, tnum = build_smood_input()
        vert_s, conn_s, tria_s, tnum_s = smood(vert, conn, tria, tnum, dict(SMOOD_OPTS))

        self.assertFalse(np.any(np.isnan(vert_s)))
        self.assertTrue(np.all(tria_s[:, 0:3] >= 0))
        self.assertTrue(np.all(tria_s[:, 0:3] < vert_s.shape[0]))

        # No degenerate (zero-area) triangles.
        p0 = vert_s[tria_s[:, 0]]
        p1 = vert_s[tria_s[:, 1]]
        p2 = vert_s[tria_s[:, 2]]
        area2 = np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )
        self.assertTrue(np.all(area2 > 0))

    def test_smood_triangles_only_mode(self):
        """
        `merge_small_links=False` must keep the mesh pure triangles (no quad
        merging) while still targeting the same dual criteria via guarded
        flips and node movement.
        """
        import numpy as np

        from pymesh2d.ortho_merge import meshkernel_orthogonalize_3 as mk3
        from pymesh2d.ortho_merge.geometry import build_edges_from_tria

        vert, conn, tria, tnum = build_smood_input()
        opts = dict(SMOOD_OPTS)
        opts["merge_small_links"] = False
        opts["iter"] = 4
        vert_s, conn_s, tria_s, tnum_s = smood(vert, conn, tria, tnum, opts)

        # Pure triangles: no quad rows and same triangle count as the input.
        self.assertEqual(tria_s.shape[1], 3)
        self.assertEqual(tria_s.shape[0], tria.shape[0])
        self.assertFalse(np.any(np.isnan(vert_s)))

        # Dual criteria on the triangle output.
        tt = np.asarray(tria_s, dtype=np.int64)
        en, ef = build_edges_from_tria(tt)
        _, _, cos = mk3.compute_cosphi_abs_from_arrays(
            vert_s[:, 0], vert_s[:, 1], tt, en, ef, use_circumcenter_3d=True
        )
        n_small, _ = mk3.compute_small_links_from_arrays(
            vert_s[:, 0], vert_s[:, 1], tt, en, ef, removesmalllinkstrsh=0.11
        )
        self.assertLessEqual(float(np.nanmax(cos)), 0.49 + 1e-9)
        self.assertEqual(int(n_small), 0)

    def test_smood_planar_mode_on_projected_mesh(self):
        """
        `spherical=False` runs the pipeline with plain 2D geometry so a mesh
        already in a projected (metres) CRS is handled directly. Project the
        lon/lat demo mesh to a local equirectangular frame and check the output
        is a valid, non-degenerate mesh.
        """
        vert, conn, tria, tnum = build_smood_input()

        # Local equirectangular projection (lon/lat degrees -> metres).
        lat0 = float(np.mean(vert[:, 1]))
        m_per_deg = 111_320.0
        vert_m = np.column_stack(
            [
                (vert[:, 0] - np.mean(vert[:, 0])) * m_per_deg * np.cos(np.radians(lat0)),
                (vert[:, 1] - lat0) * m_per_deg,
            ]
        )

        opts = dict(SMOOD_OPTS)
        opts["spherical"] = False
        vert_s, conn_s, tria_s, tnum_s = smood(vert_m, conn, tria, tnum, opts)

        self.assertFalse(np.any(np.isnan(vert_s)))
        self.assertTrue(np.all(tria_s[:, 0:3] >= 0))
        self.assertTrue(np.all(tria_s[:, 0:3] < vert_s.shape[0]))

        p0 = vert_s[tria_s[:, 0]]
        p1 = vert_s[tria_s[:, 1]]
        p2 = vert_s[tria_s[:, 2]]
        area2 = np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )
        self.assertTrue(np.all(area2 > 0))


if __name__ == "__main__":
    unittest.main()
