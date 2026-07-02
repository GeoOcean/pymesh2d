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


if __name__ == "__main__":
    unittest.main()
