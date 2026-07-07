"""
Generate the regression reference for the `smood` smoke test.

Run this script whenever `smood`'s numerical behaviour is intentionally
changed (it should otherwise stay stable across refactors of
`pymesh2d.ortho_merge` / `pymesh2d.smood`).
"""
from pymesh2d.smood import smood

from tests.smood_case import build_smood_input, SMOOD_OPTS
from tests.test_helpers import save_reference_data


def main():
    vert, conn, tria, tnum = build_smood_input()
    vert_s, conn_s, tria_s, tnum_s = smood(vert, conn, tria, tnum, dict(SMOOD_OPTS))
    save_reference_data(vert_s, tria_s[:, 0:3], "_smood")
    print(f"Saved smood reference: {vert_s.shape[0]} vertices, {tria_s.shape[0]} triangles")


if __name__ == "__main__":
    main()
