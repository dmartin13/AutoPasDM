#!/usr/bin/env python3
"""Check the expected forces of autopasdm_reflective_compare.yaml."""

import sys

from compare_vtk_forces import read_forces


EXPECTED = {
    0: (24.0, 0.0, 0.0),
    1: (-24.0, 0.0, 0.0),
    2: (0.0, 24.0, 0.0),
    3: (0.0, -24.0, 0.0),
    4: (0.0, 0.0, 24.0),
    5: (0.0, 0.0, -24.0),
    6: (24.0, 24.0, 24.0),
    7: (-24.0, -24.0, -24.0),
}


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: check_reflective_forces.py <output-root> [tolerance]")
        return 2

    tolerance = float(sys.argv[2]) if len(sys.argv) == 3 else 1e-5
    iteration, forces = read_forces(sys.argv[1])

    if set(forces) != set(EXPECTED):
        print("FAIL: particle IDs differ from the reflective regression setup.")
        print("  expected:", sorted(EXPECTED))
        print("  actual  :", sorted(forces))
        return 1

    max_abs_diff = 0.0
    for particle_id, expected in EXPECTED.items():
        actual, rank = forces[particle_id]
        diff = max(abs(a - b) for a, b in zip(actual, expected))
        max_abs_diff = max(max_abs_diff, diff)
        print(f"id={particle_id:2d} rank={rank} actual={actual} expected={expected} diff={diff:.12g}")

    print(f"\nFinal iteration             : {iteration}")
    print(f"Maximum absolute force diff : {max_abs_diff:.12g}")
    print(f"Tolerance                   : {tolerance:.12g}")

    if max_abs_diff <= tolerance:
        print("PASS: reflective wall forces match the expected values.")
        return 0

    print("FAIL: reflective wall force difference exceeds tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
