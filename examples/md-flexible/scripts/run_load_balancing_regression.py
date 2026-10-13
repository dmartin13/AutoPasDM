#!/usr/bin/env python3
"""Run the same adaptive md-flexible case with reference md-flexible and AutoPasDM.

The comparison deliberately does not require identical adaptive subdomain boundaries:
Inverted Pressure uses measured local computation times, so exact boundaries can differ
between implementations and runs. Instead this regression checks that

1. both runs actually move at least one subdomain boundary, and
2. the final owned-particle state agrees by particle ID within tolerance.

Example:
    python3 scripts/run_load_balancing_regression.py \
        --reference /path/to/AutoPas/build/examples/md-flexible/md-flexible \
        --distributed ./md-flexible \
        --config input/autopasdm_load_balancing_compare.yaml \
        --ranks 8

For SLURM, e.g. use:
    --launcher 'srun -n {ranks}'
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


PARTICLE_FILE_RE = re.compile(r"_Particles_(\d+)_(\d+)\.vtu$")
RANK_FILE_RE = re.compile(r"_Ranks_(\d+)_(\d+)\.vtu$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Original md-flexible executable")
    parser.add_argument("--distributed", required=True, type=Path, help="AutoPasDM md-flexible executable")
    parser.add_argument("--config", required=True, type=Path, help="YAML input shared by both runs")
    parser.add_argument("--ranks", type=int, default=8, help="MPI ranks (default: 8)")
    parser.add_argument(
        "--launcher",
        default="mpirun -np {ranks}",
        help="MPI launcher command. {ranks} is replaced with --ranks (default: 'mpirun -np {ranks}')",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--boundary-move-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--keep-output",
        type=Path,
        help="Keep both run directories below this directory instead of deleting temporary output",
    )
    return parser.parse_args()


def absolute_existing_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{description} does not exist: {path}")
    return path


def launcher_command(template: str, ranks: int) -> list[str]:
    if ranks <= 0:
        raise RuntimeError("--ranks must be greater than zero")
    return shlex.split(template.replace("{ranks}", str(ranks)))


def run_case(label: str, executable: Path, config: Path, run_directory: Path, launcher: list[str]) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    command = [*launcher, str(executable), "--yaml-filename", str(config)]
    log_file = run_directory / "run.log"

    print(f"Running {label}:")
    print("  " + shlex.join(command))
    print(f"  cwd: {run_directory}")

    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=run_directory, stdout=log, stderr=subprocess.STDOUT, check=False)

    if result.returncode != 0:
        tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}. Last log lines:\n" + "\n".join(tail)
        )


def data_array(piece: ET.Element, name: str) -> list[str]:
    for element in piece.iter("DataArray"):
        if element.attrib.get("Name") == name:
            return (element.text or "").split()
    raise RuntimeError(f'DataArray "{name}" not found')


def point_coordinates(piece: ET.Element) -> list[float]:
    points = next(piece.iter("Points"), None)
    if points is None:
        raise RuntimeError("Points element not found")
    array = next(points.iter("DataArray"), None)
    if array is None:
        raise RuntimeError("Points DataArray not found")
    return [float(value) for value in (array.text or "").split()]


def matching_files(root: Path, pattern: re.Pattern[str]) -> list[tuple[int, int, Path]]:
    matches: list[tuple[int, int, Path]] = []
    for filename in glob.glob(str(root / "**" / "*.vtu"), recursive=True):
        match = pattern.search(filename)
        if match:
            rank = int(match.group(1))
            iteration = int(match.group(2))
            matches.append((iteration, rank, Path(filename)))
    if not matches:
        raise RuntimeError(f"No matching VTK files found below {root}")
    return matches


def first_and_final_iterations(root: Path, pattern: re.Pattern[str]) -> tuple[int, int]:
    iterations = [iteration for iteration, _, _ in matching_files(root, pattern)]
    return min(iterations), max(iterations)


def files_at_iteration(root: Path, pattern: re.Pattern[str], iteration: int) -> list[tuple[int, Path]]:
    return sorted(
        (rank, filename)
        for file_iteration, rank, filename in matching_files(root, pattern)
        if file_iteration == iteration
    )


def read_particle_state(root: Path) -> tuple[int, dict[int, dict[str, object]]]:
    _, final_iteration = first_and_final_iterations(root, PARTICLE_FILE_RE)
    result: dict[int, dict[str, object]] = {}

    for rank, filename in files_at_iteration(root, PARTICLE_FILE_RE, final_iteration):
        tree = ET.parse(filename)
        piece = next(tree.getroot().iter("Piece"))

        ids = [int(value) for value in data_array(piece, "ids")]
        type_ids = [int(value) for value in data_array(piece, "typeIds")]
        forces = [float(value) for value in data_array(piece, "forces")]
        velocities = [float(value) for value in data_array(piece, "velocities")]
        positions = point_coordinates(piece)

        expected_vector_values = 3 * len(ids)
        if len(type_ids) != len(ids):
            raise RuntimeError(f"Type-ID/ID length mismatch in {filename}")
        for values, name in ((forces, "force"), (velocities, "velocity"), (positions, "position")):
            if len(values) != expected_vector_values:
                raise RuntimeError(f"{name}/ID length mismatch in {filename}")

        for index, particle_id in enumerate(ids):
            if particle_id in result:
                raise RuntimeError(f"Particle ID {particle_id} occurs more than once in final owned output")
            begin = 3 * index
            result[particle_id] = {
                "rank": rank,
                "type": type_ids[index],
                "position": tuple(positions[begin : begin + 3]),
                "velocity": tuple(velocities[begin : begin + 3]),
                "force": tuple(forces[begin : begin + 3]),
            }

    return final_iteration, result


def read_boxes(root: Path, iteration: int) -> dict[int, tuple[tuple[float, ...], tuple[float, ...]]]:
    boxes: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for rank, filename in files_at_iteration(root, RANK_FILE_RE, iteration):
        tree = ET.parse(filename)
        piece = next(tree.getroot().iter("Piece"))
        coordinates = point_coordinates(piece)
        if len(coordinates) != 24:
            raise RuntimeError(f"Expected 8 three-dimensional rank-box points in {filename}")
        points = [coordinates[index : index + 3] for index in range(0, len(coordinates), 3)]
        box_min = tuple(min(point[d] for point in points) for d in range(3))
        box_max = tuple(max(point[d] for point in points) for d in range(3))
        boxes[rank] = (box_min, box_max)
    return boxes


def maximum_boundary_movement(root: Path) -> tuple[int, int, float]:
    initial_iteration, final_iteration = first_and_final_iterations(root, RANK_FILE_RE)
    initial_boxes = read_boxes(root, initial_iteration)
    final_boxes = read_boxes(root, final_iteration)

    if set(initial_boxes) != set(final_boxes):
        raise RuntimeError("Rank IDs differ between initial and final domain output")

    movement = 0.0
    for rank in initial_boxes:
        for initial, final in zip(initial_boxes[rank][0] + initial_boxes[rank][1], final_boxes[rank][0] + final_boxes[rank][1]):
            movement = max(movement, abs(initial - final))
    return initial_iteration, final_iteration, movement


def close(a: float, b: float, absolute_tolerance: float, relative_tolerance: float) -> bool:
    return math.isclose(a, b, abs_tol=absolute_tolerance, rel_tol=relative_tolerance)


def compare_states(
    reference: dict[int, dict[str, object]],
    distributed: dict[int, dict[str, object]],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[bool, int, float]:
    reference_ids = set(reference)
    distributed_ids = set(distributed)
    if reference_ids != distributed_ids:
        print("FAIL: final owned particle IDs differ.")
        print("  only in reference  :", sorted(reference_ids - distributed_ids))
        print("  only in distributed:", sorted(distributed_ids - reference_ids))
        return False, 0, math.inf

    ownership_changes = 0
    maximum_difference = 0.0
    failures: list[str] = []

    for particle_id in sorted(reference_ids):
        expected = reference[particle_id]
        actual = distributed[particle_id]

        if expected["rank"] != actual["rank"]:
            ownership_changes += 1
        if expected["type"] != actual["type"]:
            failures.append(f"id={particle_id}: type {expected['type']} != {actual['type']}")

        for field in ("position", "velocity", "force"):
            expected_vector = expected[field]
            actual_vector = actual[field]
            for component, (a, b) in enumerate(zip(expected_vector, actual_vector)):
                maximum_difference = max(maximum_difference, abs(a - b))
                if not close(a, b, absolute_tolerance, relative_tolerance):
                    failures.append(
                        f"id={particle_id}: {field}[{component}] reference={a:.12g} distributed={b:.12g}"
                    )

    if failures:
        print("FAIL: final particle states differ beyond tolerance.")
        for failure in failures[:20]:
            print("  " + failure)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return False, ownership_changes, maximum_difference

    return True, ownership_changes, maximum_difference


def main() -> int:
    args = parse_args()

    try:
        reference = absolute_existing_file(args.reference, "Reference executable")
        distributed = absolute_existing_file(args.distributed, "Distributed executable")
        config = absolute_existing_file(args.config, "Configuration")
        launcher = launcher_command(args.launcher, args.ranks)

        if args.keep_output:
            work_root = args.keep_output.expanduser().resolve()
            if work_root.exists():
                shutil.rmtree(work_root)
            work_root.mkdir(parents=True)
            cleanup = None
        else:
            cleanup = tempfile.TemporaryDirectory(prefix="autopasdm_load_balancing_regression_")
            work_root = Path(cleanup.name)

        reference_root = work_root / "reference"
        distributed_root = work_root / "distributed"

        run_case("reference md-flexible", reference, config, reference_root, launcher)
        run_case("AutoPasDM md-flexible", distributed, config, distributed_root, launcher)

        ref_initial, ref_final, ref_movement = maximum_boundary_movement(reference_root)
        autopasdm_initial, autopasdm_final, autopasdm_movement = maximum_boundary_movement(distributed_root)

        print("\nAdaptive-domain check:")
        print(f"  reference   iterations {ref_initial}->{ref_final}, max boundary movement {ref_movement:.12g}")
        print(f"  distributed iterations {autopasdm_initial}->{autopasdm_final}, max boundary movement {autopasdm_movement:.12g}")

        boundaries_ok = True
        if ref_movement <= args.boundary_move_tolerance:
            print("FAIL: reference md-flexible did not measurably change its decomposition.")
            boundaries_ok = False
        if autopasdm_movement <= args.boundary_move_tolerance:
            print("FAIL: AutoPasDM did not measurably change its decomposition.")
            boundaries_ok = False

        ref_iteration, ref_particles = read_particle_state(reference_root)
        autopasdm_iteration, autopasdm_particles = read_particle_state(distributed_root)

        print("\nFinal-state comparison:")
        print(f"  reference final iteration   : {ref_iteration}")
        print(f"  distributed final iteration : {autopasdm_iteration}")
        print(f"  reference owned particles   : {len(ref_particles)}")
        print(f"  distributed owned particles : {len(autopasdm_particles)}")

        states_ok, ownership_changes, maximum_difference = compare_states(
            ref_particles,
            autopasdm_particles,
            args.absolute_tolerance,
            args.relative_tolerance,
        )
        print(f"  particles owned by different ranks: {ownership_changes} (diagnostic only)")
        print(f"  maximum absolute component diff   : {maximum_difference:.12g}")
        print(f"  absolute tolerance                : {args.absolute_tolerance:.12g}")
        print(f"  relative tolerance                : {args.relative_tolerance:.12g}")

        if boundaries_ok and states_ok:
            print("\nPASS: both decompositions adapt and the final particle state agrees within tolerance.")
            result = 0
        else:
            print("\nFAIL: load-balancing regression failed.")
            result = 1

        if args.keep_output:
            print(f"Output kept in: {work_root}")
        if cleanup is not None:
            cleanup.cleanup()
        return result

    except (OSError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
