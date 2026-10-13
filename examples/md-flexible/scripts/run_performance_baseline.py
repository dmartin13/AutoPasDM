#!/usr/bin/env python3
"""Benchmark original md-flexible against the AutoPasDM integration.

The runner executes the same fixed md-flexible configuration several times and reports
medians for wall-clock, simulation, force, and distributed-synchronization time.

For the reference implementation, synchronization is reconstructed from md-flexible's
existing accumulated timers:
  UpdateContainer + LoadBalancing + MigratingParticleExchange + HaloParticleExchange

For the AutoPasDM implementation, the integration exposes one aggregated
DistributedSynchronization timer because these operations are intentionally encapsulated.

Example:
    python3 scripts/run_performance_baseline.py \
        --reference /path/to/AutoPas/build/examples/md-flexible/md-flexible \
        --distributed ./md-flexible \
        --config input/autopasdm_performance_baseline.yaml \
        --ranks 8 \
        --repetitions 5

For SLURM, e.g. use:
    --launcher 'srun -n {ranks}'
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tempfile
import time


TIMER_RE = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s+ns\b", re.MULTILINE)


@dataclass
class RunResult:
    implementation: str
    repetition: int
    external_seconds: float
    wall_clock_seconds: float
    simulate_accumulated_seconds: float
    force_accumulated_seconds: float
    synchronization_accumulated_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Original MPI-enabled md-flexible executable")
    parser.add_argument("--distributed", required=True, type=Path, help="AutoPasDM md-flexible executable")
    parser.add_argument("--config", required=True, type=Path, help="YAML input shared by both implementations")
    parser.add_argument("--ranks", type=int, default=8, help="MPI ranks (default: 8)")
    parser.add_argument("--repetitions", type=int, default=5, help="Measured repetitions per implementation (default: 5)")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up repetitions per implementation (default: 1)")
    parser.add_argument(
        "--launcher",
        default="mpirun -np {ranks}",
        help="MPI launcher. {ranks} is replaced with --ranks (default: 'mpirun -np {ranks}')",
    )
    parser.add_argument("--csv", type=Path, help="Optional path for per-run CSV output")
    parser.add_argument("--keep-logs", type=Path, help="Optional directory in which all run logs are kept")
    return parser.parse_args()


def existing_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{description} does not exist: {path}")
    return path


def launcher_command(template: str, ranks: int) -> list[str]:
    if ranks <= 0:
        raise RuntimeError("--ranks must be greater than zero")
    return shlex.split(template.replace("{ranks}", str(ranks)))


def parse_timer_ns(log: str, timer_name: str, *, required: bool = True) -> int:
    timers = {name.strip(): int(value) for name, value in TIMER_RE.findall(log)}
    if timer_name in timers:
        return timers[timer_name]
    if required:
        available = ", ".join(sorted(timers))
        raise RuntimeError(f'Could not find timer "{timer_name}". Available timers: {available}')
    return 0


def synchronization_ns(log: str, implementation: str) -> int:
    if implementation == "distributed":
        return parse_timer_ns(log, "DistributedSynchronization")

    return sum(
        parse_timer_ns(log, name, required=False)
        for name in (
            "UpdateContainer",
            "LoadBalancing",
            "MigratingParticleExchange",
            "HaloParticleExchange",
        )
    )


def run_once(
    implementation: str,
    repetition: int,
    executable: Path,
    config: Path,
    launcher: list[str],
    run_directory: Path,
) -> RunResult:
    run_directory.mkdir(parents=True, exist_ok=True)
    command = [*launcher, str(executable), "--yaml-filename", str(config)]

    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=run_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    external_seconds = time.perf_counter() - start

    log_file = run_directory / "run.log"
    log_file.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        tail = result.stdout.splitlines()[-40:]
        raise RuntimeError(
            f"{implementation} run failed with exit code {result.returncode}. Last log lines:\n" + "\n".join(tail)
        )

    wall_clock_ns = parse_timer_ns(result.stdout, "Total wall-clock time")
    simulate_ns = parse_timer_ns(result.stdout, "Simulate")
    force_ns = parse_timer_ns(result.stdout, "ForceUpdateTotal")
    sync_ns = synchronization_ns(result.stdout, implementation)

    return RunResult(
        implementation=implementation,
        repetition=repetition,
        external_seconds=external_seconds,
        wall_clock_seconds=wall_clock_ns * 1e-9,
        simulate_accumulated_seconds=simulate_ns * 1e-9,
        force_accumulated_seconds=force_ns * 1e-9,
        synchronization_accumulated_seconds=sync_ns * 1e-9,
    )


def median(results: list[RunResult], attribute: str) -> float:
    return statistics.median(getattr(result, attribute) for result in results)


def print_summary(reference: list[RunResult], distributed: list[RunResult], ranks: int) -> None:
    metrics = (
        ("external wall time", "external_seconds", False),
        ("md-flex wall-clock", "wall_clock_seconds", False),
        ("simulate / rank", "simulate_accumulated_seconds", True),
        ("force / rank", "force_accumulated_seconds", True),
        ("synchronization / rank", "synchronization_accumulated_seconds", True),
    )

    print("\nMedian performance baseline:")
    print(f"  MPI ranks: {ranks}")
    print(f"  measured repetitions: {len(reference)}")
    print()
    print(f"  {'metric':28s} {'reference [s]':>15s} {'distributed [s]':>17s} {'ref/autopasdm':>10s}")
    print(f"  {'-' * 28} {'-' * 15} {'-' * 17} {'-' * 10}")

    for label, attribute, accumulated in metrics:
        ref = median(reference, attribute)
        autopasdm = median(distributed, attribute)
        if accumulated:
            ref /= ranks
            autopasdm /= ranks
        ratio = ref / autopasdm if autopasdm > 0 else float("inf")
        print(f"  {label:28s} {ref:15.6f} {autopasdm:17.6f} {ratio:10.3f}")

    print("\nNotes:")
    print("  ref/autopasdm > 1 means AutoPasDM was faster for that metric.")
    print("  '/ rank' metrics are accumulated MPI-rank timers divided by the number of ranks.")
    print("  Reference synchronization = updateContainer + load balancing + migration + halo exchange.")
    print("  Distributed synchronization is the aggregated prepareInteractions() phase.")


def write_csv(path: Path, results: list[RunResult], ranks: int) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "implementation",
                "repetition",
                "ranks",
                "external_seconds",
                "wall_clock_seconds",
                "simulate_accumulated_seconds",
                "force_accumulated_seconds",
                "synchronization_accumulated_seconds",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.implementation,
                    result.repetition,
                    ranks,
                    f"{result.external_seconds:.12g}",
                    f"{result.wall_clock_seconds:.12g}",
                    f"{result.simulate_accumulated_seconds:.12g}",
                    f"{result.force_accumulated_seconds:.12g}",
                    f"{result.synchronization_accumulated_seconds:.12g}",
                ]
            )
    print(f"\nWrote per-run measurements to {path}")


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0:
        raise RuntimeError("--repetitions must be greater than zero")
    if args.warmups < 0:
        raise RuntimeError("--warmups must not be negative")

    reference = existing_file(args.reference, "Reference executable")
    distributed = existing_file(args.distributed, "Distributed executable")
    config = existing_file(args.config, "Configuration")
    launcher = launcher_command(args.launcher, args.ranks)

    if args.keep_logs:
        root = args.keep_logs.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="autopasdm_performance_baseline_")
        root = Path(temporary.name)

    implementations = {
        "reference": reference,
        "distributed": distributed,
    }

    # Warm up both binaries once before collecting measurements. Alternate order to
    # avoid always giving the same implementation the colder/hotter machine state.
    for warmup in range(args.warmups):
        order = ("reference", "distributed") if warmup % 2 == 0 else ("distributed", "reference")
        for implementation in order:
            print(f"Warm-up {warmup + 1}/{args.warmups}: {implementation}")
            run_once(
                implementation,
                -(warmup + 1),
                implementations[implementation],
                config,
                launcher,
                root / "warmup" / f"{warmup:02d}_{implementation}",
            )

    measured: dict[str, list[RunResult]] = {"reference": [], "distributed": []}
    all_results: list[RunResult] = []

    for repetition in range(args.repetitions):
        order = ("reference", "distributed") if repetition % 2 == 0 else ("distributed", "reference")
        for implementation in order:
            print(f"Measured run {repetition + 1}/{args.repetitions}: {implementation}")
            result = run_once(
                implementation,
                repetition,
                implementations[implementation],
                config,
                launcher,
                root / "measured" / f"{repetition:02d}_{implementation}",
            )
            measured[implementation].append(result)
            all_results.append(result)

    print_summary(measured["reference"], measured["distributed"], args.ranks)

    if args.csv:
        write_csv(args.csv, all_results, args.ranks)

    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
