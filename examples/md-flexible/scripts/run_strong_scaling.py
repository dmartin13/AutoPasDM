#!/usr/bin/env python3
"""Strong-scaling benchmark for original md-flexible and AutoPasDM.

The runner executes the same static md-flexible configuration for several MPI rank
counts. For each implementation and rank count it reports median simulation time,
strong-scaling speedup, parallel efficiency, force time, and synchronization time.

The primary scaling metric is md-flexible's accumulated ``Simulate`` timer divided
by the number of MPI ranks. This avoids including MPI launcher startup in the
speedup calculation. External and total wall-clock times are retained as diagnostics.

Reference synchronization is reconstructed from md-flexible's existing timers:
  UpdateContainer + LoadBalancing + MigratingParticleExchange + HaloParticleExchange

AutoPasDM exposes the corresponding complete ``prepareInteractions()``
phase as ``DistributedSynchronization``.

Example:
    python3 scripts/run_strong_scaling.py \
        --reference /path/to/AutoPas/build/examples/md-flexible/md-flexible \
        --distributed ./md-flexible \
        --config input/autopasdm_performance_baseline.yaml \
        --ranks 1,2,4,8,16,32 \
        --repetitions 5

For SLURM, for example, use:
    --launcher 'srun -n {ranks}'
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tempfile
import time


TIMER_RE = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s+ns\b", re.MULTILINE)
COUNTER_RE = re.compile(r"^\s*(.*?)\s*:\s*(\d+)\s*$", re.MULTILINE)

AUTOPASDM_COARSE_TIMERS = {
    "core_sync": "AutoPasDM CoreSynchronization",
    "update_container": "AutoPasDM UpdateContainer",
    "load_balancing": "AutoPasDM LoadBalancing",
    "boundary_sync": "AutoPasDM BoundarySync",
    "resize_box": "AutoPasDM ResizeBox",
    "migration": "AutoPasDM Migration",
    "add_migrated": "AutoPasDM AddMigratedParticles",
    "collect_owned": "AutoPasDM CollectOwnedParticles",
    "halo_exchange": "AutoPasDM HaloExchange",
    "add_halos": "AutoPasDM AddHaloParticles",
}

AUTOPASDM_DETAILED_TIMERS = {
    "migration_classify": "AutoPasDM MigrationClassify",
    "halo_initial_copy": "AutoPasDM HaloInitialCopy",
    "halo_selection": "AutoPasDM HaloSelection",
    "halo_periodic_shift": "AutoPasDM HaloPeriodicShift",
    "halo_buffer_insertion": "AutoPasDM HaloBufferInsertion",
    "comm_pack": "AutoPasDM CommunicationPack",
    "count_exchange": "AutoPasDM CountExchange",
    "payload_wait": "AutoPasDM PayloadWait",
    "comm_unpack": "AutoPasDM CommunicationUnpack",
}

AUTOPASDM_COUNTERS = {
    "migration_sent": "Migration particles sent",
    "migration_received": "Migration particles received",
    "halo_sent": "Halo particles sent",
    "halo_received": "Halo particles received",
    "bytes_sent": "Payload bytes sent",
    "bytes_received": "Payload bytes received",
}


@dataclass
class RunResult:
    implementation: str
    ranks: int
    repetition: int
    external_seconds: float
    wall_clock_seconds: float
    simulate_accumulated_seconds: float
    force_accumulated_seconds: float
    synchronization_accumulated_seconds: float
    autopasdm_timer_accumulated_seconds: dict[str, float] = field(default_factory=dict)
    autopasdm_counters: dict[str, int] = field(default_factory=dict)

    @property
    def simulate_seconds_per_rank(self) -> float:
        return self.simulate_accumulated_seconds / self.ranks

    @property
    def force_seconds_per_rank(self) -> float:
        return self.force_accumulated_seconds / self.ranks

    @property
    def synchronization_seconds_per_rank(self) -> float:
        return self.synchronization_accumulated_seconds / self.ranks

    def autopasdm_timer_seconds_per_rank(self, timer: str) -> float:
        return self.autopasdm_timer_accumulated_seconds[timer] / self.ranks


@dataclass
class MedianResult:
    implementation: str
    ranks: int
    external_seconds: float
    wall_clock_seconds: float
    simulate_seconds_per_rank: float
    force_seconds_per_rank: float
    synchronization_seconds_per_rank: float
    autopasdm_timer_seconds_per_rank: dict[str, float] = field(default_factory=dict)
    autopasdm_counters: dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Original MPI-enabled md-flexible executable")
    parser.add_argument("--distributed", required=True, type=Path, help="AutoPasDM md-flexible executable")
    parser.add_argument("--config", required=True, type=Path, help="Static YAML input shared by both implementations")
    parser.add_argument(
        "--ranks",
        default="1,2,4,8,16,32",
        help="Comma-separated MPI rank counts (default: 1,2,4,8,16,32)",
    )
    parser.add_argument("--repetitions", type=int, default=5, help="Measured repetitions per point (default: 5)")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up repetitions per point (default: 1)")
    parser.add_argument(
        "--launcher",
        default="mpirun -np {ranks}",
        help="MPI launcher. {ranks} is replaced with the current rank count (default: 'mpirun -np {ranks}')",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=1,
        help="OMP_NUM_THREADS used for every MPI rank (default: 1)",
    )
    parser.add_argument("--csv", type=Path, help="Optional path for per-run CSV measurements")
    parser.add_argument("--summary-csv", type=Path, help="Optional path for median scaling results")
    parser.add_argument("--keep-logs", type=Path, help="Optional directory in which all run logs are kept")
    return parser.parse_args()


def parse_rank_counts(value: str) -> list[int]:
    try:
        ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise RuntimeError(f"Invalid --ranks value: {value}") from error

    if not ranks or any(rank <= 0 for rank in ranks):
        raise RuntimeError("--ranks must contain positive integers")
    if len(set(ranks)) != len(ranks):
        raise RuntimeError("--ranks must not contain duplicates")
    return sorted(ranks)


def existing_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{description} does not exist: {path}")
    return path


def launcher_command(template: str, ranks: int) -> list[str]:
    return shlex.split(template.replace("{ranks}", str(ranks)))


def parse_timer_ns(log: str, timer_name: str, *, required: bool = True) -> int:
    timers = {name.strip(): int(value) for name, value in TIMER_RE.findall(log)}
    if timer_name in timers:
        return timers[timer_name]
    if required:
        available = ", ".join(sorted(timers))
        raise RuntimeError(f'Could not find timer "{timer_name}". Available timers: {available}')
    return 0


def parse_counter(log: str, counter_name: str, *, required: bool = True) -> int:
    counters = {name.strip(): int(value) for name, value in COUNTER_RE.findall(log)}
    if counter_name in counters:
        return counters[counter_name]
    if required:
        available = ", ".join(sorted(counters))
        raise RuntimeError(f'Could not find counter "{counter_name}". Available counters: {available}')
    return 0


def parse_optional_autopasdm_timers(log: str) -> dict[str, float]:
    timers: dict[str, float] = {}
    for key, label in {**AUTOPASDM_COARSE_TIMERS, **AUTOPASDM_DETAILED_TIMERS}.items():
        value_ns = parse_timer_ns(log, label, required=False)
        if value_ns or re.search(rf"^\s*{re.escape(label)}\s*:", log, re.MULTILINE):
            timers[key] = value_ns * 1e-9
    return timers


def parse_optional_autopasdm_counters(log: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    for key, label in AUTOPASDM_COUNTERS.items():
        if re.search(rf"^\s*{re.escape(label)}\s*:", log, re.MULTILINE):
            counters[key] = parse_counter(log, label)
    return counters


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
    ranks: int,
    repetition: int,
    executable: Path,
    config: Path,
    launcher_template: str,
    omp_threads: int,
    run_directory: Path,
) -> RunResult:
    run_directory.mkdir(parents=True, exist_ok=True)
    command = [*launcher_command(launcher_template, ranks), str(executable), "--yaml-filename", str(config)]

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(omp_threads)

    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=run_directory,
        env=environment,
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
            f"{implementation} run with {ranks} ranks failed with exit code {result.returncode}. "
            f"Last log lines:\n" + "\n".join(tail)
        )

    return RunResult(
        implementation=implementation,
        ranks=ranks,
        repetition=repetition,
        external_seconds=external_seconds,
        wall_clock_seconds=parse_timer_ns(result.stdout, "Total wall-clock time") * 1e-9,
        simulate_accumulated_seconds=parse_timer_ns(result.stdout, "Simulate") * 1e-9,
        force_accumulated_seconds=parse_timer_ns(result.stdout, "ForceUpdateTotal") * 1e-9,
        synchronization_accumulated_seconds=synchronization_ns(result.stdout, implementation) * 1e-9,
        autopasdm_timer_accumulated_seconds=parse_optional_autopasdm_timers(result.stdout) if implementation == "distributed" else {},
        autopasdm_counters=parse_optional_autopasdm_counters(result.stdout) if implementation == "distributed" else {},
    )


def median_results(results: list[RunResult]) -> dict[tuple[str, int], MedianResult]:
    grouped: dict[tuple[str, int], list[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.implementation, result.ranks), []).append(result)

    medians: dict[tuple[str, int], MedianResult] = {}
    for (implementation, ranks), group in grouped.items():
        autopasdm_timer_keys = sorted(set().union(*(result.autopasdm_timer_accumulated_seconds.keys() for result in group)))
        autopasdm_counter_keys = sorted(set().union(*(result.autopasdm_counters.keys() for result in group)))
        medians[(implementation, ranks)] = MedianResult(
            implementation=implementation,
            ranks=ranks,
            external_seconds=statistics.median(result.external_seconds for result in group),
            wall_clock_seconds=statistics.median(result.wall_clock_seconds for result in group),
            simulate_seconds_per_rank=statistics.median(result.simulate_seconds_per_rank for result in group),
            force_seconds_per_rank=statistics.median(result.force_seconds_per_rank for result in group),
            synchronization_seconds_per_rank=statistics.median(
                result.synchronization_seconds_per_rank for result in group
            ),
            autopasdm_timer_seconds_per_rank={
                key: statistics.median(
                    result.autopasdm_timer_seconds_per_rank(key)
                    for result in group
                    if key in result.autopasdm_timer_accumulated_seconds
                )
                for key in autopasdm_timer_keys
            },
            autopasdm_counters={
                key: statistics.median(result.autopasdm_counters[key] for result in group if key in result.autopasdm_counters)
                for key in autopasdm_counter_keys
            },
        )
    return medians


def scaling_values(
    medians: dict[tuple[str, int], MedianResult], implementation: str, ranks: int, baseline_ranks: int
) -> tuple[float, float]:
    baseline = medians[(implementation, baseline_ranks)].simulate_seconds_per_rank
    current = medians[(implementation, ranks)].simulate_seconds_per_rank
    speedup = baseline / current
    ideal_speedup = ranks / baseline_ranks
    efficiency = speedup / ideal_speedup
    return speedup, efficiency


def available_autopasdm_timer_keys(
    medians: dict[tuple[str, int], MedianResult], rank_counts: list[int], candidates: dict[str, str]
) -> list[str]:
    return [
        key
        for key in candidates
        if any(key in medians[("distributed", ranks)].autopasdm_timer_seconds_per_rank for ranks in rank_counts)
    ]


def print_autopasdm_timer_table(
    title: str,
    medians: dict[tuple[str, int], MedianResult],
    rank_counts: list[int],
    timer_keys: list[str],
    labels: dict[str, str],
) -> None:
    if not timer_keys:
        return

    short_labels = {
        "core_sync": "core sync",
        "update_container": "update",
        "load_balancing": "load bal",
        "boundary_sync": "boundary",
        "resize_box": "resize",
        "migration": "migration",
        "add_migrated": "add migr",
        "collect_owned": "collect",
        "halo_exchange": "halo",
        "add_halos": "add halo",
        "migration_classify": "migr class",
        "halo_selection": "halo select",
        "halo_periodic_shift": "periodic",
        "comm_pack": "pack",
        "count_exchange": "count",
        "payload_wait": "wait",
        "comm_unpack": "unpack",
    }

    print(f"\n{title} (median seconds / rank):")
    header = f"  {'ranks':>5s}" + "".join(f" {short_labels.get(key, labels[key]):>11s}" for key in timer_keys)
    print(header)
    print(f"  {'-' * 5}" + "".join(f" {'-' * 11}" for _ in timer_keys))
    for ranks in rank_counts:
        result = medians[("distributed", ranks)]
        values = []
        for key in timer_keys:
            value = result.autopasdm_timer_seconds_per_rank.get(key)
            values.append(f" {value:11.6f}" if value is not None else f" {'-':>11s}")
        print(f"  {ranks:5d}" + "".join(values))


def print_autopasdm_counter_table(
    medians: dict[tuple[str, int], MedianResult], rank_counts: list[int]
) -> None:
    keys = [
        key
        for key in AUTOPASDM_COUNTERS
        if any(key in medians[("distributed", ranks)].autopasdm_counters for ranks in rank_counts)
    ]
    if not keys:
        return

    print("\nAutoPasDM communication counters (median global sum):")
    print(
        f"  {'ranks':>5s} {'migr sent':>11s} {'migr recv':>11s} "
        f"{'halo sent':>11s} {'halo recv':>11s} {'bytes sent':>14s} {'bytes recv':>14s}"
    )
    print(f"  {'-' * 5} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 14} {'-' * 14}")
    for ranks in rank_counts:
        counters = medians[("distributed", ranks)].autopasdm_counters
        def formatted(key: str) -> str:
            value = counters.get(key)
            return "-" if value is None else f"{value:.0f}"
        print(
            f"  {ranks:5d} {formatted('migration_sent'):>11s} {formatted('migration_received'):>11s} "
            f"{formatted('halo_sent'):>11s} {formatted('halo_received'):>11s} "
            f"{formatted('bytes_sent'):>14s} {formatted('bytes_received'):>14s}"
        )


def print_summary(
    medians: dict[tuple[str, int], MedianResult], rank_counts: list[int], repetitions: int, omp_threads: int
) -> None:
    baseline_ranks = rank_counts[0]

    print("\nStrong-scaling summary:")
    print(f"  baseline MPI ranks: {baseline_ranks}")
    print(f"  measured repetitions per point: {repetitions}")
    print(f"  OpenMP threads per MPI rank: {omp_threads}")

    print("\nSimulation scaling (median Simulate / rank):")
    print(
        f"  {'ranks':>5s} {'ref [s]':>10s} {'ref speedup':>11s} {'ref eff.':>9s} "
        f"{'autopasdm [s]':>10s} {'autopasdm speedup':>11s} {'autopasdm eff.':>9s} {'ref/autopasdm':>9s}"
    )
    print(
        f"  {'-' * 5} {'-' * 10} {'-' * 11} {'-' * 9} "
        f"{'-' * 10} {'-' * 11} {'-' * 9} {'-' * 9}"
    )

    for ranks in rank_counts:
        ref = medians[("reference", ranks)]
        autopasdm = medians[("distributed", ranks)]
        ref_speedup, ref_efficiency = scaling_values(medians, "reference", ranks, baseline_ranks)
        autopasdm_speedup, autopasdm_efficiency = scaling_values(medians, "distributed", ranks, baseline_ranks)
        ratio = ref.simulate_seconds_per_rank / autopasdm.simulate_seconds_per_rank
        print(
            f"  {ranks:5d} {ref.simulate_seconds_per_rank:10.6f} {ref_speedup:11.3f} {ref_efficiency:9.3f} "
            f"{autopasdm.simulate_seconds_per_rank:10.6f} {autopasdm_speedup:11.3f} {autopasdm_efficiency:9.3f} {ratio:9.3f}"
        )

    print("\nTime breakdown (median seconds / rank):")
    print(
        f"  {'ranks':>5s} {'ref force':>11s} {'autopasdm force':>11s} "
        f"{'ref sync':>11s} {'autopasdm sync':>11s} {'sync ref/autopasdm':>12s}"
    )
    print(f"  {'-' * 5} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 11} {'-' * 12}")
    for ranks in rank_counts:
        ref = medians[("reference", ranks)]
        autopasdm = medians[("distributed", ranks)]
        sync_ratio = (
            ref.synchronization_seconds_per_rank / autopasdm.synchronization_seconds_per_rank
            if autopasdm.synchronization_seconds_per_rank > 0.0
            else float("inf")
        )
        print(
            f"  {ranks:5d} {ref.force_seconds_per_rank:11.6f} {autopasdm.force_seconds_per_rank:11.6f} "
            f"{ref.synchronization_seconds_per_rank:11.6f} {autopasdm.synchronization_seconds_per_rank:11.6f} "
            f"{sync_ratio:12.3f}"
        )

    coarse_keys = available_autopasdm_timer_keys(medians, rank_counts, AUTOPASDM_COARSE_TIMERS)
    detailed_keys = available_autopasdm_timer_keys(medians, rank_counts, AUTOPASDM_DETAILED_TIMERS)
    print_autopasdm_timer_table("AutoPasDM coarse breakdown", medians, rank_counts, coarse_keys, AUTOPASDM_COARSE_TIMERS)
    print_autopasdm_timer_table(
        "AutoPasDM detailed breakdown", medians, rank_counts, detailed_keys, AUTOPASDM_DETAILED_TIMERS
    )
    print_autopasdm_counter_table(medians, rank_counts)

    print("\nWall-clock diagnostics:")
    print(f"  {'ranks':>5s} {'ref internal':>13s} {'autopasdm internal':>13s} {'ref external':>13s} {'autopasdm external':>13s}")
    print(f"  {'-' * 5} {'-' * 13} {'-' * 13} {'-' * 13} {'-' * 13}")
    for ranks in rank_counts:
        ref = medians[("reference", ranks)]
        autopasdm = medians[("distributed", ranks)]
        print(
            f"  {ranks:5d} {ref.wall_clock_seconds:13.6f} {autopasdm.wall_clock_seconds:13.6f} "
            f"{ref.external_seconds:13.6f} {autopasdm.external_seconds:13.6f}"
        )

    print("\nNotes:")
    print("  Speedup and efficiency use the smallest requested rank count as the baseline.")
    print("  With the default rank list this is conventional T(1) strong scaling.")
    print("  ref/autopasdm > 1 means AutoPasDM was faster for that metric.")
    print("  Internal scaling uses Simulate / rank and excludes MPI launcher startup.")
    print("  OMP_NUM_THREADS is fixed for all runs so the experiment varies MPI ranks only.")
    if available_autopasdm_timer_keys(medians, rank_counts, AUTOPASDM_COARSE_TIMERS):
        print("  AutoPasDM timer values are accumulated over MPI ranks by md-flexible and divided by ranks here.")
    if any(medians[("distributed", ranks)].autopasdm_counters for ranks in rank_counts):
        print("  AutoPasDM communication counters are global sums over all MPI ranks, not per-rank values.")


def write_raw_csv(path: Path, results: list[RunResult]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    autopasdm_timer_keys = sorted(set().union(*(result.autopasdm_timer_accumulated_seconds.keys() for result in results)))
    autopasdm_counter_keys = sorted(set().union(*(result.autopasdm_counters.keys() for result in results)))

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "implementation",
                "ranks",
                "repetition",
                "external_seconds",
                "wall_clock_seconds",
                "simulate_accumulated_seconds",
                "simulate_seconds_per_rank",
                "force_accumulated_seconds",
                "force_seconds_per_rank",
                "synchronization_accumulated_seconds",
                "synchronization_seconds_per_rank",
                *[f"autopasdm_{key}_accumulated_seconds" for key in autopasdm_timer_keys],
                *[f"autopasdm_{key}_seconds_per_rank" for key in autopasdm_timer_keys],
                *[f"autopasdm_{key}" for key in autopasdm_counter_keys],
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.implementation,
                    result.ranks,
                    result.repetition,
                    f"{result.external_seconds:.12g}",
                    f"{result.wall_clock_seconds:.12g}",
                    f"{result.simulate_accumulated_seconds:.12g}",
                    f"{result.simulate_seconds_per_rank:.12g}",
                    f"{result.force_accumulated_seconds:.12g}",
                    f"{result.force_seconds_per_rank:.12g}",
                    f"{result.synchronization_accumulated_seconds:.12g}",
                    f"{result.synchronization_seconds_per_rank:.12g}",
                    *[
                        f"{result.autopasdm_timer_accumulated_seconds[key]:.12g}"
                        if key in result.autopasdm_timer_accumulated_seconds
                        else ""
                        for key in autopasdm_timer_keys
                    ],
                    *[
                        f"{result.autopasdm_timer_seconds_per_rank(key):.12g}"
                        if key in result.autopasdm_timer_accumulated_seconds
                        else ""
                        for key in autopasdm_timer_keys
                    ],
                    *[str(result.autopasdm_counters[key]) if key in result.autopasdm_counters else "" for key in autopasdm_counter_keys],
                ]
            )
    print(f"\nWrote per-run measurements to {path}")


def write_summary_csv(
    path: Path, medians: dict[tuple[str, int], MedianResult], rank_counts: list[int]
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_ranks = rank_counts[0]
    distributed_results = [medians[("distributed", ranks)] for ranks in rank_counts]
    autopasdm_timer_keys = sorted(set().union(*(result.autopasdm_timer_seconds_per_rank.keys() for result in distributed_results)))
    autopasdm_counter_keys = sorted(set().union(*(result.autopasdm_counters.keys() for result in distributed_results)))

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "implementation",
                "ranks",
                "external_seconds",
                "wall_clock_seconds",
                "simulate_seconds_per_rank",
                "speedup",
                "parallel_efficiency",
                "force_seconds_per_rank",
                "synchronization_seconds_per_rank",
                *[f"autopasdm_{key}_seconds_per_rank" for key in autopasdm_timer_keys],
                *[f"autopasdm_{key}" for key in autopasdm_counter_keys],
            ]
        )
        for ranks in rank_counts:
            for implementation in ("reference", "distributed"):
                result = medians[(implementation, ranks)]
                speedup, efficiency = scaling_values(medians, implementation, ranks, baseline_ranks)
                writer.writerow(
                    [
                        implementation,
                        ranks,
                        f"{result.external_seconds:.12g}",
                        f"{result.wall_clock_seconds:.12g}",
                        f"{result.simulate_seconds_per_rank:.12g}",
                        f"{speedup:.12g}",
                        f"{efficiency:.12g}",
                        f"{result.force_seconds_per_rank:.12g}",
                        f"{result.synchronization_seconds_per_rank:.12g}",
                        *[
                            f"{result.autopasdm_timer_seconds_per_rank[key]:.12g}"
                            if key in result.autopasdm_timer_seconds_per_rank
                            else ""
                            for key in autopasdm_timer_keys
                        ],
                        *[
                            f"{result.autopasdm_counters[key]:.12g}" if key in result.autopasdm_counters else ""
                            for key in autopasdm_counter_keys
                        ],
                    ]
                )
    print(f"Wrote median scaling results to {path}")


def main() -> int:
    args = parse_args()
    rank_counts = parse_rank_counts(args.ranks)

    if args.repetitions <= 0:
        raise RuntimeError("--repetitions must be greater than zero")
    if args.warmups < 0:
        raise RuntimeError("--warmups must not be negative")
    if args.omp_threads <= 0:
        raise RuntimeError("--omp-threads must be greater than zero")

    reference = existing_file(args.reference, "Reference executable")
    distributed = existing_file(args.distributed, "Distributed executable")
    config = existing_file(args.config, "Configuration")

    if args.keep_logs:
        root = args.keep_logs.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="autopasdm_strong_scaling_")
        root = Path(temporary.name)

    implementations = {
        "reference": reference,
        "distributed": distributed,
    }

    # Warm up every scaling point. Reverse the rank order every second warm-up and
    # alternate implementation order to reduce systematic ordering effects.
    for warmup in range(args.warmups):
        current_ranks = rank_counts if warmup % 2 == 0 else list(reversed(rank_counts))
        for rank_index, ranks in enumerate(current_ranks):
            order = (
                ("reference", "distributed")
                if (warmup + rank_index) % 2 == 0
                else ("distributed", "reference")
            )
            for implementation in order:
                print(f"Warm-up {warmup + 1}/{args.warmups}, ranks {ranks}: {implementation}")
                run_once(
                    implementation,
                    ranks,
                    -(warmup + 1),
                    implementations[implementation],
                    config,
                    args.launcher,
                    args.omp_threads,
                    root / "warmup" / f"ranks_{ranks}" / f"{warmup:02d}_{implementation}",
                )

    results: list[RunResult] = []
    for repetition in range(args.repetitions):
        current_ranks = rank_counts if repetition % 2 == 0 else list(reversed(rank_counts))
        for rank_index, ranks in enumerate(current_ranks):
            order = (
                ("reference", "distributed")
                if (repetition + rank_index) % 2 == 0
                else ("distributed", "reference")
            )
            for implementation in order:
                print(
                    f"Measured run {repetition + 1}/{args.repetitions}, ranks {ranks}: {implementation}"
                )
                results.append(
                    run_once(
                        implementation,
                        ranks,
                        repetition,
                        implementations[implementation],
                        config,
                        args.launcher,
                        args.omp_threads,
                        root / "measured" / f"ranks_{ranks}" / f"{repetition:02d}_{implementation}",
                    )
                )

    medians = median_results(results)
    print_summary(medians, rank_counts, args.repetitions, args.omp_threads)

    if args.csv:
        write_raw_csv(args.csv, results)
    if args.summary_csv:
        write_summary_csv(args.summary_csv, medians, rank_counts)

    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
