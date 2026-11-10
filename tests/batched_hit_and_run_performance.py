"""Compare scalar and batched hit-and-run performance on rotated boxes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import PolytopeRegion
from art.sampling import (
    BatchedHitAndRunSampler,
    HitAndRunSampler,
    NumbaBatchedHitAndRunSampler,
    ParallelNumbaBatchedHitAndRunSampler,
)
from examples.benchmark_functions import random_orthogonal_matrix


@dataclass(frozen=True)
class PhaseTiming:
    total_seconds: float
    slack_matmul_seconds: float
    constraint_product_matmul_seconds: float
    random_generation_seconds: float
    transitions_seconds: float

    @property
    def other_seconds(self) -> float:
        measured = (
            self.slack_matmul_seconds
            + self.constraint_product_matmul_seconds
            + self.random_generation_seconds
            + self.transitions_seconds
        )
        return max(0.0, self.total_seconds - measured)


@dataclass(frozen=True)
class TimingResult:
    dimension: int
    thinning: int
    n_samples: int
    original_times: tuple[float, ...]
    stepwise_times: tuple[float, ...]
    precomputed_times: tuple[float, ...]
    numba_times: tuple[float, ...]
    parallel_numba_times: tuple[float, ...]
    parallel_rng_numba_times: tuple[float, ...]
    precomputed_phase_timings: tuple[PhaseTiming, ...]
    parallel_rng_numba_phase_timings: tuple[PhaseTiming, ...]

    @property
    def original_median(self) -> float:
        return float(np.median(self.original_times))

    @property
    def stepwise_median(self) -> float:
        return float(np.median(self.stepwise_times))

    @property
    def precomputed_median(self) -> float:
        return float(np.median(self.precomputed_times))

    @property
    def numba_median(self) -> float:
        return float(np.median(self.numba_times))

    @property
    def stepwise_speedup(self) -> float:
        return self.original_median / self.stepwise_median

    @property
    def precomputed_speedup(self) -> float:
        return self.original_median / self.precomputed_median

    @property
    def numba_speedup(self) -> float:
        return self.original_median / self.numba_median

    @property
    def parallel_numba_median(self) -> float:
        return float(np.median(self.parallel_numba_times))

    @property
    def parallel_numba_speedup(self) -> float:
        return self.original_median / self.parallel_numba_median

    @property
    def parallel_rng_numba_median(self) -> float:
        return float(np.median(self.parallel_rng_numba_times))

    @property
    def parallel_rng_numba_speedup(self) -> float:
        return self.original_median / self.parallel_rng_numba_median


def make_rotated_box(
    dimension: int,
    random_state: int,
) -> tuple[PolytopeRegion, np.ndarray]:
    """Return a rotated unit box and the rotation mapping box points into it."""

    rotation = random_orthogonal_matrix(dimension, random_state=random_state)
    axes = rotation.T
    region = PolytopeRegion(
        A=np.vstack([axes, -axes]),
        b=np.ones(2 * dimension),
        tag=f"rotated_box_d{dimension}",
    )
    return region, rotation


def sample_starting_pool(
    pool_size: int,
    dimension: int,
    rotation: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Sample uniformly in the axis-aligned box and rotate the points."""

    rng = np.random.default_rng(random_state)
    return rng.uniform(-1.0, 1.0, size=(pool_size, dimension)) @ rotation.T


def run_once(
    sampler: HitAndRunSampler,
    region: PolytopeRegion,
    rotation: np.ndarray,
    n_samples: int,
    pool_size: int,
    start_seed: int,
    sampling_seed: int,
) -> tuple[float, np.ndarray]:
    """Generate starting points, then time only the sampler call."""

    starting_pool = sample_starting_pool(
        pool_size,
        region.dimension,
        rotation,
        start_seed,
    )
    x0 = starting_pool if isinstance(sampler, BatchedHitAndRunSampler) else starting_pool[0]
    start = perf_counter()
    samples = sampler.sample(
        region,
        n_samples,
        random_state=sampling_seed,
        x0=x0,
    )
    elapsed = perf_counter() - start
    return elapsed, samples


def run_profiled_once(
    sampler: BatchedHitAndRunSampler,
    region: PolytopeRegion,
    rotation: np.ndarray,
    n_samples: int,
    pool_size: int,
    start_seed: int,
    sampling_seed: int,
) -> tuple[PhaseTiming, np.ndarray]:
    """Run a sampler and combine its internal phases with wall-clock time."""

    if not sampler.profile_timing:
        raise ValueError("Profiled runs require profile_timing=True.")
    elapsed, samples = run_once(
        sampler,
        region,
        rotation,
        n_samples,
        pool_size,
        start_seed,
        sampling_seed,
    )
    phases = sampler.last_phase_timings
    return (
        PhaseTiming(
            total_seconds=elapsed,
            slack_matmul_seconds=phases["slack_matmul_seconds"],
            constraint_product_matmul_seconds=phases[
                "constraint_product_matmul_seconds"
            ],
            random_generation_seconds=phases["random_generation_seconds"],
            transitions_seconds=phases["transitions_seconds"],
        ),
        samples,
    )


def validate_samples(
    samples: np.ndarray,
    region: PolytopeRegion,
    n_samples: int,
) -> None:
    expected_shape = (n_samples, region.dimension)
    if samples.shape != expected_shape:
        raise RuntimeError(f"Sampler returned {samples.shape}; expected {expected_shape}.")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("Sampler returned non-finite values.")
    if not np.all(region.contains(samples, tol=1e-9)):
        raise RuntimeError("Sampler returned points outside the rotated box.")


def initialize_numba_samplers(
    seed: int,
    random_threads: int,
) -> dict[str, float]:
    """Trigger serial and parallel JIT compilation before measured runs."""

    region, rotation = make_rotated_box(3, random_state=seed + 17)
    samplers = {
        "serial_numba": NumbaBatchedHitAndRunSampler(
            burn_in=1,
            thinning=2,
            n_chains=3,
        ),
        "parallel_numba": ParallelNumbaBatchedHitAndRunSampler(
            burn_in=1,
            thinning=2,
            n_chains=3,
            parallel_random=False,
        ),
        "parallel_rng_numba": ParallelNumbaBatchedHitAndRunSampler(
            burn_in=1,
            thinning=2,
            n_chains=3,
            parallel_random=True,
            random_threads=random_threads,
        ),
    }
    initialization_times: dict[str, float] = {}
    for index, (name, sampler) in enumerate(samplers.items()):
        elapsed, samples = run_once(
            sampler,
            region,
            rotation,
            n_samples=5,
            pool_size=6,
            start_seed=seed + 18 + index,
            sampling_seed=seed + 20 + index,
        )
        validate_samples(samples, region, n_samples=5)
        initialization_times[name] = elapsed
    return initialization_times


def benchmark_configuration(
    dimension: int,
    thinning: int,
    args: argparse.Namespace,
) -> TimingResult:
    region, rotation = make_rotated_box(
        dimension,
        random_state=args.seed + 1009 * dimension,
    )
    n_samples = args.sample_multiplier * dimension
    pool_size = args.start_pool_multiplier * args.n_chains
    samplers = {
        "original": HitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
        ),
        "stepwise": BatchedHitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
            n_chains=args.n_chains,
            precompute_ad=False,
        ),
        "precomputed": BatchedHitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
            n_chains=args.n_chains,
            precompute_ad=True,
        ),
        "numba": NumbaBatchedHitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
            n_chains=args.n_chains,
        ),
        "parallel_numba": ParallelNumbaBatchedHitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
            n_chains=args.n_chains,
            parallel_random=False,
        ),
        "parallel_rng_numba": ParallelNumbaBatchedHitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
            n_chains=args.n_chains,
            parallel_random=True,
            random_threads=args.random_threads,
        ),
    }

    config_seed = args.seed + 100_003 * dimension + 1_009 * thinning
    for index, sampler in enumerate(samplers.values()):
        _, samples = run_once(
            sampler,
            region,
            rotation,
            n_samples,
            pool_size,
            start_seed=config_seed + 10 + index,
            sampling_seed=config_seed + 20 + index,
        )
        validate_samples(samples, region, n_samples)

    timings: dict[str, list[float]] = {name: [] for name in samplers}
    sampler_names = tuple(samplers)
    for repeat in range(args.timing_repeats):
        offset = repeat % len(sampler_names)
        order = sampler_names[offset:] + sampler_names[:offset]
        start_seed = config_seed + 100 + repeat
        sampling_seed = config_seed + 10_000 + repeat
        for name in order:
            elapsed, samples = run_once(
                samplers[name],
                region,
                rotation,
                n_samples,
                pool_size,
                start_seed=start_seed,
                sampling_seed=sampling_seed,
            )
            validate_samples(samples, region, n_samples)
            timings[name].append(elapsed)

    profiled_samplers = {
        "precomputed": BatchedHitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
            n_chains=args.n_chains,
            precompute_ad=True,
            profile_timing=True,
        ),
        "parallel_rng_numba": ParallelNumbaBatchedHitAndRunSampler(
            burn_in=args.burn_in,
            thinning=thinning,
            n_chains=args.n_chains,
            parallel_random=True,
            random_threads=args.random_threads,
            profile_timing=True,
        ),
    }
    for index, sampler in enumerate(profiled_samplers.values()):
        _, samples = run_profiled_once(
            sampler,
            region,
            rotation,
            n_samples,
            pool_size,
            start_seed=config_seed + 30 + index,
            sampling_seed=config_seed + 40 + index,
        )
        validate_samples(samples, region, n_samples)

    phase_timings: dict[str, list[PhaseTiming]] = {
        name: [] for name in profiled_samplers
    }
    profile_names = tuple(profiled_samplers)
    for repeat in range(args.timing_repeats):
        offset = repeat % len(profile_names)
        order = profile_names[offset:] + profile_names[:offset]
        start_seed = config_seed + 20_000 + repeat
        sampling_seed = config_seed + 30_000 + repeat
        for name in order:
            phases, samples = run_profiled_once(
                profiled_samplers[name],
                region,
                rotation,
                n_samples,
                pool_size,
                start_seed=start_seed,
                sampling_seed=sampling_seed,
            )
            validate_samples(samples, region, n_samples)
            phase_timings[name].append(phases)

    return TimingResult(
        dimension=dimension,
        thinning=thinning,
        n_samples=n_samples,
        original_times=tuple(timings["original"]),
        stepwise_times=tuple(timings["stepwise"]),
        precomputed_times=tuple(timings["precomputed"]),
        numba_times=tuple(timings["numba"]),
        parallel_numba_times=tuple(timings["parallel_numba"]),
        parallel_rng_numba_times=tuple(timings["parallel_rng_numba"]),
        precomputed_phase_timings=tuple(phase_timings["precomputed"]),
        parallel_rng_numba_phase_timings=tuple(
            phase_timings["parallel_rng_numba"]
        ),
    )


def format_grid(
    results: list[TimingResult],
    dimensions: list[int],
    thinnings: list[int],
    attribute: str,
    title: str,
    unit: str,
) -> str:
    values = {(result.dimension, result.thinning): result for result in results}
    column_width = 16
    lines = [title]
    header = "dimension".ljust(12) + "".join(
        f"thin={value}".rjust(column_width) for value in thinnings
    )
    lines.append(header)
    for dimension in dimensions:
        row = [f"d={dimension}".ljust(12)]
        for thinning in thinnings:
            value = float(getattr(values[(dimension, thinning)], attribute))
            row.append(f"{value:.6f}{unit}".rjust(column_width))
        lines.append("".join(row))
    return "\n".join(lines)


def median_phase_summary(timings: tuple[PhaseTiming, ...]) -> dict[str, float]:
    """Return median phase durations and percentages of median wall time."""

    phase_names = (
        "slack_matmul_seconds",
        "constraint_product_matmul_seconds",
        "random_generation_seconds",
        "transitions_seconds",
    )
    summary = {
        "total_seconds": float(
            np.median([timing.total_seconds for timing in timings])
        )
    }
    for name in phase_names:
        summary[name] = float(np.median([getattr(timing, name) for timing in timings]))
    summary["other_seconds"] = max(
        0.0,
        summary["total_seconds"] - sum(summary[name] for name in phase_names),
    )
    for name in (*phase_names, "other_seconds"):
        summary[name.replace("_seconds", "_percent")] = (
            100.0 * summary[name] / summary["total_seconds"]
        )
    return summary


def format_phase_breakdown(
    results: list[TimingResult],
    timing_attribute: str,
    title: str,
) -> str:
    """Format one row per configuration with phase time and total share."""

    lines = [title]
    lines.append(
        "configuration  total_ms  XA^T_ms(%)  DA^T_ms(%)  "
        "rng_ms(%)  transitions_ms(%)  other_ms(%)"
    )
    for result in results:
        summary = median_phase_summary(getattr(result, timing_attribute))

        def phase(name: str) -> str:
            return (
                f"{1e3 * summary[name + '_seconds']:.3f}"
                f"({summary[name + '_percent']:.1f}%)"
            )

        lines.append(
            f"d={result.dimension},thin={result.thinning:<2d}  "
            f"{1e3 * summary['total_seconds']:.3f}  "
            f"{phase('slack_matmul')}  "
            f"{phase('constraint_product_matmul')}  "
            f"{phase('random_generation')}  "
            f"{phase('transitions')}  "
            f"{phase('other')}"
        )
    return "\n".join(lines)


def write_csv(results: list[TimingResult], output_path: Path) -> None:
    fields = (
        "dimension",
        "thinning",
        "n_samples",
        "original_median_seconds",
        "stepwise_median_seconds",
        "precomputed_median_seconds",
        "numba_median_seconds",
        "parallel_numba_median_seconds",
        "parallel_rng_numba_median_seconds",
        "original_over_stepwise_ratio",
        "original_over_precomputed_ratio",
        "original_over_numba_ratio",
        "original_over_parallel_numba_ratio",
        "original_over_parallel_rng_numba_ratio",
        "original_iqr_seconds",
        "stepwise_iqr_seconds",
        "precomputed_iqr_seconds",
        "numba_iqr_seconds",
        "parallel_numba_iqr_seconds",
        "parallel_rng_numba_iqr_seconds",
        "original_samples_per_second",
        "stepwise_samples_per_second",
        "precomputed_samples_per_second",
        "numba_samples_per_second",
        "parallel_numba_samples_per_second",
        "parallel_rng_numba_samples_per_second",
        "original_times_seconds",
        "stepwise_times_seconds",
        "precomputed_times_seconds",
        "numba_times_seconds",
        "parallel_numba_times_seconds",
        "parallel_rng_numba_times_seconds",
        "precomputed_total_seconds",
        "precomputed_slack_matmul_seconds",
        "precomputed_slack_matmul_percent",
        "precomputed_constraint_product_matmul_seconds",
        "precomputed_constraint_product_matmul_percent",
        "precomputed_random_generation_seconds",
        "precomputed_random_generation_percent",
        "precomputed_transitions_seconds",
        "precomputed_transitions_percent",
        "precomputed_other_seconds",
        "precomputed_other_percent",
        "parallel_rng_numba_total_seconds",
        "parallel_rng_numba_slack_matmul_seconds",
        "parallel_rng_numba_slack_matmul_percent",
        "parallel_rng_numba_constraint_product_matmul_seconds",
        "parallel_rng_numba_constraint_product_matmul_percent",
        "parallel_rng_numba_random_generation_seconds",
        "parallel_rng_numba_random_generation_percent",
        "parallel_rng_numba_transitions_seconds",
        "parallel_rng_numba_transitions_percent",
        "parallel_rng_numba_other_seconds",
        "parallel_rng_numba_other_percent",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            precomputed_phases = median_phase_summary(
                result.precomputed_phase_timings
            )
            parallel_phases = median_phase_summary(
                result.parallel_rng_numba_phase_timings
            )
            original_iqr = interquartile_range(result.original_times)
            stepwise_iqr = interquartile_range(result.stepwise_times)
            precomputed_iqr = interquartile_range(result.precomputed_times)
            numba_iqr = interquartile_range(result.numba_times)
            parallel_numba_iqr = interquartile_range(result.parallel_numba_times)
            parallel_rng_numba_iqr = interquartile_range(
                result.parallel_rng_numba_times
            )
            writer.writerow(
                {
                    "dimension": result.dimension,
                    "thinning": result.thinning,
                    "n_samples": result.n_samples,
                    "original_median_seconds": result.original_median,
                    "stepwise_median_seconds": result.stepwise_median,
                    "precomputed_median_seconds": result.precomputed_median,
                    "numba_median_seconds": result.numba_median,
                    "parallel_numba_median_seconds": result.parallel_numba_median,
                    "parallel_rng_numba_median_seconds": (
                        result.parallel_rng_numba_median
                    ),
                    "original_over_stepwise_ratio": result.stepwise_speedup,
                    "original_over_precomputed_ratio": result.precomputed_speedup,
                    "original_over_numba_ratio": result.numba_speedup,
                    "original_over_parallel_numba_ratio": (
                        result.parallel_numba_speedup
                    ),
                    "original_over_parallel_rng_numba_ratio": (
                        result.parallel_rng_numba_speedup
                    ),
                    "original_iqr_seconds": original_iqr,
                    "stepwise_iqr_seconds": stepwise_iqr,
                    "precomputed_iqr_seconds": precomputed_iqr,
                    "numba_iqr_seconds": numba_iqr,
                    "parallel_numba_iqr_seconds": parallel_numba_iqr,
                    "parallel_rng_numba_iqr_seconds": parallel_rng_numba_iqr,
                    "original_samples_per_second": (
                        result.n_samples / result.original_median
                    ),
                    "stepwise_samples_per_second": (
                        result.n_samples / result.stepwise_median
                    ),
                    "precomputed_samples_per_second": (
                        result.n_samples / result.precomputed_median
                    ),
                    "numba_samples_per_second": result.n_samples / result.numba_median,
                    "parallel_numba_samples_per_second": (
                        result.n_samples / result.parallel_numba_median
                    ),
                    "parallel_rng_numba_samples_per_second": (
                        result.n_samples / result.parallel_rng_numba_median
                    ),
                    "original_times_seconds": ";".join(
                        f"{value:.12e}" for value in result.original_times
                    ),
                    "stepwise_times_seconds": ";".join(
                        f"{value:.12e}" for value in result.stepwise_times
                    ),
                    "precomputed_times_seconds": ";".join(
                        f"{value:.12e}" for value in result.precomputed_times
                    ),
                    "numba_times_seconds": ";".join(
                        f"{value:.12e}" for value in result.numba_times
                    ),
                    "parallel_numba_times_seconds": ";".join(
                        f"{value:.12e}" for value in result.parallel_numba_times
                    ),
                    "parallel_rng_numba_times_seconds": ";".join(
                        f"{value:.12e}"
                        for value in result.parallel_rng_numba_times
                    ),
                    **{
                        f"precomputed_{key}": value
                        for key, value in precomputed_phases.items()
                    },
                    **{
                        f"parallel_rng_numba_{key}": value
                        for key, value in parallel_phases.items()
                    },
                }
            )


def interquartile_range(values: tuple[float, ...]) -> float:
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[5, 10, 20, 40])
    parser.add_argument("--thinnings", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--sample-multiplier", type=int, default=50)
    parser.add_argument("--n-chains", type=int, default=10)
    parser.add_argument("--start-pool-multiplier", type=int, default=10)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--numba-threads", type=int, default=None)
    parser.add_argument("--random-threads", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "batched_hit_and_run_performance_outputs",
    )
    args = parser.parse_args()
    if any(value < 1 for value in args.dimensions):
        parser.error("--dimensions values must be positive.")
    if any(value < 1 for value in args.thinnings):
        parser.error("--thinnings values must be positive.")
    if args.sample_multiplier < 1:
        parser.error("--sample-multiplier must be positive.")
    if args.n_chains < 1:
        parser.error("--n-chains must be positive.")
    if args.start_pool_multiplier < 1:
        parser.error("--start-pool-multiplier must be positive.")
    if args.burn_in < 0:
        parser.error("--burn-in must be nonnegative.")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive.")
    if args.numba_threads is not None and args.numba_threads < 1:
        parser.error("--numba-threads must be positive.")
    if args.random_threads < 1:
        parser.error("--random-threads must be positive.")
    return args


def main() -> None:
    args = parse_args()
    from numba import get_num_threads, set_num_threads

    if args.numba_threads is not None:
        set_num_threads(args.numba_threads)
    numba_threads = get_num_threads()
    dimensions = list(dict.fromkeys(args.dimensions))
    thinnings = list(dict.fromkeys(args.thinnings))
    numba_initialization_seconds = initialize_numba_samplers(
        args.seed,
        args.random_threads,
    )
    results = [
        benchmark_configuration(dimension, thinning, args)
        for dimension in dimensions
        for thinning in thinnings
    ]

    grids = (
        format_grid(
            results,
            dimensions,
            thinnings,
            "original_median",
            "Original hit-and-run median time",
            "s",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "stepwise_median",
            "Batched hit-and-run median time (stepwise AD)",
            "s",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "precomputed_median",
            "Batched hit-and-run median time (one-shot AD)",
            "s",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "numba_median",
            "Serial Numba hit-and-run median time (one-shot AD)",
            "s",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "parallel_numba_median",
            "Parallel Numba median time (serial NumPy RNG)",
            "s",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "parallel_rng_numba_median",
            "Parallel Numba median time (threaded NumPy RNG)",
            "s",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "stepwise_speedup",
            "Ratio (original / batched stepwise)",
            "x",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "precomputed_speedup",
            "Ratio (original / batched one-shot)",
            "x",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "numba_speedup",
            "Ratio (original / serial Numba)",
            "x",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "parallel_numba_speedup",
            "Ratio (original / parallel Numba, serial RNG)",
            "x",
        ),
        format_grid(
            results,
            dimensions,
            thinnings,
            "parallel_rng_numba_speedup",
            "Ratio (original / parallel Numba, threaded RNG)",
            "x",
        ),
        format_phase_breakdown(
            results,
            "precomputed_phase_timings",
            "Batched one-shot AD phase breakdown",
        ),
        format_phase_breakdown(
            results,
            "parallel_rng_numba_phase_timings",
            "Parallel Numba with threaded RNG phase breakdown",
        ),
    )
    configuration = "\n".join(
        [
            "[configuration]",
            f"dimensions: {dimensions}",
            f"thinnings: {thinnings}",
            f"sample_multiplier: {args.sample_multiplier}",
            f"n_chains: {args.n_chains}",
            f"start_pool_size: {args.start_pool_multiplier * args.n_chains}",
            f"burn_in: {args.burn_in}",
            f"timing_repeats: {args.timing_repeats}",
            f"numba_threads: {numba_threads}",
            f"random_threads: {args.random_threads}",
            f"seed: {args.seed}",
            "isotropic_sampling: false",
            "starting_pool_generation_timed: false",
            "warmup_runs_per_configuration: 1",
            "phase_profile_runs_per_configuration: "
            f"{args.timing_repeats}",
            "phase_profile_total_includes_starting_pool_generation: false",
            "serial_numba_initialization_seconds: "
            f"{numba_initialization_seconds['serial_numba']:.12e}",
            "parallel_numba_initialization_seconds: "
            f"{numba_initialization_seconds['parallel_numba']:.12e}",
            "parallel_rng_numba_initialization_seconds: "
            f"{numba_initialization_seconds['parallel_rng_numba']:.12e}",
            "numba_initialization_excluded_from_timing: true",
        ]
    )
    report = configuration + "\n\n" + "\n\n".join(grids) + "\n"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    write_csv(results, args.output_dir / "timings.csv")
    print(report, end="")
    print(f"saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
