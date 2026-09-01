"""Diagnostics for hit-and-run autocorrelation decay in a box."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from art.domain import BoxDomain, PolytopeRegion
from art.sampling import (
    HitAndRunSampler,
    autocorrelation_by_lag,
    floor_covariance_eigenvalues,
    make_thinning_candidates,
    sample_covariance_eigendecomposition,
)


def random_unit_vectors(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=(n, d))
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 1e-14
    if not np.any(keep):
        return np.empty((0, d), dtype=float)
    return vectors[keep] / norms[keep, None]


def make_probe_matrix(d: int, num_random_probes: int, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    coordinate_probes = np.eye(d)
    random_probes = random_unit_vectors(num_random_probes, d, rng)
    probes = np.vstack([coordinate_probes, random_probes]) if random_probes.size else coordinate_probes
    labels = [f"x{i + 1}" for i in range(d)]
    labels.extend([f"r{i + 1}" for i in range(random_probes.shape[0])])
    return probes, labels


def append_constraints(region: PolytopeRegion, A_new: np.ndarray, b_new: np.ndarray) -> PolytopeRegion:
    return PolytopeRegion(
        A=np.vstack([region.A, np.asarray(A_new, dtype=float)]),
        b=np.concatenate([region.b, np.asarray(b_new, dtype=float).reshape(-1)]),
        depth=region.depth,
        tag=region.tag,
    )


def random_orthonormal_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def make_test_polytope(
    kind: str,
    d: int,
    low: float,
    high: float,
    rng: np.random.Generator,
    slab_width: float,
    random_cuts: int,
    cut_bound_low: float,
    cut_bound_high: float,
    rotated_width_ratio: float,
) -> tuple[PolytopeRegion, np.ndarray, str]:
    if kind == "box":
        domain = BoxDomain.hypercube(d, low, high)
        return domain.as_region(), np.mean(domain.bounds, axis=1), f"[{low}, {high}]^d"

    if kind in ("halfspace_box", "slab_box", "random_oblique_box"):
        domain = BoxDomain.hypercube(d, low, high)
        region = domain.as_region()
        x0 = np.mean(domain.bounds, axis=1)

        if kind == "halfspace_box":
            normal = np.ones(d) / np.sqrt(d)
            bound = 0.25 * np.sqrt(d)
            region = append_constraints(region, normal.reshape(1, -1), np.array([bound]))
            return region, x0, f"[{low}, {high}]^d with sum(x)/sqrt(d) <= {bound:.3g}"

        if kind == "slab_box":
            normal = random_unit_vectors(1, d, rng)[0]
            A_new = np.vstack([normal, -normal])
            b_new = np.array([slab_width, slab_width])
            region = append_constraints(region, A_new, b_new)
            return region, x0, f"[{low}, {high}]^d with |v.x| <= {slab_width:.3g}"

        normals = random_unit_vectors(random_cuts, d, rng)
        bounds_new = rng.uniform(cut_bound_low, cut_bound_high, size=normals.shape[0])
        region = append_constraints(region, normals, bounds_new)
        return region, x0, f"[{low}, {high}]^d with {normals.shape[0]} random oblique cuts"

    if kind == "simplex":
        A = np.vstack([-np.eye(d), np.ones((1, d))])
        b = np.concatenate([np.zeros(d), np.array([1.0])])
        x0 = np.full(d, 0.5 / d)
        return PolytopeRegion(A=A, b=b, tag="simplex"), x0, "{x >= 0, sum(x) <= 1}"

    if kind == "rotated_box":
        if not (0.0 < rotated_width_ratio <= 1.0):
            raise ValueError("--rotated-width-ratio must satisfy 0 < ratio <= 1.")
        q = random_orthonormal_matrix(d, rng)
        widths = np.geomspace(rotated_width_ratio, 1.0, d)
        A = np.vstack([q.T, -q.T])
        b = np.concatenate([widths, widths])
        return PolytopeRegion(A=A, b=b, tag="rotated_box"), np.zeros(d), (
            f"rotated box with widths in [{rotated_width_ratio:.3g}, 1]"
        )

    raise ValueError(f"Unknown polytope kind: {kind}.")


def save_chain_scatter(chain: np.ndarray, out_path: Path) -> None:
    if chain.shape[1] < 2:
        return
    plt.figure(figsize=(6, 6))
    plt.plot(chain[:, 0], chain[:, 1], linewidth=0.6, alpha=0.35)
    plt.scatter(chain[:, 0], chain[:, 1], s=8, alpha=0.6)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Hit-and-run diagnostic chain")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")


def save_autocorr_plot(
    lags: np.ndarray,
    autocorr: np.ndarray,
    labels: list[str],
    threshold: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.plot(lags, autocorr[:, i], linewidth=1.5, label=label)
    plt.axhline(threshold, color="black", linestyle="--", linewidth=1)
    plt.axhline(-threshold, color="black", linestyle="--", linewidth=1)
    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title("Signed autocorrelation by probe")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")


def save_max_abs_plot(
    candidates: np.ndarray,
    max_abs_autocorr: np.ndarray,
    selected_thinning: int,
    threshold: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(candidates, max_abs_autocorr, marker="o", linewidth=2)
    plt.axhline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
    plt.axvline(selected_thinning, color="tab:red", linestyle=":", linewidth=2, label="selected thinning")
    plt.xlabel("candidate thinning")
    plt.ylabel("max |autocorrelation|")
    plt.title("Thinning selection diagnostic")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"saved plot: {out_path}")


def save_summary(summary: dict[str, object], out_path: Path) -> None:
    lines = [f"{key}: {value}" for key, value in summary.items()]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved summary: {out_path}")


def select_thinning(
    candidates: np.ndarray,
    max_abs_autocorr: np.ndarray,
    threshold: float,
    stable_window: int,
) -> int:
    window_size = min(int(stable_window), candidates.size)
    for idx in range(candidates.size - window_size + 1):
        if float(np.max(max_abs_autocorr[idx : idx + window_size])) <= threshold:
            return int(candidates[idx])
    return int(candidates[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument(
        "--polytope",
        choices=["box", "halfspace_box", "slab_box", "random_oblique_box", "simplex", "rotated_box"],
        default="box",
    )
    parser.add_argument("--low", type=float, default=-1.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=3000)
    parser.add_argument("--max-lag", type=int, default=200)
    parser.add_argument("--candidate-mode", choices=["linear", "powers_of_two"], default="linear")
    parser.add_argument("--acf-threshold", type=float, default=0.1)
    parser.add_argument("--stable-window", type=int, default=3)
    parser.add_argument("--num-random-probes", type=int, default=6)
    parser.add_argument(
        "--covariance-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Estimate a covariance spectrum from an ordinary pilot chain, then "
            "generate the diagnostic chain with covariance-shaped directions."
        ),
    )
    parser.add_argument(
        "--direction-eigenvalue-floor",
        type=float,
        default=1e-2,
        help="Trace-scaled floor ratio applied to covariance eigenvalues.",
    )
    parser.add_argument(
        "--covariance-pilot-multiplier",
        type=int,
        default=50,
        help="Number of covariance pilot samples per input dimension.",
    )
    parser.add_argument("--slab-width", type=float, default=0.25)
    parser.add_argument("--random-cuts", type=int, default=None)
    parser.add_argument("--cut-bound-low", type=float, default=0.2)
    parser.add_argument("--cut-bound-high", type=float, default=1.0)
    parser.add_argument("--rotated-width-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("hit_and_run_autocorr_outputs"))
    args = parser.parse_args()

    if args.dim < 1:
        raise ValueError("--dim must be at least 1.")
    if args.low >= args.high:
        raise ValueError("--low must be less than --high.")
    if args.n_steps < 3:
        raise ValueError("--n-steps must be at least 3.")
    if args.max_lag < 1:
        raise ValueError("--max-lag must be at least 1.")
    if args.num_random_probes < 0:
        raise ValueError("--num-random-probes must be nonnegative.")
    if not np.isfinite(args.direction_eigenvalue_floor) or args.direction_eigenvalue_floor <= 0.0:
        raise ValueError("--direction-eigenvalue-floor must be positive.")
    if args.covariance_pilot_multiplier < 1:
        raise ValueError("--covariance-pilot-multiplier must be at least 1.")
    if args.slab_width <= 0.0:
        raise ValueError("--slab-width must be positive.")
    if args.random_cuts is not None and args.random_cuts < 0:
        raise ValueError("--random-cuts must be nonnegative.")
    if args.cut_bound_low <= 0.0 or args.cut_bound_high <= 0.0:
        raise ValueError("--cut-bound-low and --cut-bound-high must be positive.")
    if args.cut_bound_low > args.cut_bound_high:
        raise ValueError("--cut-bound-low must be at most --cut-bound-high.")

    rng = np.random.default_rng(args.seed)
    probe_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    random_cuts = args.dim if args.random_cuts is None else args.random_cuts
    region, x0, description = make_test_polytope(
        kind=args.polytope,
        d=args.dim,
        low=args.low,
        high=args.high,
        rng=rng,
        slab_width=args.slab_width,
        random_cuts=random_cuts,
        cut_bound_low=args.cut_bound_low,
        cut_bound_high=args.cut_bound_high,
        rotated_width_ratio=args.rotated_width_ratio,
    )
    sampler = HitAndRunSampler(
        burn_in=0,
        thinning=1,
        direction_eigenvalue_floor=args.direction_eigenvalue_floor,
    )
    pilot_chain = None
    covariance_eigenvalues = None
    floored_eigenvalues = None
    eigenvalue_floor = None
    covariance_condition_number = None
    covariance_floor_saturated = False

    if args.covariance_sampling:
        covariance_pilot_steps = args.covariance_pilot_multiplier * args.dim
        pilot_chain = sampler.sample(
            region,
            n=covariance_pilot_steps,
            random_state=rng,
            x0=x0,
        )
        eigenvectors, covariance_eigenvalues = sample_covariance_eigendecomposition(pilot_chain)
        floored_eigenvalues, eigenvalue_floor = floor_covariance_eigenvalues(
            covariance_eigenvalues,
            args.direction_eigenvalue_floor,
        )
        covariance_condition_number = float(
            np.max(floored_eigenvalues) / np.min(floored_eigenvalues)
        )
        covariance_floor_saturated = bool(
            np.any(covariance_eigenvalues <= eigenvalue_floor)
        )
        sampler = HitAndRunSampler(
            burn_in=0,
            thinning=1,
            direction_eigenvectors=eigenvectors,
            direction_eigenvalues=covariance_eigenvalues,
            direction_eigenvalue_floor=args.direction_eigenvalue_floor,
        )
        x0 = pilot_chain[-1]

    chain = sampler.sample(
        region,
        n=args.n_steps,
        random_state=rng,
        x0=x0,
    )

    probes, labels = make_probe_matrix(args.dim, args.num_random_probes, probe_rng)
    lags = make_thinning_candidates(min(args.max_lag, chain.shape[0] // 2), mode=args.candidate_mode)
    probe_series = chain @ probes.T
    autocorr = autocorrelation_by_lag(probe_series, lags)
    candidates = lags
    max_abs_autocorr = np.max(np.abs(autocorr), axis=1)
    selected_thinning = select_thinning(
        candidates,
        max_abs_autocorr,
        threshold=args.acf_threshold,
        stable_window=args.stable_window,
    )

    save_chain_scatter(chain, output_dir / "chain_scatter.png")
    save_autocorr_plot(lags, autocorr, labels, args.acf_threshold, output_dir / "autocorr_by_probe.png")
    save_max_abs_plot(
        candidates,
        max_abs_autocorr,
        selected_thinning,
        args.acf_threshold,
        output_dir / "max_abs_autocorr.png",
    )
    save_summary(
        {
            "polytope": args.polytope,
            "description": description,
            "dimension": args.dim,
            "n_constraints": region.A.shape[0],
            "n_steps": args.n_steps,
            "covariance_sampling": args.covariance_sampling,
            "covariance_pilot_multiplier": args.covariance_pilot_multiplier,
            "covariance_pilot_steps": (
                args.covariance_pilot_multiplier * args.dim
                if args.covariance_sampling
                else 0
            ),
            "direction_eigenvalue_floor_ratio": args.direction_eigenvalue_floor,
            "covariance_eigenvalue_floor": eigenvalue_floor,
            "covariance_condition_number": covariance_condition_number,
            "covariance_floor_saturated": covariance_floor_saturated,
            "covariance_eigenvalues": (
                covariance_eigenvalues.tolist() if covariance_eigenvalues is not None else None
            ),
            "floored_covariance_eigenvalues": (
                floored_eigenvalues.tolist() if floored_eigenvalues is not None else None
            ),
            "max_lag": args.max_lag,
            "candidate_mode": args.candidate_mode,
            "acf_threshold": args.acf_threshold,
            "stable_window": args.stable_window,
            "num_random_probes": args.num_random_probes,
            "slab_width": args.slab_width,
            "random_cuts": random_cuts,
            "cut_bound_low": args.cut_bound_low,
            "cut_bound_high": args.cut_bound_high,
            "rotated_width_ratio": args.rotated_width_ratio,
            "selected_thinning": selected_thinning,
            "candidate_thinnings": candidates.tolist(),
            "max_abs_autocorrelation": max_abs_autocorr.tolist(),
        },
        output_dir / "summary.txt",
    )


if __name__ == "__main__":
    main()
