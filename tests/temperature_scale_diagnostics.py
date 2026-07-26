"""Diagnostics for automatic soft-split temperature scales."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXAMPLES_DIR = PROJECT_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from art.domain import BoxDomain
from art.sampling import sample_uniform_box
from art.temperature import (
    estimate_temperature,
    median_nearest_neighbor_distance,
    median_pairwise_distance,
    subsample_points,
)
from test_helpers import parse_csv_floats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--low", type=float, default=-1.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-values", type=str, default="0.5,1.0,2.0,4.0")
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--use-all-points", action="store_true")
    parser.add_argument("--nn-method", choices=["kdtree", "bruteforce"], default="kdtree")
    args = parser.parse_args()

    if args.dim < 1:
        raise ValueError("--dim must be at least 1.")
    if args.low >= args.high:
        raise ValueError("--low must be less than --high.")

    d = args.dim
    n_samples = 100 * (d + 1)
    bounds = BoxDomain.hypercube(d, args.low, args.high).bounds
    X = sample_uniform_box(bounds, n_samples, random_state=args.seed)

    max_points = None if args.use_all_points else args.max_points
    X_used = subsample_points(X, max_points=max_points, random_state=args.seed)
    c_values = parse_csv_floats(args.c_values)

    box_diameter = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
    median_nn = median_nearest_neighbor_distance(X_used, method=args.nn_method)
    median_pairwise = median_pairwise_distance(X_used)
    median_pairwise_scaled = median_pairwise * (X_used.shape[0] ** (-1.0 / d))

    print(f"dimension: {d}")
    print(f"n_samples: {n_samples}")
    print(f"n_used_for_temperature: {X_used.shape[0]}")
    print(f"bounds: [{args.low}, {args.high}]^d")
    print(f"box_diameter: {box_diameter:.8e}")
    print(f"median_nearest_neighbor_distance: {median_nn:.8e}")
    print(f"median_pairwise_distance: {median_pairwise:.8e}")
    print(f"median_pairwise_distance * n_used^(-1/d): {median_pairwise_scaled:.8e}")
    print(f"median_nn / box_diameter: {median_nn / box_diameter:.8e}")
    print(f"median_pairwise_scaled / box_diameter: {median_pairwise_scaled / box_diameter:.8e}")

    print("\nCandidate temperatures")
    for mode in ("median_nn", "median_pairwise_scaled"):
        print(f"\nmode: {mode}")
        for c in c_values:
            temperature = estimate_temperature(
                X,
                mode=mode,
                c=c,
                max_points=max_points,
                random_state=args.seed,
                nn_method=args.nn_method,
            )
            print(f"  c={c:g}: {temperature:.8e}")


if __name__ == "__main__":
    main()
