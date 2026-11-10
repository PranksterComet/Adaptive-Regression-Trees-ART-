"""Run and plot a fully wet 1D dam break over a Gaussian bed bump.

Uses the PyClaw bathymetry solver demonstrated in:
https://www.clawpack.org/gallery/pyclaw/gallery/sill.html
Run with the Conda environment containing Clawpack activated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta-left", type=float, default=2.0)
    parser.add_argument("--eta-right", type=float, default=1.0)
    parser.add_argument("--bump-height", type=float, default=0.3)
    parser.add_argument("--cells", type=int, default=500)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sensor", type=float, default=3.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "dam_break",
    )
    args = parser.parse_args()
    parameters = [args.eta_left, args.eta_right, args.bump_height,
                  args.final_time, args.sensor]
    if not np.all(np.isfinite(parameters)):
        parser.error("Physical parameters must be finite.")
    if not args.eta_left > args.eta_right > args.bump_height >= 0:
        parser.error("Require eta-left > eta-right > bump-height >= 0 (fully wet).")
    if args.cells < 50 or args.cells % 2:
        parser.error("Use an even number of cells >= 50 so the dam is a cell edge.")
    if args.final_time <= 0:
        parser.error("final-time must be positive.")
    dx = 20.0 / args.cells
    if not -10.0 + dx / 2 <= args.sensor <= 10.0 - dx / 2:
        parser.error("sensor must lie between the first and last cell centers.")

    # Import here so --help does not require Clawpack.
    from clawpack import pyclaw, riemann

    setup_start = perf_counter()
    domain = pyclaw.Domain(pyclaw.Dimension(-10.0, 10.0, args.cells, name="x"))
    state = pyclaw.State(domain, 2, 1)
    x = state.grid.x.centers.copy()
    bed = args.bump_height * np.exp(-((x - 2.0) / 0.75) ** 2)
    initial_surface = np.where(x < 0.0, args.eta_left, args.eta_right)
    state.aux[0, :] = bed
    state.q[0, :] = initial_surface - bed
    state.q[1, :] = 0.0  # Momentum h*u; initially stationary water.
    state.problem_data.update(grav=9.81, dry_tolerance=1e-6, sea_level=0.0)

    solver = pyclaw.ClawSolver1D(riemann.shallow_bathymetry_fwave_1D)
    solver.fwave = True
    solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.bc_lower[0] = solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = solver.aux_bc_upper[0] = pyclaw.BC.extrap
    solver.verbosity = 0

    simulation = pyclaw.Controller()
    simulation.solution = pyclaw.Solution(state, domain)
    simulation.solver = solver
    simulation.tfinal = args.final_time
    simulation.num_output_times = 4
    simulation.keep_copy = True
    simulation.output_format = None
    setup_seconds = perf_counter() - setup_start
    solve_start = perf_counter()
    simulation.run()
    solve_seconds = perf_counter() - solve_start

    times = np.array([frame.t for frame in simulation.frames])
    q = np.stack([frame.state.q.copy() for frame in simulation.frames])
    if not np.isclose(times[-1], args.final_time):
        raise RuntimeError("Solver did not reach the requested final time.")
    if not np.all(np.isfinite(q)) or np.any(q[:, 0, :] <= 0):
        raise RuntimeError("Simulation produced nonfinite values or nonpositive depth.")
    depth = q[:, 0, :]
    velocity = q[:, 1, :] / depth
    surface = depth + bed[None, :]
    # This is a discrete sensor convention, not an exact continuum point value.
    sensor_depth = float(np.interp(args.sensor, x, depth[-1]))
    mass = depth.sum(axis=1) * dx

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "solution.npz", x=x, time=times, bed=bed,
        depth=depth, momentum=q[:, 1, :], velocity=velocity,
        surface=surface,
    )
    summary = {
        "eta_left_m": args.eta_left, "eta_right_m": args.eta_right,
        "bump_height_m": args.bump_height, "bump_center_m": 2.0,
        "bump_width_m": 0.75, "dam_position_m": 0.0,
        "domain_m": [-10.0, 10.0], "cells": args.cells,
        "final_time_s": args.final_time, "sensor_position_m": args.sensor,
        "sensor_depth_m": sensor_depth,
        "minimum_saved_depth_m": float(depth.min()),
        "relative_mass_change": float((mass[-1] - mass[0]) / mass[0]),
        "setup_seconds_excluding_imports": setup_seconds,
        "solve_seconds_including_snapshot_copies": solve_seconds,
    }
    from importlib.metadata import version
    summary["clawpack_version"] = version("clawpack")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             layout="constrained")
    axes[0].fill_between(x, 0.0, bed, color="0.7", label="Riverbed")
    for i, time in enumerate(times):
        label = f"t = {time:.2f} s"
        axes[0].plot(x, surface[i], label=label)
        axes[1].plot(x, velocity[i], label=label)
    for ax in axes:
        ax.axvline(args.sensor, color="0.4", linestyle=":")
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Surface elevation h + b (m)",
                title="Dam break at x = 0 over an underwater bump")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].set(xlabel="Position x (m)", ylabel="Velocity u (m/s)")
    fig.savefig(args.output_dir / "dam_break.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved plot and data to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
