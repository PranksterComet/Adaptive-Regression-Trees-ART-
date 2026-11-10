"""Resolve recorded and generated oracles for Tasmanian comparisons.

This module does not import Tasmanian or eagerly initialize PyClaw. Recorded
presets are checked-in snapshots, while generated benchmarks store all random
parameters in their run metadata.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from examples.benchmark_functions import (
    ActiveSubspaceGaussianFunction, PlaneWaveFunction, RastriginFunction,
    RosenbrockFunction,
    SphericalPiecewisePolynomialFunction, fixed_spherical_affine_pieces,
)

PRESETS = json.loads(Path(__file__).with_name("oracle_presets.json").read_text())
DEFAULT_DIMENSIONS = {
    "active_subspace_gaussian": 5,
    "rosenbrock": 5, "rastrigin": 3, "plane_wave": 3,
    "spherical_piecewise": 3, "shallow_water_sensor": 3,
    "shallow_water_space_time": 5,
}


class ConfiguredOracle:
    """Callable output shift that preserves access to the underlying target."""

    def __init__(self, target, offset: float):
        self.target = target
        self.offset = float(offset)

    def __call__(self, points):
        return self.target(points) + self.offset


def add_oracle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", choices=tuple(DEFAULT_DIMENSIONS), default="rosenbrock")
    parser.add_argument("--dimension", type=int,
                        help="Default: 5 for active Gaussian/Rosenbrock/space-time, 3 otherwise.")
    parser.add_argument("--output-offset", "--offset", dest="output_offset", type=float,
                        help="Override the recorded benchmark offset.")
    parser.add_argument("--domain-low", type=float, help="Synthetic domain lower bound in every dimension.")
    parser.add_argument("--domain-high", type=float, help="Synthetic domain upper bound in every dimension.")
    parser.add_argument("--seed", type=int, help="Active-subspace rotation seed; default: 1.")
    parser.add_argument("--active-dimension", "--k", dest="active_dimension", type=int,
                        help="Active Gaussian intrinsic dimension; default: 3.")
    parser.add_argument("--eta", type=float, help="Active Gaussian inactive precision budget; default: 0.1.")
    parser.add_argument("--a", type=float, help="Active Gaussian domain half-width; default: 1.")
    parser.add_argument("--tau", type=float, nargs="+",
                        help="Active Gaussian scales; default: 0.2, 0.5, 0.8, ...")
    parser.add_argument("--active-mean", type=float, nargs="+",
                        help="Active-coordinate Gaussian mean; default: zero.")
    parser.add_argument("--rosenbrock-a", type=float)
    parser.add_argument("--rosenbrock-b", type=float)
    parser.add_argument("--rastrigin-a", type=float)
    parser.add_argument("--frequency", "--plane-wave-frequency", dest="frequency", type=float)
    parser.add_argument("--amplitude", "--plane-wave-amplitude", dest="amplitude", type=float)
    parser.add_argument("--phase", "--plane-wave-phase", dest="phase", type=float)
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument("--normal", choices=("axis", "axis_x1", "diagonal", "dense_signs"),
                             help="Default: exact dense_signs direction saved for the dimension in results/.")
    orientation.add_argument("--plane-wave-normal", type=float, nargs="+", help="Explicit vector, normalized by the oracle.")
    parser.add_argument("--sphere-radius", type=float, help="Default: exact saved radius for dimension 2–5.")
    parser.add_argument("--eta-left-range", type=float, nargs=2)
    parser.add_argument("--eta-right-range", type=float, nargs=2)
    parser.add_argument("--bump-height-range", type=float, nargs=2)
    parser.add_argument("--sensor-position", type=float)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--position-range", type=float, nargs=2)
    parser.add_argument("--time-range", type=float, nargs=2)
    parser.add_argument("--cells", type=int)


def make_oracle(args: argparse.Namespace):
    """Return (callable, input bounds, resolved metadata), respecting overrides."""
    name, d = args.benchmark, args.dimension
    if d is None:
        d = DEFAULT_DIMENSIONS[name]
    if d < 1:
        raise ValueError("--dimension must be positive")
    water = name.startswith("shallow_water_")
    expected = DEFAULT_DIMENSIONS[name]
    if water and d != expected:
        raise ValueError(f"{name} requires --dimension {expected}")
    if (args.domain_low is None) != (args.domain_high is None):
        raise ValueError("Supply both --domain-low and --domain-high")
    if water and args.domain_low is not None:
        raise ValueError("Shallow-water inputs remain normalized [-1,1]; use physical range flags instead")

    # Reject parameters which would otherwise silently do nothing.
    groups = {
        "active_subspace_gaussian": (
            "seed", "active_dimension", "eta", "a", "tau", "active_mean",
        ),
        "rosenbrock": ("rosenbrock_a", "rosenbrock_b"),
        "rastrigin": ("rastrigin_a",),
        "plane_wave": ("frequency", "amplitude", "phase", "normal", "plane_wave_normal"),
        "spherical_piecewise": ("sphere_radius",),
        "water": ("eta_left_range", "eta_right_range", "bump_height_range", "cells"),
        "shallow_water_sensor": ("sensor_position", "final_time"),
        "shallow_water_space_time": ("position_range", "time_range"),
    }
    for group, fields in groups.items():
        if group == name or (group == "water" and water):
            continue
        for field in fields:
            if getattr(args, field) is not None:
                raise ValueError(f"--{field.replace('_', '-')} does not apply to {name}")

    if name == "active_subspace_gaussian":
        if args.domain_low is not None:
            raise ValueError("Use --a to set the active Gaussian domain [-a, a]")
        target = ActiveSubspaceGaussianFunction(
            dimension=d,
            active_dimension=3 if args.active_dimension is None else args.active_dimension,
            eta=0.1 if args.eta is None else args.eta,
            a=1.0 if args.a is None else args.a,
            tau=None if args.tau is None else np.asarray(args.tau, dtype=float),
            active_mean=(
                None
                if args.active_mean is None
                else np.asarray(args.active_mean, dtype=float)
            ),
            random_state=1 if args.seed is None else args.seed,
        )
        offset = 1.0 if args.output_offset is None else args.output_offset
        if not np.isfinite(offset):
            raise ValueError("--output-offset must be finite")
        bounds = np.tile([-target.a, target.a], (d, 1))
        params = {
            "active_dimension": target.active_dimension,
            "inactive_dimension": target.dimension - target.active_dimension,
            "a": target.a,
            "eta": target.eta,
            "tau": target.tau.tolist(),
            "active_mean": target.active_mean.tolist(),
            "active_basis_U": target.U.tolist(),
            "inactive_variance_scaled_coordinates": target.inactive_variance,
            "rotation_distribution": "Haar",
            "rotation_seed": 1 if args.seed is None else args.seed,
            "offset": float(offset),
        }
        explicit_fields = (
            "dimension", "output_offset", "seed", "active_dimension", "eta",
            "a", "tau", "active_mean",
        )
        metadata = {
            "benchmark": name,
            "dimension": d,
            "input_bounds": bounds.tolist(),
            "parameters": params,
            "preset_source_report": "generated active-subspace Gaussian defaults",
            "notes": [],
            "explicit_overrides": {
                field: getattr(args, field)
                for field in explicit_fields
                if getattr(args, field) is not None
            },
        }
        return ConfiguredOracle(target, offset), bounds, metadata

    key = "shallow_water_space_time" if water else name
    choices = PRESETS[key]
    notes = []
    if str(d) in choices:
        preset = deepcopy(choices[str(d)])
    elif name in ("rosenbrock", "rastrigin") or water:
        preset = deepcopy(choices[next(iter(choices))])
        if not water:
            notes.append(f"Dimension {d} is not recorded for this preset; extending its function and interval.")
    elif name == "plane_wave" and (args.normal in ("axis", "axis_x1", "diagonal") or args.plane_wave_normal is not None):
        preset = deepcopy(choices["3"])
        notes.append(f"Dimension {d} is not recorded; using the explicitly requested direction.")
    else:
        raise ValueError(f"No {name} preset for dimension {d}; recorded dimensions: {', '.join(choices)}")

    params = preset["target"]
    bounds = np.tile(preset["bounds"][0], (d, 1)).astype(float)
    if args.domain_low is not None:
        bounds[:] = (args.domain_low, args.domain_high)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Domain bounds must be finite and strictly increasing")
    offset = params.get("offset", 0.0) if args.output_offset is None else args.output_offset
    if not np.isfinite(offset):
        raise ValueError("--output-offset must be finite")

    def value(flag, default):
        override = getattr(args, flag)
        return default if override is None else override

    if name == "rosenbrock":
        a, b = value("rosenbrock_a", params["a"]), value("rosenbrock_b", params["b"])
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("Rosenbrock parameters must be finite")
        target = RosenbrockFunction(d, a=a, b=b)
        params = {"a": target.a, "b": target.b}
    elif name == "rastrigin":
        target = RastriginFunction(d, A=value("rastrigin_a", params["A"]))
        params = {"A": target.A}
    elif name == "plane_wave":
        normal = params["normal"]
        if args.plane_wave_normal is not None:
            normal = args.plane_wave_normal
        elif args.normal in ("axis", "axis_x1"):
            normal = np.eye(1, d)[0]
        elif args.normal == "diagonal":
            normal = np.ones(d)
        target = PlaneWaveFunction(d, beta=params["beta"], normal=np.asarray(normal),
                                  frequency=value("frequency", params["frequency"]),
                                  amplitude=value("amplitude", params["amplitude"]),
                                  phase=value("phase", params["phase"]))
        params = {"beta": target.beta, "normal": target.normal.tolist(), "amplitude": target.amplitude,
                  "frequency": target.frequency, "phase": target.phase, "feature_mode": target.feature_mode}
    elif name == "spherical_piecewise":
        inside, outside = fixed_spherical_affine_pieces(d)
        target = SphericalPiecewisePolynomialFunction(inside, outside,
                    radius=value("sphere_radius", params["radius"]), center=np.asarray(params["center"]))
        params.update(radius=target.radius, radius_source="explicit",
                      inside_beta=inside.beta, outside_beta=outside.beta,
                      inside_linear_coefficients=inside.w.tolist(), outside_linear_coefficients=outside.w.tolist())
    else:
        from oracles.shallow_water import ShallowWaterSensorOracle
        space_time = name == "shallow_water_space_time"
        physical = params["physical_parameter_bounds"]
        target = ShallowWaterSensorOracle(
            eta_left_range=tuple(value("eta_left_range", physical[0])),
            eta_right_range=tuple(value("eta_right_range", physical[1])),
            bump_height_range=tuple(value("bump_height_range", physical[2])),
            sensor_position=value("sensor_position", 3.0), final_time=value("final_time", 0.75),
            cells=value("cells", params["cells"]), channel_bounds=tuple(params["channel_bounds"]),
            dam_position=params["dam_position"], bump_center=params["bump_center"],
            bump_width=params["bump_width"], gravity=params["gravity"], dry_tolerance=params["dry_tolerance"],
            sensor_position_range=tuple(value("position_range", physical[3])) if space_time else None,
            time_range=tuple(value("time_range", physical[4])) if space_time else None,
        )
        params.update(input_coordinates=f"normalized [-1, 1]^{d}", parameter_names=list(target.parameter_names),
                      physical_parameter_bounds=target.parameter_bounds.tolist(), cells=target.cells)
        if space_time:
            params.update(sensor_position_range=list(target.sensor_position_range), time_range=list(target.time_range))
        else:
            params.pop("sensor_position_range")
            params.pop("time_range")
            params.update(sensor_position=target.sensor_position, final_time=target.final_time,
                          quantity_of_interest="final water depth at sensor")
            notes.append("No sensor-mode TACC report found. Uses the existing sensor oracle defaults x=3, T=0.75 with the recorded physical/PDE settings.")
    params["offset"] = float(offset)
    metadata = {"benchmark": name, "dimension": d, "input_bounds": bounds.tolist(), "parameters": params,
                "preset_source_report": preset["source_report"], "notes": notes,
                "explicit_overrides": {field: getattr(args, field) for field in
                    ("dimension", "domain_low", "domain_high", "output_offset", *(f for fields in groups.values() for f in fields))
                    if getattr(args, field) is not None}}
    return ConfiguredOracle(target, offset), bounds, metadata
