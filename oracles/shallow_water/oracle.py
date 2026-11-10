"""PyClaw oracle for a one-dimensional dam break over a bed bump."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ParameterRange = tuple[float, float]


@dataclass
class ShallowWaterSensorOracle:
    """Return final water depth at a sensor for normalized dam-break inputs.

    Inputs lie in ``[-1, 1]^3`` and map, in order, to ``eta_left``,
    ``eta_right``, and ``bump_height``. A matrix input evaluates one
    independent PyClaw simulation per row. Supply both sensor_position_range
    and time_range to append position and time as fourth and fifth inputs.
    """

    eta_left_range: ParameterRange = (1.5, 3.0)
    eta_right_range: ParameterRange = (0.6, 1.2)
    bump_height_range: ParameterRange = (0.1, 0.4)
    sensor_position: float = 3.0
    final_time: float = 0.75
    cells: int = 500
    channel_bounds: ParameterRange = (-10.0, 10.0)
    dam_position: float = 0.0
    bump_center: float = 2.0
    bump_width: float = 0.75
    gravity: float = 9.81
    dry_tolerance: float = 1e-6
    sensor_position_range: ParameterRange | None = None
    time_range: ParameterRange | None = None
    _pyclaw: Any = field(default=None, init=False, repr=False)
    _riemann: Any = field(default=None, init=False, repr=False)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = ("eta_left", "eta_right", "bump_height")
        return names + ("position", "time") if self.space_time else names

    @property
    def space_time(self) -> bool:
        return self.sensor_position_range is not None

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    def __post_init__(self) -> None:
        self.eta_left_range = _validate_range(
            self.eta_left_range, "eta_left_range"
        )
        self.eta_right_range = _validate_range(
            self.eta_right_range, "eta_right_range"
        )
        self.bump_height_range = _validate_range(
            self.bump_height_range, "bump_height_range"
        )
        self.channel_bounds = _validate_range(
            self.channel_bounds, "channel_bounds"
        )

        scalar_fields = (
            "sensor_position",
            "final_time",
            "dam_position",
            "bump_center",
            "bump_width",
            "gravity",
            "dry_tolerance",
        )
        for name in scalar_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            setattr(self, name, value)

        self.cells = int(self.cells)
        if self.cells < 50:
            raise ValueError("cells must be at least 50.")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive.")
        if self.bump_width <= 0.0:
            raise ValueError("bump_width must be positive.")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive.")
        if self.dry_tolerance <= 0.0:
            raise ValueError("dry_tolerance must be positive.")

        lower, upper = self.channel_bounds
        dx = (upper - lower) / self.cells
        dam_cell_edge = (self.dam_position - lower) / dx
        if not lower < self.dam_position < upper:
            raise ValueError("dam_position must lie inside channel_bounds.")
        if not np.isclose(dam_cell_edge, round(dam_cell_edge)):
            raise ValueError("dam_position must coincide with a finite-volume cell edge.")
        first_center = lower + 0.5 * dx
        last_center = upper - 0.5 * dx
        if not first_center <= self.sensor_position <= last_center:
            raise ValueError("sensor_position must lie between the outer cell centers.")

        if (self.sensor_position_range is None) != (self.time_range is None):
            raise ValueError("Supply both sensor_position_range and time_range.")
        if self.space_time:
            self.sensor_position_range = _validate_range(
                self.sensor_position_range, "sensor_position_range"
            )
            self.time_range = _validate_range(self.time_range, "time_range")
            if not (first_center <= self.sensor_position_range[0]
                    < self.sensor_position_range[1] <= last_center):
                raise ValueError("sensor_position_range must lie between the outer cell centers.")
            if self.time_range[0] < 0.0:
                raise ValueError("time_range must be nonnegative.")

        if self.eta_left_range[0] <= self.eta_right_range[1]:
            raise ValueError(
                "The full parameter box must satisfy eta_left > eta_right."
            )
        if self.eta_right_range[0] <= self.bump_height_range[1]:
            raise ValueError(
                "The full parameter box must satisfy eta_right > bump_height "
                "to remain fully wet."
            )

    @property
    def parameter_bounds(self) -> np.ndarray:
        """Return physical parameter bounds in oracle input order."""

        ranges = [self.eta_left_range, self.eta_right_range, self.bump_height_range]
        if self.space_time:
            ranges.extend([self.sensor_position_range, self.time_range])
        return np.asarray(ranges, dtype=float)

    def __call__(self, coordinates: np.ndarray) -> float | np.ndarray:
        """Evaluate normalized coordinates from the ART parameter domain."""

        return self.evaluate_physical(self.map_to_physical(coordinates))

    def map_to_physical(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points from the normalized box to configured physical ranges."""

        values = np.asarray(coordinates, dtype=float)
        if values.ndim not in (1, 2) or values.shape[-1] != self.dimension:
            raise ValueError(
                f"coordinates must have shape ({self.dimension},) or "
                f"(n, {self.dimension}), got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("coordinates must be finite.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError(
                f"coordinates must lie in the normalized box [-1, 1]^{self.dimension}."
            )

        bounds = self.parameter_bounds
        lower = bounds[:, 0]
        return lower + 0.5 * (values + 1.0) * (bounds[:, 1] - lower)

    def evaluate_physical(self, parameters: np.ndarray) -> float | np.ndarray:
        """Evaluate one physical point or a batch of physical points directly."""

        values = np.asarray(parameters, dtype=float)
        if values.ndim == 1:
            return self._evaluate_physical_one(values)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"parameters must have shape ({self.dimension},) or "
                f"(n, {self.dimension}), got {values.shape}."
            )
        return np.asarray(
            [self._evaluate_physical_one(row) for row in values],
            dtype=float,
        )

    def _evaluate_physical_one(self, parameters: np.ndarray) -> float:
        """Run one simulation and return final depth at the sensor."""

        values = self._validate_parameters(parameters)
        eta_left, eta_right, bump_height = values[:3]
        sensor_position, final_time = (
            values[3:] if self.space_time else (self.sensor_position, self.final_time)
        )
        pyclaw, riemann = self._load_backend()

        lower, upper = self.channel_bounds
        domain = pyclaw.Domain(
            pyclaw.Dimension(lower, upper, self.cells, name="x")
        )
        state = pyclaw.State(domain, 2, 1)
        x = state.grid.x.centers.copy()
        bed = bump_height * np.exp(
            -((x - self.bump_center) / self.bump_width) ** 2
        )
        initial_surface = np.where(
            x < self.dam_position,
            eta_left,
            eta_right,
        )
        state.aux[0, :] = bed
        state.q[0, :] = initial_surface - bed
        state.q[1, :] = 0.0
        state.problem_data.update(
            grav=self.gravity,
            dry_tolerance=self.dry_tolerance,
            sea_level=0.0,
        )

        solver = pyclaw.ClawSolver1D(riemann.shallow_bathymetry_fwave_1D)
        solver.fwave = True
        solver.limiters = pyclaw.limiters.tvd.vanleer
        solver.bc_lower[0] = solver.bc_upper[0] = pyclaw.BC.extrap
        solver.aux_bc_lower[0] = solver.aux_bc_upper[0] = pyclaw.BC.extrap
        solver.verbosity = 0
        solution = pyclaw.Solution(state, domain)
        # T=0 uses exactly the same cell-center interpolation as positive times.
        # At x=dam_position this averages the adjacent initial cell values.
        if final_time > 0.0:
            solver.evolve_to_time(solution, final_time)

        final_state = solution.state.q
        depth = final_state[0]
        if not np.isclose(solution.t, final_time):
            raise RuntimeError("PyClaw did not reach the requested final time.")
        if not np.all(np.isfinite(final_state)):
            raise RuntimeError("PyClaw produced nonfinite state values.")
        if np.any(depth <= 0.0):
            raise RuntimeError("PyClaw produced nonpositive water depth.")
        return float(np.interp(sensor_position, x, depth))

    def _validate_parameters(self, parameters: np.ndarray) -> tuple[float, ...]:
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"parameters must have shape ({self.dimension},), got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite.")
        bounds = self.parameter_bounds
        if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
            raise ValueError("parameters lie outside the configured physical ranges.")

        eta_left, eta_right, bump_height = map(float, values[:3])
        if not eta_left > eta_right > bump_height >= 0.0:
            raise ValueError(
                "parameters must satisfy eta_left > eta_right > bump_height >= 0."
            )
        return tuple(map(float, values))

    def _load_backend(self) -> tuple[Any, Any]:
        if self._pyclaw is None or self._riemann is None:
            try:
                from clawpack import pyclaw, riemann
            except ImportError as exc:
                raise ImportError(
                    "ShallowWaterSensorOracle requires Clawpack/PyClaw. "
                    "Activate the project's 'art' Conda environment."
                ) from exc
            self._pyclaw = pyclaw
            self._riemann = riemann
        return self._pyclaw, self._riemann


def _validate_range(values: ParameterRange, name: str) -> ParameterRange:
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values.")
    lower, upper = map(float, array)
    if lower >= upper:
        raise ValueError(f"{name} must satisfy lower < upper.")
    return lower, upper
