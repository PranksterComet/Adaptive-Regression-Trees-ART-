# Shallow-water example

From the repository root, with Clawpack installed in the `art` Conda environment:

```bash
conda activate art
python oracles/shallow_water/dam_break_example.py
```

The example evolves a fully wet, initially stationary channel after instantaneous
removal of a dam at x = 0. The channel extends from -10 to 10 m. Initially the
surface elevation is 2 m on the left and 1 m on the right. A Gaussian bed bump
has height 0.3 m, center 2 m, and width parameter 0.75 m. Initial depth is
`h = eta - b`, and initial momentum `h*u` is zero.

It uses 500 finite-volume cells, gravity 9.81 m/s², extrapolation boundary
conditions, and runs to 1 second. The compiled PyClaw bathymetry f-wave solver
handles the bed slope. Five snapshots show the downstream wave and upstream
rarefaction. The default domain leaves room for the waves during this short run;
longer runs can interact with the numerical boundaries.

Outputs are written beside this script under `outputs/dam_break/`:

- `dam_break.png`: surface elevation and velocity at the saved times.
- `solution.npz`: positions, times, bed, depth, momentum, velocity, and surface.
- `summary.json`: parameters, version, timings, and diagnostics.

The reported sensor depth is linearly interpolated from cell-center depth values
at x = 3 m at the final time. Timings separate setup and time integration; they
exclude imports, plotting, and file writes. Integration includes snapshot copies,
so this is a demonstration timing, not an optimized oracle benchmark. Mass change
is a diagnostic; extrapolation boundaries permit water to enter or leave.

Parameters can be changed from the command line:

```bash
python oracles/shallow_water/dam_break_example.py --eta-left 2.5 --eta-right 1.0 --bump-height 0.4
python oracles/shallow_water/dam_break_example.py --cells 1000 --output-dir oracles/shallow_water/outputs/fine
```

The reusable ART oracle accepts normalized coordinates in `[-1, 1]^3`. It maps
them, in order, to the configured physical ranges for
`(eta_left, eta_right, bump_height)` and returns final water depth at a configured
sensor location:

```python
from oracles.shallow_water import ShallowWaterSensorOracle

oracle = ShallowWaterSensorOracle(
    eta_left_range=(1.5, 3.0),
    eta_right_range=(0.6, 1.2),
    bump_height_range=(0.1, 0.4),
    sensor_position=3.0,
    final_time=0.75,
    cells=500,
)
sensor_depth = oracle([0.0, 0.0, 0.0])
sensor_depths = oracle([[0.0, 0.0, 0.0], [0.5, -0.5, -0.5]])
```

Use `map_to_physical()` to inspect the mapping and `evaluate_physical()` for
direct experiments in physical coordinates. The oracle supports scalar and
batched inputs. Batched evaluations currently run the independent simulations
sequentially. Benchmark-level timing should be used to measure aggregate and
average oracle evaluation costs.

The ART benchmark harness uses the normalized input box, saves the trained tree
and diagnostics, and reports the average PyClaw query time:

```bash
python examples/tree_shallow_water_benchmark.py --leaf-degree 1 --max-depth 6
```

Use `--n-test` to control the independent test budget; its default is 10,000
because every test point requires a separate PyClaw simulation.

## Five-input space-time surrogate

To learn `(eta_left, eta_right, bump_height, position, time) -> depth`, enable
space-time mode. The default physical observation ranges are position `[0, 5]` m
and time `[0, 0.75]` s; all five ART coordinates are normalized to `[-1, 1]`.

```bash
python examples/tree_shallow_water_benchmark.py --space-time \
  --position-range 0 5 --time-range 0 0.75 \
  --leaf-degree 2 --max-depth 6 --n-test 2000 \
  --error-metric max_pointwise_relative --error-tolerance 0.01
```

The default 5D output directory is
`examples/tree_shallow_water_benchmark_outputs/space_time_5d/degree_2/` for
quadratic leaves. The 3D results have a separate output directory. Fixed
`--sensor-position` and `--final-time` are used only in 3D mode.

The corresponding callable is:

```python
oracle = ShallowWaterSensorOracle(
    sensor_position_range=(0.0, 5.0), time_range=(0.0, 0.75), cells=500,
)
depth = oracle.evaluate_physical([2.0, 1.0, 0.3, 3.0, 0.75])
depth_at_parameter_midpoint = oracle([0.0] * 5)
```

Every query starts a fresh simulation and integrates directly to its requested
time. At time zero, the oracle returns the interpolated initial grid state
without integration. All times use linear interpolation of cell-center depths;
at exactly `(position, time) = (0, 0)`, this averages the two cells beside the
dam. It is a numerical convention at the initial discontinuity.

Uniform 5D test coverage measures volume jointly over physical parameters,
position, and time. It does not ensure the same coverage for every fixed time
or physical scenario. Inspect space-time slices and errors near the shock as
well. Accuracy against this finite-grid oracle is distinct from convergence to
the continuum PDE; the sensor output requires a grid-refinement study.

The plotting script remains a standalone demonstration rather than a
grid-convergence study. Each run overwrites files in its selected output directory.
PyClaw may also create `pyclaw.log` in the working directory.

References: [PyClaw sill example](https://www.clawpack.org/gallery/pyclaw/gallery/sill.html)
and [dam-break example](https://www.clawpack.org/gallery/pyclaw/gallery/dam_break.html).
Clawpack is an external dependency distributed under the BSD-3-Clause license.
