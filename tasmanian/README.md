# Tasmanian comparison

Run from the ART repository root:

```bash
conda activate art-tasmanian
python tasmanian/tasmanian_comparison.py
```

Oracle defaults now match the saved TACC experiments. The default run is
**5D Rosenbrock with offset 1**, replacing the original example's 2D,
zero-offset setup. Tasmanian training defaults remain order 2, `localp`,
initial depth 2, `fds` refinement, surplus tolerance `1e-5`, at most 10,000
training points and 20 batches. This is an example, not a tuned paper baseline.

## Oracle presets

Exact settings and representative source-report paths are frozen in
[`oracle_presets.json`](oracle_presets.json). Experiments can run without a
local `results/` directory. The factory in [`oracle_config.py`](oracle_config.py)
uses the same function implementations as ART. It prints resolved settings
before evaluation and saves them under `[target]` in `report.txt`, including
source reports and explicit overrides. Overrides describe a different oracle
whenever they change its domain or function parameters.

| `--benchmark` | Default dimension | Input domain | Output offset | Function settings |
|---|---:|---|---:|---|
| `rosenbrock` | 5 | `[-2,3]^d` | 1 | `a=1`, `b=100`; no rotation |
| `rastrigin` | 3 | 3D: `[-5.12,5.12]^3`; 4D: `[-2.5,2.5]^4`; 5D: `[-1.5,1.5]^5` | 1 | `A=10`; no rotation |
| `plane_wave` | 3 | `[-3,3]^d` | 2 | amplitude 1, frequency 4, phase 0, base beta 0; recorded `dense_signs` direction |
| `spherical_piecewise` | 3 | `[-3,3]^d` | 4.15 | Fixed affine pieces, zero center, exact recorded dimension-dependent radius |
| `shallow_water_sensor` | 3 (fixed) | Normalized `[-1,1]^3` | 0 | Water depth at fixed `x=3`, `T=0.75` |
| `shallow_water_space_time` | 5 (fixed) | Normalized `[-1,1]^5` | 0 | Water depth at input position/time: `x∈[0,5]`, `T∈[0,0.75]` |

The Rastrigin defaults select the **targeted deep** experiments. Domain sweeps
also exist; use `--domain-low` and `--domain-high` to reproduce another domain.
For Rosenbrock dimensions other than 5, the same parameters/domain are extended
to the requested dimension. For Rastrigin dimensions outside 3–5, the fallback
domain is `[-5.12,5.12]^d`. These extensions are identified in report notes as
dimensions without recorded presets. The `rotation` field in the ART reports
does not rotate the Rosenbrock/Rastrigin implementations.

### Plane waves

The default directions are the exact saved dense-sign vectors, normalized to
unit length. They are **not** an all-positive diagonal or a new random draw:

| Dimension | Signs before normalization |
|---:|---|
| 3 | `[1,-1,1]` |
| 5 | `[1,-1,1,1,-1]` |
| 8 | `[1,-1,1,1,-1,-1,1,-1]` |

`--normal axis` (or `axis_x1`) reproduces the other TACC orientation, `[1,0,...]`.
`--normal diagonal` retains the original example's all-positive direction as an
explicit alternative. Other dimensions require an explicit axis/diagonal choice
or `--plane-wave-normal` vector. Frequency is in physical input coordinates.

### Spherical piecewise

The affine coefficients are prefixes of the existing fixed vectors:

- Inside: `5 + [-0.25,0.35,-0.15,0.20,-0.10]·x`.
- Outside: `0 + [0.40,-0.30,0.20,-0.10,0.05]·x`.
- Add `4.15` to either piece. The inside branch includes the sphere boundary.

| Dimension | Exact radius recorded in TACC reports |
|---:|---:|
| 2 | 2.394139996923 |
| 3 | 2.954149913692 |
| 4 | 3.418315145003 |
| 5 | 3.83470329729 |

These are the recorded approximately 50%-volume cases. The radii are copied,
not re-estimated with a fresh Monte Carlo sample. Supported dimensions are 2–5;
`--sphere-radius` can override the radius. Changing the domain does not silently
recenter the sphere or recompute its radius.

### Shallow water

Both modes reuse `oracles/shallow_water/ShallowWaterSensorOracle` and require
Clawpack in the active environment. Normalized inputs map to these physical
parameters in order:

| Parameter | Range |
|---|---|
| Left initial surface elevation | `[1.5,3]` |
| Right initial surface elevation | `[0.6,1.2]` |
| Bed-bump height | `[0.1,0.4]` |
| Position (space-time only) | `[0,5]` |
| Time (space-time only) | `[0,0.75]` |

PDE defaults are 500 finite-volume cells, channel `[-10,10]`, dam at 0,
bump center 2, bump width 0.75, gravity 9.81, and dry tolerance `1e-6`.
The output is linearly interpolated cell-center water depth, including at
`T=0` in space-time mode. The existing solver and boundary conditions are reused.

**Only space-time results were found under `results/shallow_water/`.** Sensor
mode uses the same physical/PDE settings and the existing sensor oracle's
defaults `x=3`, `T=0.75`; it is not presented as a separate recorded TACC run.

Use `--eta-left-range`, `--eta-right-range`, `--bump-height-range`, and `--cells`
to override physical settings. Sensor mode additionally accepts
`--sensor-position` and `--final-time`; space-time accepts `--position-range`
and `--time-range`. The dimension is fixed by mode. Synthetic domain flags are
rejected for water; use the physical range flags instead. Each input row still
runs an independent simulation, so water test sets take longer than synthetic ones.

## Examples

```bash
# TACC Rosenbrock (5D, offset 1)
python tasmanian/tasmanian_comparison.py --benchmark rosenbrock

# TACC targeted Rastrigin (4D, half-width 2.5, offset 1)
python tasmanian/tasmanian_comparison.py --benchmark rastrigin --dimension 4

# An alternative recorded domain sweep
python tasmanian/tasmanian_comparison.py --benchmark rastrigin --dimension 3 --domain-low -1.5 --domain-high 1.5 --output-dir tasmanian/outputs/rastrigin_3d_width1p5

# Recorded dense-sign and axis directions
python tasmanian/tasmanian_comparison.py --benchmark plane_wave --dimension 5
python tasmanian/tasmanian_comparison.py --benchmark plane_wave --dimension 5 --normal axis --output-dir tasmanian/outputs/plane_wave_5d_axis

python tasmanian/tasmanian_comparison.py --benchmark spherical_piecewise --dimension 3
python tasmanian/tasmanian_comparison.py --benchmark shallow_water_sensor
python tasmanian/tasmanian_comparison.py --benchmark shallow_water_space_time

# Small solver smoke run, retaining all physical oracle defaults
python tasmanian/tasmanian_comparison.py --benchmark shallow_water_space_time --initial-depth 0 --max-iterations 1 --n-test 3 --output-dir tasmanian/outputs/water_smoke
```

Additional synthetic overrides: `--rosenbrock-a`, `--rosenbrock-b`,
`--rastrigin-a`, `--frequency`, `--amplitude`, `--phase`, and `--output-offset`
(alias `--offset`). An explicit offset replaces the benchmark's default shift.
Irrelevant function flags are rejected rather than silently ignored.

## Refinement, metrics, and saved results

The algorithm queries the initial grid, loads its values, requests surplus
refinement, and repeats. Oracle calls and predictions are evaluated in NumPy
batches. Each refinement batch must be loaded in full: if it would exceed
`--max-points`, the example stops at the previous complete grid. The point
budget is a ceiling, not an exact target. `--max-iterations` includes the
initial batch. Tasmanian depth is unrelated to ART tree depth.

`--tolerance` controls hierarchical surplus refinement; it does not guarantee
pointwise relative accuracy. Evaluation defaults to `10000 * dimension`
uniform test points with seed 19 (override `--n-test` to match a TACC report's test size).
The fixed test set never guides refinement or stopping. Relative errors use
`abs(prediction-truth)/max(abs(truth), floor)`, with floor `1e-12`.

Results go under `tasmanian/outputs/` (ignored by Git):

- `report.txt`: ART-style sectioned configuration, resolved oracle metadata,
  test metrics and pointwise-relative quantiles, oracle range, sparse-grid
  status, query count, and timing.
- `history.csv`: errors and cumulative training times after each loaded batch.
- `grid.tsgrid`: saved Tasmanian grid.
- `test_predictions.npz`: test inputs (in the reported input coordinates), truth,
  and predictions.
- `comparison.png`: oracle, interpolant, and relative error for 2D runs. Error
  colors use a symmetric logarithmic scale, linear below 0.001%; metrics are not clipped.

Training time includes grid creation, refinement, oracle evaluation, and loading
values; it excludes test evaluation, plotting, and file writing. Test values
are cached once. Training queries count grid observations only; test and plot
observations are separate. Prediction timing is batch throughput, not isolated
single-query latency. History times precede the next refinement request; final
report time also includes the last request.

The default output-directory name includes benchmark, dimension, order, and
refinement. It does not encode oracle overrides; use `--output-dir` to keep
different domains/directions/offsets separate. Repeating a directory overwrites
its files. Existing example outputs are not migrated to the new oracle defaults.

Reload a saved model with:

```python
import Tasmanian
import numpy as np

grid = Tasmanian.SparseGrid()
grid.read("tasmanian/outputs/rosenbrock_5d_order2_fds/grid.tsgrid")
prediction = grid.evaluateBatch(np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]))
```

See Tasmanian's [local refinement example](https://github.com/ORNL/TASMANIAN/blob/master/InterfacePython/example_sparse_grids_09.py).
