# Random-forest comparison

Run from the ART repository root in the environment that contains
scikit-learn:

```bash
conda activate art-tasmanian
python random_forest/forest_comparison.py --benchmark rosenbrock
```

The runner uses the same oracle presets, output offsets, independent uniform
test sets, active-Gaussian weighted test set, metrics, quantiles, and report
layout as `tasmanian/tasmanian_comparison.py`.

The default forest is a strong, unpruned deterministic-oracle baseline:

- 500 trees.
- Squared-error splitting.
- Unlimited depth and leaf count.
- Minimum leaf size 1 and minimum split size 2.
- All input features considered at every split.
- Full-size bootstrap samples.
- No impurity threshold or cost-complexity pruning.
- One worker for timing comparability with ART.
- Forest seed 1.

The training set contains 10,000 independent uniform oracle queries by
default. Use `--training-points` (or `--max-points`) to match the exact oracle
query count of an ART run. Uniform test sets use `10000 * dimension` points.

```bash
python random_forest/forest_comparison.py \
  --benchmark plane_wave \
  --dimension 5 \
  --training-points 50100 \
  --forest-seed 1 \
  --output-dir random_forest/outputs/plane_wave_5d_seed1
```

Results include:

- `report.txt`: ART-style performance, quantiles, oracle range, forest size,
  and timing sections.
- `forest.joblib`: reloadable fitted `RandomForestRegressor`.
- `test_predictions.npz`: fixed test inputs, oracle values, and predictions.
- `history.csv`: one row describing the completed forest fit.
- `comparison.png`: oracle, forest, and relative-error panels for 2D runs.

Reload a model with:

```python
import joblib

forest = joblib.load("random_forest/outputs/my_run/forest.joblib")
prediction = forest.predict(X)
```
