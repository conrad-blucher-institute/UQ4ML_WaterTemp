# UQ4ML WaterTemp - TODO Tracker

Last updated: 2026-03-21

---

## CRITICAL: OOM / Skip-Resume Bugs (from repo_architecture_fixes_needed.md)

| #   | Issue                                                      | Status    | Notes                                                                                                                                              |
| --- | ---------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | crps_tuner.py syntax error (hyphen in module name)         | NOT FIXED | crps_tuner not in active use yet                                                                                                                   |
| 2   | `preparingData()` called TWICE per config                  | NOT FIXED | Preview call at line ~124 + actual call at ~142 in both tuners. Need to cache the first result or infer shape differently                          |
| 3   | No `tf.keras.backend.clear_session()` after training       | NOT FIXED | TF graph accumulates across configs -> OOM. Need to add `K.clear_session()` + `gc.collect()` + `del model` in `_tune_single_config` after each run |
| 4   | Models built 2-3 times per config (preview + rebuild)      | NOT FIXED | Related to #2. The preview builds a model, then it's rebuilt after data shape is known. Leaked model objects consume memory                        |
| 5   | All 1,440 futures submitted at once to ProcessPoolExecutor | NOT FIXED | `base_tuner.py:205` submits ALL configs as futures simultaneously. Should batch them (e.g. 10-20 at a time) to limit concurrent memory             |
| 6   | `is_completed()` fallback ignores `run_num`                | FIXED     | `tuner_utils.py` now checks `run_num` in the row-by-row comparison                                                                                 |
| 7   | Errored configs never retried on restart                   | FIXED     | `is_completed()` now skips rows with status starting with 'error'                                                                                  |

### OOM Summary

Issues #2-5 are the OOM root causes. **None of them are fixed yet.** The tuner will still OOM on long runs. The fixes needed:

- **#2+#4**: Call `preparingData()` once, cache the result, infer `input_shape` from the cached data, build the model once.
- **#3**: After `_train_model()` returns in `base_tuner.py:_tune_single_config()`, add:
  ```python
  import tensorflow.keras.backend as K
  import gc
  K.clear_session()
  gc.collect()
  ```
- **#5**: Replace the single `futures = {executor.submit(...) for config in configs}` with a batched submission loop that only has N futures in flight at once.

---

## Completed This Session (2026-03-21)

- [x] Added `prepare_independent_year()` to `utils_mse_crps.py` — shared utility for 2021 eval
- [x] Added post-hoc `val_mae12` computation to both tuners (using `mae12` from `src.helper.utils`)
- [x] Added post-hoc 2021 independent test set evaluation to both tuners (`mae_2021`, `mape_2021`/`mse_2021`, `mae12_2021`)
- [x] Removed `loss_2021` (doesn't make sense as a metric)
- [x] Promoted `val_mae`, `val_mae12`, `mae_2021`, `mae12_2021` to top-level CSV columns in `tuner_utils.py`
- [x] Fixed `is_completed()` to check `run_num` and skip errored rows (#6, #7)
- [x] Added `.keras` file saving (flat folder at `{run_dir}/keras_files/`)
- [x] Updated `data_loader.py` to use `header=0` and named columns (position-independent)
- [x] Added Y-axis metric dropdown to `scatter_plot_v8.py`
- [x] Added missing data count badge to scatter plot
- [x] Changed output path to `results/esb_tuner_MM-DD-YYYY/`

## Completed Last Session (2026-03-20)

- [x] Analyzed `run_all_tuners.py` — confirmed skip/resume logic exists but has bugs
- [x] Identified OOM root causes (7 issues)
- [x] Created `repo_architecture.drawio`
- [x] Created `repo_architecture_fixes_needed.md` with full code quality analysis (20 prioritized issues)
- [x] Updated `coupling_cohesion.py` to use `rglob` for recursive scanning

## Still TODO (Current Priority)

- [ ] **Fix OOM issues #2-5** (see summary above) — must fix before running full 1440x10 tuning
- [ ] Replace `iterrows()` in `offSetCreator()` (`utils_mse_crps.py:582`) with vectorized pandas
- [ ] Run `coupling_cohesion.py` on the full repo
- [ ] Fix remaining HIGH priority issues (#8-15 from `repo_architecture_fixes_needed.md`)
- [ ] Fix MEDIUM priority issues (#16-20)
- [ ] Update `run_esb2020_2021_inference.py` to import `prepare_independent_year` from `utils_mse_crps.py` instead of defining its own

---

## Future TODO

### Time Series Visualization for 1440 Tuning Results

Need a way to present time series plots for each of the 1440 tuning configurations without creating 1440 separate HTML files or one massive slow file.

**Options to explore:**

- **Lazy-loading single HTML**: one HTML file with a config selector dropdown that loads data on-demand from a JSON sidecar file. Only renders the selected config's plot — fast startup, no 1440 separate files.
- **Dash app**: a lightweight local Dash/Streamlit app with dropdowns for (lead_time, cycle, activation, layers, neurons). Renders one plot at a time from the saved `.keras` files + 2021 data. Most interactive, but requires running a server.
- **Pre-rendered thumbnail grid + click-to-expand**: generate small static PNGs for all 1440, display as a grid in one HTML, click to open full interactive Plotly plot.

### Refactor Utils into Sub-modules

Split `utils_mse_crps.py` (600+ lines) and other utils files into focused sub-modules:

```
src/helper/
  data_io.py          -> readingData, splittingData, dateTimeRetriever
  feature_eng.py      -> creatingAdditionalColumns, offSetCreator, prepare_independent_year
  preprocessing.py    -> deletingMissingValues, countingMissingValues, dataframe_checker
  reshaping.py        -> reshaping
  losses.py           -> crps_loss, crps
  constants.py        -> MISSING_VALUE = -999
```

Move driver scripts out of `src/driver/` to top-level or `scripts/`:

```
scripts/
  train_mse_crps.py          (was: src/driver/operational_mse_crps_driver.py)
  run_inference_2021.py       (was: esb2020_2021_experiment/run_esb2020_2021_inference.py)
  visualize_results.py        (was: src/driver/visualization_driver.py)
```

### Memory Profiling

Before running the full 1440x10 grid search, profile memory usage to verify OOM fixes work:

- **`memory_profiler`** — Python lib, decorates functions with `@profile`, shows line-by-line memory usage. Best for finding which line allocates the most. Install: `pip install memory-profiler`, run: `python -m memory_profiler your_script.py`
- **`tracemalloc`** — built into Python stdlib, no install needed. Take snapshots before/after training a model, diff them to see what wasn't freed.
- **`cProfile`** — what your coworkers use. Good for **CPU time** profiling (which functions take longest), but does NOT measure memory. Use it to find slow functions, not memory leaks.
- **`objgraph`** — shows reference graphs for Python objects. Useful for finding why an object isn't being garbage collected (e.g., a TF model with circular references).

**Recommended approach**: Use `tracemalloc` snapshots in `_tune_single_config` (before and after each model) to verify memory returns to baseline after `clear_session()` + `gc.collect()`.

### Clean Up Visualization File Variants

90+ versioned copies (v1-v8) in `tuning-viz-folder/`. Keep only the latest version of each plot type, delete the rest.

# Summary for after compaction

Session summary for next chat:

Read todo.md and repo_architecture_fixes_needed.md for full context. This session we:

Added prepare_independent_year() to utils_mse_crps.py (shared 2021 eval utility)
Added post-hoc val_mae12, mae_2021, mse_2021/mape_2021, mae12_2021 to both tuners using that function
Added .keras saving to flat keras_files/ folder inside each run dir
Promoted metrics to top-level CSV columns in tuner_utils.py
Updated data_loader.py to use header=0 (named columns)
Added Y-axis metric dropdown + missing count badge to scatter_plot_v8.py
Debug mode now uses timestamped folder, 1 worker, 10 models, 2 reps; production uses results/esb_tuner/ (resumable)
2021 eval had a shape mismatch error — fixed by using prepare_independent_year() instead of manual calls. Not yet retested.
OOM issues #2-5 are NOT fixed. Must fix before full 1440x10 run.
