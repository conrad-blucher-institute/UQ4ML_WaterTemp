# UQ4ML WaterTemp - Repository Architecture


---

## Code Quality Analysis - What To Care About

### CRITICAL Issues (Fix First)

| # | Issue | Where | Impact |
|---|-------|-------|--------|
| 1 | **crps_tuner.py has a syntax error** in import (hyphen in module name) | crps_tuner.py:15 | Won't import at all |
| 2 | **`preparingData()` called TWICE per config** in both MAPE and MSE tuners | mape_tuner.py:130+152, mse_tuner.py:124+142 | Doubles data loading time; massive memory waste |
| 3 | **No `tf.keras.backend.clear_session()`** after training each model | mape_tuner.py, mse_tuner.py | TF graph accumulates in memory -> OOM |
| 4 | **Models built 2-3 times per config** (preview + rebuild after data shape) | mape_tuner.py:147+189, mse_tuner.py:139+176 | Leaked model objects consume memory |
| 5 | **All 1,440 futures submitted at once** to ProcessPoolExecutor | base_tuner.py:197 | All workers try to allocate GPU memory simultaneously |
| 6 | **`is_completed()` fallback ignores `run_num`** | tuner_utils.py:103-113 | Run 2 skipped because run 1 key exists in set |
| 7 | **Errored configs are never retried** on restart | tuner_utils.py:89-113 | `is_completed` finds the error row and returns True |

### HIGH Priority (Architectural Debt)

| # | Issue | Where | Why It Matters |
|---|-------|-------|----------------|
| 8 | **`crps_mme_runner.py:temp()` is 312 lines** with 7 nested loops | crps_mme_runner.py:67-379 | Untestable, unmaintainable god-function |
| 9 | **`preparingData()` is 132 lines** doing load + transform + split + validate | utils_mse_crps.py:48-180 | Single Responsibility violation; hard to debug |
| 10 | **Hardcoded `-999` as missing value sentinel** scattered everywhere | utils_mse_crps.py:251,262,267,272,467 | Should be a named constant |
| 11 | **Dead code** after premature return | utils_mse_crps.py:447-454 | `countingMissingValues()` returns early, rest unreachable |
| 12 | **`iterrows()` in `offSetCreator()`** | utils_mse_crps.py:582-598 | Orders of magnitude slower than vectorized pandas |
| 13 | **Windows backslash paths** in save paths | crps_mme_runner.py:199,318 | Breaks on Linux/Mac; use `Path` |
| 14 | **Logger.py `close()` is commented out** | Logger.py:28 | File handles never explicitly closed |
| 15 | **Undefined `output_units` referenced before assignment** | crps_mme_runner.py:170 | Will throw NameError at runtime |

### MEDIUM Priority (Code Smells)

| # | Issue | Where |
|---|-------|-------|
| 16 | Duplicate try-except import pattern (tries 2 paths) | mape_tuner.py:100-106, mse_tuner.py:100-106 |
| 17 | Bare `except Exception` silently swallows callback import failures | mape_tuner.py:221, mse_tuner.py:206 |
| 18 | CSV field names duplicated in 3 places (not DRY) | tuner_utils.py:77-80, 139-141 |
| 19 | Identical MAPE/MSE try-except blocks in orchestrator | run_all_tuners.py:124-150 vs 155-181 |
| 20 | 90+ visualization file variants in tuning-viz-folder (v1-v8 copies) | tuning-viz-folder/ |

### What To Study To Understand The Repo

1. **Start here**: `src/driver/operational_mse_crps_driver.py` - the main training loop. Understand how cycles, iterations, and lead times combine.
2. **Data pipeline**: `src/helper/utils_mse_crps.py::preparingData()` - this is the bottleneck and the heart of feature engineering.
3. **Loss functions**: `utils_mse_crps.py::crps_loss()` and `utils.py::crps_loss()` - understand CRPS vs MSE tradeoffs.
4. **Evaluation**: `src/evaluations/evaluation_functions.py` - the probabilistic metrics (PITD, spread-skill) are the novel contribution.
5. **Cross-validation**: Understand the 10-fold rolling origin scheme (cycles 0-9) - this is core to the experimental design.
6. **The tuning framework**: `base_tuner.py` -> `mse_tuner.py` - your newest code, where the OOM lives.

### Quick Wins For Improvement

1. Extract `MISSING_VALUE = -999` as a module constant
2. Replace `iterrows()` with vectorized mask: `df.loc[df["Air Average"] != -999, "Air Average"] += IPPOffset`
3. Delete unreachable code in `countingMissingValues()`
4. Add `tf.keras.backend.clear_session()` + `gc.collect()` + `del model` after each training run
5. Cache `preparingData()` result instead of calling it twice
6. Use `pathlib.Path` everywhere instead of string concatenation with backslashes
