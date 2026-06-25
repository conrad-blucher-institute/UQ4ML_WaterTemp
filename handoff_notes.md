# Handoff Notes — OOM & Skip-Logic Fixes (2026-04-06)

## Branch: `esb_dev_add_training_trace`
- `base_tuner.py` is identical on `esb_dev_backup_2026320` — no divergence.

## What was done

### 1. OOM Fix (prior session)
- **Batched executor**: `run_tuning()` submits configs in batches of `max_workers * 2`, spinning up a fresh `ProcessPoolExecutor` per batch so worker memory is fully reclaimed between batches.
- **TF cleanup in workers**: `mape_tuner.py` and `mse_tuner.py` call `clear_session()` + `gc.collect()` + explicit `del` of model/data after each training run.

### 2. Skip-if-done bug fix (this session)
- **Pre-filtering**: Completed configs are filtered out *before* submitting to workers (lines 187–196 in `base_tuner.py`), using the O(1) `is_completed()` set lookup.
- **Early exit**: If all configs are already done, `run_tuning()` prints "Nothing to do" and returns immediately — no workers spawned.
- **Removed redundant per-worker check**: `_tune_single_config` no longer calls `is_completed()` since only remaining configs reach it.

## Known remaining items
- **Duplicate `preparingData()` call**: Workers may call data preparation twice (once in tuner init, once in `_train_model`). Not an OOM root cause but wastes memory and time. Could be consolidated.
- The progress CSV may have >1440 rows (e.g., 1441) due to duplicates from prior runs — harmless since `is_completed()` uses a set, but the "Already completed" count may exceed total grid size.
