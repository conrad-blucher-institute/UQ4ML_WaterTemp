# Repo notes

## Parked WIP on this laptop (dev-Proto_Incorp)

There is a stash on this laptop holding uncommitted work for `dev-Proto_Incorp`, parked while focus shifted back to `esb_dev`.

- **Stash message:** `proto-incorp WIP: import fixes + register_keras_serializable + run configs`
- **Created via:** `git stash push -u` (includes untracked files)
- **What's in it:**
  - Import path fixes: `from evaluations.X` → `from src.evaluations.X` across `visualization_driver.py`, `TWC_AirTemp_Integrated_Visuals.py`, `aggregate_tables.py`, `cross_validation_visuals_paper.py`, `utils_visuals.py`
  - `@register_keras_serializable()` decorators added to `crps()` and `crps_loss()` in `src/helper/utils_mse_crps.py`
  - Personal run-config tweaks in `src/driver/visualization_driver.py` (`cycles=[8]`, `iterations=30`, `obsVsPred='2021'`, dropped `'mse'` from architectures, added lead time `120`) — likely should NOT be committed; revert before staging
  - Small comment/print additions in `TWC_Air_Temp_Proto_Incorp.py`, `pnn_mme_driver.py`, `TWC_AirTemp_Integrated_Visuals.py`
  - Untracked: `env_snapshot.txt`, `env_snapshot_revisions.txt`, `ram_and_cpu_check.py` (has hardcoded absolute MODEL_PATH — fix before committing), `run/` folder with driver-runner wrappers

### When resuming

`git stash pop` will conflict against Jarett's full-file rewrite of `src/helper/utils_mse_crps.py` (commit `97df30f` on `origin/dev-Proto_Incorp`). The decorator additions need to be re-applied on top of the rewritten file rather than auto-merged.
