# Repo notes

## North Star — design direction for the pipeline

**Push complexity into the code; keep the human interface simple.** Complexity can't be
deleted, only moved (Tesler's Law) — and with one author + thousands of runs, the *code*
should absorb it, not the user. The target: **one command, one named profile, smart defaults,
loud failures**, sitting on **low-coupling / high-cohesion** stages.

Guardrails (so we don't recreate the "bandaid → bad code" cycle):
1. **Hidden ≠ invisible** — defaults must be inspectable and overridable; "just works" must
   never mean "silently does something."
2. **Magic needs loud errors** — auto-discovery/conventions must fail clearly (never trust
   unsorted-glob ordering).
3. **Don't over-abstract** — new abstraction layers require explicit approval before code.

Reference: the tuning viz suite (`run_v16.py`) proves a simple interface is achievable
(one command, folder picker, auto-discovery, all plots) but NOT yet internal cohesion
(version sprawl, the `infer_metric_column()`-always-`val_mae` stub, headless-incompatible
tkinter picker).

Full design + visuals: **`docs/ESB_PIPELINE_DESIGN.md`**. The `main` clean-room refactor
follows the same philosophy but is behavior-preserving (paper-replication baseline).

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
