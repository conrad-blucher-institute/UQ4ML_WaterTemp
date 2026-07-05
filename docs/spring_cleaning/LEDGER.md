# Spring-Cleaning Decision Ledger

**Workflow:** Claude catalogues + recommends with evidence; Hector fills the **DECISION**
column row by row (`delete` / `keep` / `move` / `port` / `defer` — or anything else).
Executed decisions happen on a dedicated branch, **one commit per batch**, ledger rows
quoted in the commit message. Git history is the archive — no `_old/` folders.

**Gates after every deletion batch:** both golden checks (unscaled `d80ae54`, scaled
`3ae83946`) + `esb/tests/` + `python -m esb run --profile mape_smoke --dry-run`.

**Timing rule:** no batches execute while a tuning campaign is writing to `results/`.

Evidence snapshot: **2026-07-05** (import greps run fresh; cross-repo rows lean on the
parent CLAUDE.md redundancy map from late June — re-verify those before executing).
Batches are ordered easiest-decision-first; spend your judgment budget on Batch 4+.

Legend — **Risk**: `none` = zero importers, loud if wrong · `low` = importers are
themselves dead · `med` = referenced by legacy-but-runnable code · `high` = live path
or contract-touching.

---

## Batch 1 — Self-declared dead / broken / strays (easiest approvals)

| File | Category | Evidence | Suggested action | Risk | DECISION |
|---|---|---|---|---|---|
| `src/driver/crps_mme_runner copy.py` | stray editor copy | filename has literal " copy"; last commit 2025-12-02 | delete | none | |
| `esb2020_2021_experiment/run_esb2020_2021_visualization_NOT-WORKING-RN.py` | self-declared broken | filename; imports bare `evaluations.*` (broken path form) | delete | none | |
| `esb_03_2026_tune_experiment/crps_tuner.py` | unimportable | line 15 `from esb_03-2026_tune_experiment...` — hyphens = invalid module name; body = NotImplementedError stubs; only reference is a commented-out import in `run_all_tuners.py:57` | delete (recreate from `mape_tuner.py` if a CRPS tuner is ever wanted) | none | |
| `src/driver/parallel_tuner_setup_llm_needs_adapting.py` | self-declared stub | filename says it; never imported | delete | none | |
| `test_deletion.py` (root) | debug stray | root-level scratch; last commit 2026-02-09 cleanup sweep; no importers | delete | none | |
| `test_preparingData.py` (root) | debug stray | same sweep; no importers | delete | none | |
| `test3.py` (root) | debug stray | same sweep; no importers | delete | none | |
| `verify_fix.py` (root) | debug stray | same sweep; no importers | delete | none | |
| `diagnose_hyperparams.py` (root) | debug stray | same sweep; no importers | delete | none | |
| `esb_03_2026_tune_experiment/tuning-viz-folder/test_06.py` | scratch test | superseded by `test_06_v6.py`; neither imported | delete | none | |
| `esb_03_2026_tune_experiment/tuning-viz-folder/test_06_v6.py` | scratch test | scratch by name; verify it's not a smoke script you still run | delete | none | |
| `src/helper/tuner.py`-era leftovers — *none in this repo*; `tuner.py` with the `"PUT CUSTOM LOSS FUNCTION HERE"` placeholder lives in coolTurtles (see Batch 6) | — | — | — | — | |

## Batch 2 — Superseded versions (keep latest, delete the rest)

The versioned-file convention left old versions in-tree. WaterTemp's tuning-viz folder is
already mostly pruned (only `run_v16.py`, `unified_plot_v15.py`, `timeseries_compare_v16.py`
remain tracked) — the sprawl lives in **GridExplorer**, which is a *worktree of this repo*:
version pruning there is really a branch-merge/checkout decision, not file deletion (Batch 6).

| File | Category | Evidence | Suggested action | Risk | DECISION |
|---|---|---|---|---|---|
| `esb_03_2026_tune_experiment/tuning-viz-folder/adapt_results.py` | superseded | `adapt_results_v2.py` exists; v1 unreferenced | delete v1 | none | |
| `esb_03_2026_tune_experiment/tuning-viz-folder/serve_timeseries.py` | superseded | `serve_timeseries_v2.py` exists | delete v1 | none | |
| `esb_03_2026_tune_experiment/tuning-viz-folder/run_all.py` | pinned to dead version | `run_all.py:122` pinned to run_v16 behavior; decide vs `run_v16.py` — keep ONE runner entry point | keep `run_v16.py`, delete `run_all.py` | low | |
| `esb_03_2026_tune_experiment/tuning-viz-folder/tuning_viz/plot_types/__init__.py` | stale exports | exports v1 function names; run scripts import versioned modules directly | empty the exports (keep file for package) | low | |
| `scripts/run_tests.py` + `scripts/tuning_tests/run_tests.py` | superseded | replaced by `scripts/tuning_tests/run_all_tests.py` (untracked yet); `tuning_tests/run_tests.py` already deleted in working tree | delete both once `run_all_tests.py` is committed | none | |
| `esb_03_2026_tune_experiment/preparingdata_call_benchmark.py`, `tuning-viz-folder/benchmark_predict.py`, `esb2020_2021_experiment/benchmark_creatingAdditionalColumns.py` | one-off benchmarks | answered their question; results (if kept) belong in a doc, not runnable strays | delete; paste any numbers worth keeping into `docs/` first | none | |
| `esb_03_2026_tune_experiment/tuning-viz-folder/USAGE_EXAMPLES.py` | docs-as-code | examples file; likely stale vs run_v16 flags | fold into a README in that folder, delete the .py | none | |

## Batch 3 — Zero-importer orphans (import graph says dead; confirm not run-directly)

| File | Category | Evidence | Suggested action | Risk | DECISION |
|---|---|---|---|---|---|
| `src/helper/utils_from_coolturtles.py` | orphan copy | zero importers (grep 2026-07-05); duplicates coolTurtles logic | delete | none | |
| `src/driver/tuner_retriever.py` | orphan | zero importers; superseded by tuning-viz suite | delete | none | |
| `src/driver/tuner_models_retriever.py` | orphan | zero importers | delete | none | |
| `src/driver/tunerResults.py` | orphan + hardcoded `C:\Users\cduff4\...` path | only import is its own `my_parser` use; nothing imports it; path broken on every current machine | delete | none | |
| `src/driver/hyperparameter_visualizations.py` | orphan (diverged twin in coolTurtles) | zero importers here; predates VE suite | delete (coolTurtles copies decided in Batch 6) | none | |
| `run_spaghetti_graphs.py` (root) | root-level driver | imports `evaluations.*` bare form (works only with `src/` on path); overlaps `quick_viz_spaghetti.py` | decide which spaghetti script is canonical, delete the other | low | |
| `esb2020_2021_experiment/run_esb2020_2021_inference_slow.py` + `_less_slow.py` | superseded iterations | `run_esb2020_2021_inference.py` is the survivor by naming | delete `_slow` + `_less_slow` | low | |
| `code_analysis/analysis/coupling_cohesion.py` | analysis one-off | fed the June architecture audit; keep only if you'll rerun it | delete (or move under `scripts/`) | none | |

## Batch 4 — Legacy-but-referenced (the real judgment calls; take these slowly)

| File | Category | Evidence | Suggested action | Risk | DECISION |
|---|---|---|---|---|---|
| `src/helper/utils.py` (old 10-yr pipeline) | legacy reference | still the home of real `crps_loss` history; imported by legacy drivers | keep until Stage D completes; then re-evaluate | high | |
| `src/helper/utils_mse_crps.py` | live god module | THE Stage D migration source — not a spring-cleaning target | keep (Stage D shrinks it) | high | |
| `src/helper/utils_pnn.py` + `src/driver/pnn_mme_driver.py` | PNN path | imported only by each other + metrics from utils_mse_crps; PNN work parked on dev-Proto_Incorp stash | keep (parked project); revisit after R6 PNN→NLL rename | med | |
| `src/driver/operational_mse_crps_driver.py` | legacy driver | superseded by `esb run` for tuning, but still the non-tuner training driver; known warts (hardcoded hyperparams, backslash path) | keep; candidates for Stage F replacement, not deletion | high | |
| `src/driver/crps_mme_runner.py` | legacy runner | RandomSearch path formally PARKED as future feature (Stage A decision R3); known `output_units` NameError for non-CRPS | keep-as-parked; add a header comment saying so | med | |
| `src/driver/visualization_driver.py` + `src/evaluations/*` (aggregate_tables, boxplot_figures, cross_validation_visuals_paper, evaluation_functions) | paper-figure chain | interlinked import chain (verified 2026-07-05); produces the paper figures; `evaluation_functions.py` is the canonical metrics impl | keep the whole chain intact | high | |
| `src/helper/my_parser.py` | legacy shared parser | imported by pnn_mme_driver, crps_mme_runner, pnn_to_csv, tunerResults | keep while any importer lives | med | |
| `src/helper/Logger.py`, `src/helper/job_iterator.py` | legacy helpers | imported by pnn_mme_driver only | keep while PNN path lives | low | |
| `src/helper/pnn_to_csv.py` | PNN utility | part of PNN path | keep with PNN decision | low | |
| `src/helper/losses.py`, `src/helper/metrics.py` | extracted canonical modules | live (mape_tuner imports metrics; utils_mse_crps imports losses) | keep — also the port-back-to-main candidates | high | |
| `esb2020_2021_experiment/` (remaining runnable scripts) | past experiment | experiment concluded; scripts reproduce it | decide: archive the folder wholesale after extracting any doc-worthy results | med | |
| `esb_03_2026_tune_experiment/mse_tuner.py` | half-wired tuner | has dropout `_build_model` but NOT scale wiring; double `preparingData()` waste | keep (campaign infra) but flag: finish scale wiring or explicitly declare mape-only | med | |
| `esb_03_2026_tune_experiment/run_all_tuners.py` | orchestrator | superseded by `esb run` as front door? verify nothing but docs reference it | probably delete after Stage D; defer | med | |

## Batch 5 — Code-level dead code inside live files (edits, not deletions)

| Location | What | Suggested action | Risk | DECISION |
|---|---|---|---|---|
| `utils_mse_crps.py` `og_splittingData()` | never called | delete function (behavior-neutral, flag in commit) | low | |
| `utils_mse_crps.py` unreachable block after return in `countingMissingValues()` | dead -999 check | delete block | low | |
| `utils_mse_crps.py` / coolTurtles `dataframe_checker()` `exit()` | kills process | replace with raised exception (Stage D rule already allows) | low | |
| dead `independent_year` block in `preparingData` (R2) | feature-engineers then overwrites | Stage D decision — leave for Stage D, not this pass | med | |
| `visualization_driver.py:157` `runFig5 = False` | figure_5 never called | decide: dead flag or intentional toggle — ask-Hector row | low | |

## Batch 6 — Not-this-repo rows (branches, worktree, sibling repo)

| Item | Category | Evidence | Suggested action | Risk | DECISION |
|---|---|---|---|---|---|
| branch `esb_dev_auto_agent` | dead branch | "idk temp" | delete branch | none | |
| branch `dev` | dead branch | fully contained in esb_dev | delete branch | none | |
| branch `esb_dev_docs_restored` | dead branch | only commit already in esb_dev | delete branch | none | |
| branch `esb_dev_refactoring_into_config_files` | subset branch | strict subset of esb_dev_normalization; normalization logic since lifted into Stage C | delete branch | low | |
| branch `esb_dev_normalization` | harvested | scaler + prepare_independent_year lifted in Stage C; R1 confirmed byte-identical | delete after confirming nothing else unique | low | |
| branch `esb_dev_add_training_trace_restoring_grendal` | competing scaler | second StandardScaler impl — Stage C chose the other one | delete after skim for the "training trace" part | med | |
| branch `dev-Proto_Incorp` (+ laptop stash) | abandoned prototype | Jarett's; stash holds import fixes + serializable decorators worth cherry-picking | harvest stash items, then treat branch as read-only reference | med | |
| **UQ4ML_GridExplorer worktree** | worktree divergence | holds newer viz versions (run_v18, unified v16, timeseries v18) than WaterTemp tracks | merge the viz-suite branch back into WaterTemp, then prune old versions ONCE, in one place | med | |
| **coolTurtles repo** | sibling repo cleanup | its own dead-code table exists in parent CLAUDE.md (ObservationOLD dup, tuner.py placeholder loss, MarinaMLP, exit() calls, pickle race) | separate ledger, same workflow, after this repo's pass | — | |
| `main` port-back (`main_claude_refactor`) | port, not cleanup | cherry-pick losses/metrics extraction + path fixes + dead-code removal into paper baseline | separate parked task (already recorded); not this ledger | — | |

---

## Execution order (when campaign is quiet)

1. Batch 1 + 2 + 3 approvals → one branch `spring_cleaning`, three commits, gates after each.
2. Batch 5 edits → one commit each, golden checks mandatory (these touch live files).
3. Batch 4 → mostly "keep" decisions recorded here; the few deletes ride along.
4. Batch 6 → branch deletions (after push-state check), GridExplorer merge as its own task.
