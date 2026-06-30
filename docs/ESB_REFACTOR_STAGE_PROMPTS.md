# ESB Pipeline Refactor — Stage Prompts

Paste-ready prompts to drive the ESB pipeline redesign **one stage at a time**, each in a
fresh conversation. They implement the strangler-fig migration in `docs/ESB_PIPELINE_DESIGN.md`.

## How to use this file
- Run stages **in order** (A → B → C → D). Each depends on the previous.
- Run from the repo root. Work on a dedicated branch `esb_refactor` (created in Stage B), **not**
  `esb_dev` directly, until each stage is reviewed.
- If you want to keep another convo open editing code simultaneously, give the refactor convo its
  own **git worktree** (`git worktree add ../UQ4ML_stageB esb_refactor`) to avoid clobbering the
  shared working directory. Otherwise, only one convo edits code at a time.
- Each prompt tells the agent to read the two design docs first — that's the shared context, so
  the prompts stay short.

---

## SHARED PREAMBLE (the agent reads this from the repo, but it's restated here)

> Before doing anything, read `docs/ESB_PIPELINE_DESIGN.md` (especially the North Star and the
> three guardrails) and `docs/ESB_PIPELINE_CURRENT.md` (the as-is). Internalize:
> - **North Star:** push complexity into the code; the human interface stays simple (one command,
>   named profile, smart defaults, loud failures); low coupling / high cohesion.
> - **Guardrails:** (1) hidden ≠ invisible — defaults inspectable & overridable; (2) magic needs
>   loud errors — never silently pick (no unsorted-glob ordering); (3) don't over-abstract — new
>   abstraction layers require my explicit approval **before** you write them.
> - **Discipline:** propose before building anything structural; ask when ambiguous; small,
>   reviewable commits; never bundle unrelated changes.
>
> **Global working rules (apply to EVERY stage):**
> 1. **Always work on a dedicated branch**, never on `esb_dev`/`main` directly. Create it at the
>    start of the stage (e.g. `git checkout -b esb_refactor` for Stage B; rename passes get their
>    own branch too). State the branch name in your first message.
> 2. **Commit as you go** with small, focused git commits — one logical change per commit, clear
>    messages, never bundle unrelated changes. End each commit message with the required
>    `Co-Authored-By` trailer.
> 3. **Every report/analysis deliverable is written to a file in `docs/`** (e.g.
>    `docs/ESB_STAGE_X_REPORT.md`), not just printed in chat, so it's reviewable later.
> 4. **Every report starts with a "TL;DR — Visual Summary" section of mermaid diagrams** at the
>    very top (grouped/vertical `flowchart TB` for readability), summarizing the dense tables that
>    follow — same pattern as `docs/ESB_STAGE_A_REPORT.md`. The detailed tables stay below.

---

## STAGE A — Design validation & concrete schema/contracts (NO CODE)

```
Read docs/ESB_PIPELINE_DESIGN.md and docs/ESB_PIPELINE_CURRENT.md, then survey the real code:
src/driver/crps_mme_runner.py, src/driver/operational_mse_crps_driver.py,
src/helper/my_parser.py, src/helper/utils_mse_crps.py (especially preparingData, reshaping,
prepare_independent_year), and the configs/ files.

Do NOT write or edit any code. Produce a design-validation report — WRITE IT TO
docs/ESB_STAGE_A_REPORT.md (not just chat), and START the file with a "TL;DR — Visual Summary"
section of mermaid diagrams (vertical flowchart TB, grouped) summarizing the sections below.
The report contains:

1. CONFIG SCHEMA (the single source of truth, to be built in Stage B): a consolidated table of
   EVERY option the pipeline needs — name, type, default, allowed values/choices, help text,
   and which stage consumes it. Pull these from my_parser.py, the configs/*.txt files, AND the
   hardcoded hyperparameters in operational_mse_crps_driver.py. Flag duplicates/conflicts.

2. STAGE CONTRACTS: the typed containers passed between stages (proposed dataclasses:
   RawFrames, FeatureFrames, CleanFrames, Split, Arrays, ScaledArrays, Metrics) — list fields
   and types for each.

3. BEHAVIORAL QUIRKS THAT MUST BE PRESERVED through the refactor (these are the landmines):
   e.g. the col_start = 1 (descending) / 3 (ascending) invariant; sorted-glob that skips the
   independent year; -999 + NaN handling; batch_size = full training-set length; any others you
   find. Cite file:line.

4. GAPS / RISKS / OPEN QUESTIONS for me to decide before Stage B.

STOP after the report. I will review and approve before any code.
```

---

## STAGE B — Skeleton (strangler fig), strictly behavior-preserving

```
Read docs/ESB_PIPELINE_DESIGN.md, docs/ESB_PIPELINE_CURRENT.md, and the approved Stage A report.

Goal: build the new entry point + orchestration as a THIN SHELL over the EXISTING grid-search
MAPE-tuning code. No pipeline logic moves yet. This is the strangler-fig skeleton.

IMPORTANT (from Stage A, R1): the legacy ESB pipeline DOES NOT RUN as-is on esb_dev
(preparingData calls splittingData with 5 args but it now requires 6 — a TypeError). So there is
NO working "old output" to diff against. Verification is therefore FORWARD-CORRECTNESS, not
reproduce-the-past: get a MAPE grid-search run working end-to-end through the new entry, freeze
THAT output as the golden baseline, and every later stage (C, D) must match it. Do NOT try to run
the old broken entry point. If fixing the splittingData arity is required just to get the
skeleton running, treat that as the smallest necessary behavior fix, flag it explicitly, and
commit it on its own.

Reference entry point (Stage A, R3): the GRID-SEARCH MAPE tuner —
esb_03_2026_tune_experiment/mape_tuner.py -> base_tuner.py -> tuner_utils.py
(GridSearchConfig.generate_configs + ProgressTracker checkpoint/resume). This, not the
keras-tuner RandomSearch path, is what the new entry must launch. The RandomSearch path
(crps_mme_runner.py + configs/tuner_mape.txt) is PARKED as a future feature — do not target it.

Branch: git checkout -b esb_refactor (off esb_dev). Work only on this branch; small focused
commits as you go (one logical change each), never bundle unrelated changes.

Build, under a new `esb/` package:
- config.py   — the Config schema = SINGLE SOURCE OF TRUTH. Typed fields with metadata
                (type, default, choices, help). It both (a) generates the argparse CLI and
                (b) is introspectable so a future GUI can render from it. Includes validation
                that FAILS LOUD at startup (valid loss/lead/structure; data path exists;
                output-unit consistency, e.g. CRPS needs >1 output unit).
- cli.py      — single entry point. Support: `python -m esb run --profile NAME`,
                `python -m esb run @profiles/NAME.txt`, bare flags with defaults filling in,
                `esb profiles` (list available profiles), and `--dry-run` (print the fully
                resolved config and exit without training).
- pipeline.py — run_experiment(config): the ONE orchestration loop over lead_times × rotations
                (use `rotation`, not `cycle` — Stage A R6) calling the stages in order.
- stages/*.py — read, features, clean, split, reshape, scale, model, train, evaluate, infer —
                each as a THIN WRAPPER that simply calls the existing function(s) in
                src/helper/utils_mse_crps.py / the grid-search tuner. Do not move logic in yet.
- io/results.py — writes predictions/model/metrics AND calls
                src/helper/run_provenance.write_provenance(result_dir, resolved_config)
                AUTOMATICALLY at the single write site. Provenance must be impossible to skip.
- profiles/   — named config files. MUST include a WORKING MAPE profile that launches the
                GRID-SEARCH MAPE tuner end-to-end (R3). Build this profile from the grid-search
                tuner's own config mechanism (tuner_utils.GridSearchConfig), NOT the broken
                configs/tuner_mape.txt (that file belongs to the parked RandomSearch path).
                For callback knobs (Stage A R5): wire --early_stop_patience, --lr_reducer_patience,
                --min_delta to the real callbacks with defaults = current literals (early-stop 25,
                lr-reducer 15, min_delta 0.001); drop the redundant --patience.

Hard constraints:
- ZERO behavior change vs the intended grid-search tuner behavior. Add nothing new except the
  entry layer + provenance call + the R5 callback wiring (which is byte-identical at defaults).
- Any abstraction beyond what's described here: propose it and wait for my yes.

Verification — FORWARD-CORRECTNESS (do this, report results):
1. There is no runnable old entry to diff against (R1). Instead: pick the smallest representative
   MAPE grid-search config (e.g. a 1-leadtime, 1-rotation, few-epochs debug profile), pin seeds,
   and run it end-to-end through the NEW entry until it produces results + run_provenance.json.
2. Freeze that run's outputs + a hash of the model.fit inputs (x/y arrays, shapes, hyperparams,
   compiled loss/metrics) into ./_golden_baseline/ (gitignored). THIS becomes the baseline that
   Stages C and D must reproduce byte-for-byte.
3. Re-run the same profile once more and assert the deterministic outputs + model.fit-input hash
   are IDENTICAL across the two runs (proves determinism before we lock the baseline). Report any
   difference instead of papering over it.

STOP for approval at two points: (a) after you present the Config schema + CLI design, before
implementing; (b) after the skeleton runs and the determinism/baseline check passes.
```

---

## STAGE C — The `scale` stage (first cohesive stage; reconciles two implementations)

```
Read docs/ESB_PIPELINE_DESIGN.md (sections 4–6) and the Stage B code on branch esb_refactor.

Goal: replace the thin `scale` wrapper with ONE real, cohesive scaling stage that reconciles the
two existing implementations. NOTE: this stage is the ONE place you may look at other branches.

Source material to reconcile:
- esb_dev_normalization (Ayesha, commit d718c6a): main-driver/config integration + debug
  cleanup. Use as the integration BASE.
- esb_dev_add_training_trace_restoring_grendal (Hector, commit f1965ad): the inference path —
  prepare_independent_year(scaler, column_map) + .joblib scaler persistence + Laguna Madre
  cross-dataset support. ADD this on top.
- Both fit StandardScaler on x_train ONLY (leakage-safe) — keep that.
- Also check: does esb_dev_normalization commit ccc31b7 ("fixed splitting data bug") duplicate
  38eba23 already on esb_dev, or is it a different fix? Report before merging.

Build esb/stages/scale.py owning: fit-on-train-only, transform train/val/test, persist scaler
(.joblib), and provide the inference transform (with column_map) for the infer stage. Wire a
`scale` option into the Config schema + add scaled profiles.

Verification:
- With scale=DISABLED: output must be IDENTICAL to Stage B (proves zero regression).
- With scale=ENABLED: new behavior — prove the scaler is fit on training data only (no leakage),
  is persisted, and the infer stage applies the SAME fitted scaler. Show the leakage check.

STOP for approval after presenting the reconciliation plan, and again after it passes verification.
```

---

## STAGE D — Migrate remaining stages into real cohesive modules (one at a time)

```
Read docs/ESB_PIPELINE_DESIGN.md (section 4 contracts) and the current esb_refactor branch.

Goal: one stage at a time, move the REAL logic out of src/helper/utils_mse_crps.py into the
cohesive stage module, behind its typed contract, replacing the thin wrapper. Stages remaining:
read, features, clean, split, reshape, model, train, evaluate, infer.

Rules:
- ONE stage per commit. After EACH migration, re-run the golden check from Stage B and confirm
  output is still IDENTICAL. If it diverges, stop and report — do not proceed.
- Preserve every quirk listed in the Stage A report (col_start, glob ordering, -999/NaN, full-set
  batch size, etc.).
- You MAY remove clearly dead/harmful code ONLY where it is behavior-neutral and you flag it:
  e.g. replace exit() in dataframe_checker with a raised exception; drop debug CSV dumps. Note
  each such change explicitly.
- No new abstraction layers without my approval.

Order suggestion: read → features → clean → split → reshape → evaluate → model → train → infer
(pure/deterministic stages first; training last).

STOP after each stage (or a small batch) for my review before continuing.
```

---

## OUT OF SCOPE (do not start without explicit approval)
- The **GUI** — it is the capstone, built LAST, only after the CLI + Config schema are frozen
  (see design doc §8). Not part of Stages A–D.
- **Remote/SLURM job submission** — Run is local-only by design.
- Any **research/model changes** beyond the scaling reconciliation in Stage C.
