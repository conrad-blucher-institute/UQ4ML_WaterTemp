# ESB Pipeline Refactor — Stage Prompts

Paste-ready prompts to drive the ESB pipeline redesign **one stage at a time**, each in a
fresh conversation. They implement the strangler-fig migration in `docs/ESB_PIPELINE_DESIGN.md`.

> **Status (2026-06-29).** Stage A ✅ (`docs/ESB_STAGE_A_REPORT.md`). Stage B ✅ — skeleton on
> branch `esb_refactor` (12 commits, report `docs/ESB_STAGE_B_REPORT.md`); golden fit-input digest
> frozen: **`d80ae5421e8ba0c69593777ac3857512609a16b962ccd7cd1cd7da2fbb63c9e7`** (unscaled path).
> Stage C is NEXT, on branch **`esb_refactor_post_stageC`** (already created off `esb_refactor`).
> **Do NOT merge into `esb_dev`** — Hector reviews merges manually later; stages stay stacked.
> Run env: Anaconda `python 3.10.9 / TF 2.12` with `PYTHONIOENCODING=utf-8` (workspace `python`
> has no TF; a pre-existing `→`-print breaks on cp1252 stdout).

## How to use this file
- **Execution order (Hector's plan, 2026-06-29): A → B → C → E → D → F.** The stage *labels* are
  fixed identifiers; only the *order* is reordered. Stage E (GUI + `--shard` sharding) now runs
  **after C and before D**, because Hector wants to launch sharded tuning (`--shard 1/20`) as soon
  as C+E finish, then do the internal migration (D) while the runs proceed. This is safe: the GUI
  is schema-driven and the Config schema is frozen at the end of C, so D (internal-only) doesn't
  move it.
- Run from the repo root. Each stage works on its OWN dedicated branch (Stage B = `esb_refactor`;
  Stage C = `esb_refactor_post_stageC`, already created off `esb_refactor`), **never** `esb_dev`
  directly. Branches stay **stacked** — do not merge into `esb_dev`; Hector reviews merges manually.
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
BRANCH: work on `esb_refactor_post_stageC` (ALREADY CREATED off esb_refactor — just confirm it is
checked out with `git branch --show-current`; do NOT create a new one and do NOT branch off esb_dev).
Do NOT merge into esb_dev — Hector reviews merges manually later; stages stay stacked on esb_refactor.
Small focused commits, one logical change each, Co-Authored-By trailer.

Read docs/ESB_PIPELINE_DESIGN.md — ESPECIALLY §6 which now holds the LOCKED Stage C decisions
(C1–C3) and two guardrails — plus docs/ESB_STAGE_B_REPORT.md and the Stage B esb/ code.

LOCKED DECISIONS (do not re-litigate; design doc §6):
- C1: scaling is OPT-IN, OFF BY DEFAULT (a `--scale` flag). Stage C stays behavior-preserving
  until --scale is passed; the Stage B golden digest d80ae54 must still hold with scale OFF.
- C2: integration BASE = origin/esb_dev_normalization. Inspection confirms it ALREADY contains
  BOTH halves — the fit-on-train-only StandardScaler (gated by a `scale` flag, returns an 11-tuple
  when on) AND prepare_independent_year(scaler=, column_map=) for cross-dataset (Laguna Madre)
  inference. So LIFT that code into the new stage; you are NOT merging two rival implementations.
  esb_dev_add_training_trace_restoring_grendal (has src/driver/mape_scaled_driver.py) is NOT the
  base — only consult it if something is missing from the normalization branch.
- C3: typed containers Arrays/ScaledArrays are APPROVED (Guardrail-3 satisfied). Build them so the
  scale stage is Arrays -> ScaledArrays + scaler, REPLACING the legacy 10-vs-11-tuple return.

GUARDRAILS (design doc §6):
- Leakage is now possible. fit() MUST be on x_train ONLY; add an explicit assert/test proving
  val/test (and the infer path) are transform-only.
- --scale changes the fit-input digest (arrays become standardized). The Stage B digest d80ae54
  covers the UNSCALED path only — freeze a SEPARATE second golden baseline for the scaled path;
  do NOT overwrite the first. (Tooling exists: `python -m esb._verify freeze|check`.)
- R1 (splittingData arity) is ALREADY FIXED on esb_refactor by passing independent_year='cycle'.
  Before lifting normalization-branch code, check that its splitting logic (e.g. commit ccc31b7
  "fixed splitting data bug") does not re-introduce or conflict with that fix — report before
  applying.

Build esb/stages/scale.py owning: fit-on-train-only, transform train/val/test, persist scaler
(.joblib) at the single io write site, and provide the inference transform (with column_map) for
the infer stage. Wire the `scale` option into the Config schema (single source of truth) + add
scaled profiles. Introduce Arrays/ScaledArrays (C3) as the stage contracts.

ALSO IN THIS STAGE — SEARCH-SPACE UPDATE (approved model change; see design doc §9).
This is a DELIBERATE, APPROVED change to the hyperparameter grid — the only model change allowed.
Keep it in its OWN commit(s), clearly labeled, NOT bundled with the scale reconciliation. Implement
in the model stage / the tuner's GridSearchConfig, and add each as a Config-schema field (single
source of truth, so CLI + future GUI pick them up automatically):
- neurons: [16, 32, 64, 128, 256]   (remove 100)
- dropout: tuned choice [0.0, 0.05, 0.1, 0.3], applied as a dropout layer after EACH hidden Dense
           (0.0 = off / a no-op layer)
- DROPOUT TYPE DEPENDS ON ACTIVATION: use keras AlphaDropout when activation == 'selu', and
  regular Dropout for relu / leaky_relu. Plain Dropout on SELU silently breaks self-normalization.
- keep activation [relu, selu, leaky_relu] and layers 1–3
- selection metric stays val_mae, BUT persist the FULL metric suite (mae, mae12, me, me12, mape, …)
  per config/rep so any of them can be inspected later in the viz suite.

Verification:
- SCALE RECONCILIATION (behavior-preserving): with scale=DISABLED AND dropout=0.0 on the Stage B
  baseline config, output must be IDENTICAL to the Stage B golden baseline (proves zero regression
  from the scale stage). With scale=ENABLED: prove the scaler is fit on training data only (no
  leakage), is persisted, and the infer stage applies the SAME fitted scaler. Show the leakage check.
- SEARCH-SPACE UPDATE (intended behavior change — validated on its own terms, NOT against the Stage
  B baseline): assert the generated grid contains the new dropout options, contains NO 100-neuron
  configs, that selu configs build an AlphaDropout layer (and relu/leaky_relu build Dropout), and
  that the results CSV now carries all metric columns.

STOP for approval after presenting (a) the scale reconciliation plan AND (b) the search-space plan,
before implementing; and again after each passes its verification above.
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

## STAGE E — GUI + multi-machine sharding  (EXECUTION: run AFTER Stage C, BEFORE Stage D)

```
Read docs/ESB_PIPELINE_DESIGN.md (§8 GUI, §9 search space, §10 sharding) and the post-Stage-C code.
Branch off the latest post-C branch (e.g. `git checkout -b esb_refactor_stageE` off the Stage C
branch); do NOT merge into esb_dev (stacked branches; Hector merges manually). Small focused commits.

This stage has TWO parts, each its own commit(s). Do PART 1 (sharding) FIRST — Hector wants to launch
`--shard 1/20` as soon as this stage lands.

PART 1 — MULTI-MACHINE SHARDING (CLI/pipeline feature; design doc §10)
- Add `--shard k/N` to the Config schema + CLI. The pipeline enumerates the FULL deterministic job
  list (lead_time × rotation × config × rep, stable sort); machine k runs only its slice. Default to
  CONTIGUOUS blocks (split the job list into N equal contiguous chunks, machine k = chunk k) for easy
  manual tracking/hand-out; document the choice. No flag (or `--shard 0/1`) = run everything.
- OUTPUT CONTRACT (so an offline SSD union-merge is trivial): self-contained, uniquely-named result
  folders whose path encodes leadtime/rotation/config/rep (model + scaler + run_provenance.json
  together). Each shard writes its OWN progress_shard_k.csv — NEVER one shared global CSV.
- BEHAVIOR-PRESERVING: with no shard flag, behavior is IDENTICAL to Stage C/D — the golden check must
  still pass. Sharding only SELECTS a subset of jobs; it never changes any job's result.
- Verification: assert `--shard 0/2` ∪ `--shard 1/2` = the full job set with ZERO overlap and ZERO
  gaps; `--dry-run` prints exactly which jobs a given shard would run.
- The cross-machine `esb aggregate` merge step stays in Stage F — running shards does not need it.

PART 2 — GUI (schema-driven; design doc §8)
- Lightweight Python web UI (Streamlit OR Gradio — pick one, justify briefly in your report). It MUST
  introspect the Config schema to render its form — NEVER hardcode the option list (single source of
  truth is the whole point).
- Three actions:
  (a) "Build config & stop" → writes a profiles/*.txt the CLI accepts (the dry-run artifact).
  (b) "Run locally" → shells out to `python -m esb run ...` on THIS machine.
  (c) "Open viz suite" → launches the existing viz on a chosen results folder.
- Surface the LOCAL-ONLY blurb (design doc §8): Run trains on THIS machine; cluster/SLURM is manual —
  build the config, submit it on the cluster yourself.
- May expose a `--shard k/N` field for a local run, but make clear multi-machine = run the CLI on each
  machine manually (the GUI does NOT orchestrate remote machines).
- Verification (prove single-source-of-truth): the GUI renders every Config field; a config it builds
  is accepted by the CLI unchanged; adding/removing a schema field changes the GUI form with NO
  GUI-code edit. Demonstrate that last point explicitly.

STOP for approval at: PART 1 sharding design (before coding); PART 1 verification; PART 2 GUI design
(tech choice + layout); PART 2 verification.
```

---

## OUT OF SCOPE (do not start without explicit approval)
- **Remote/SLURM job submission** — Run is local-only by design (manual on the cluster).
- Any **research/model changes** beyond the scaling reconciliation AND the approved search-space
  update (design doc §9) in Stage C. No *other* model changes without explicit approval.
- **Stage F (later):** the `esb aggregate` offline SSD-merge step, PNG/PDF share export, `esb new`
  config wizard, and other throughput/UX niceties.

> NOTE: the **GUI + `--shard k/N` sharding are now Stage E**, reordered to run after Stage C and
> before Stage D (see the execution-order note at the top). They are no longer out of scope.
