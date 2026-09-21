"""Phase-5 retrain: the chosen architectures, on every year we have.

Phases 1-4 selected an architecture per lead time. This module does NOT select
anything -- it retrains those picks with the previously held-out data folded in,
so the operational models have seen as much history as possible.

    python -m esb.retrain_final --help

WHAT MOVES WHERE (cycle 1, verified by running splittingData's rotation):

    esb_2023_2024   train        ->  train
    esb_2024_2025   train        ->  train
    esb_2022_2023   test         ->  TRAIN   (folded in)
    esb_2020_2021   independent  ->  TRAIN   (folded in, Winter Storm Uri)
    esb_2021_2022   val          ->  val     (early stopping only)
    esb_2025_2026   -            ->  holdout (not yet received)

THE ARRAYS ARE CONCATENATED, NOT RE-SPLIT. preparingData() and
prepare_independent_year() are used unchanged. That is safe because reshaping()
is pure column slicing (``iloc[:, col_start:-1].values``) with no interaction
between rows, and prepare_independent_year() slices with the identical
col_start -- so concatenating the numpy arrays afterwards is identical to
concatenating the dataframes beforehand, without forking splittingData().

ORDER MATTERS: concatenate FIRST, then let the scale stage fit. fit_scaler()
carries the leakage assert and must see the array actually trained on. Fitting
before the concatenation would put 2021 on a different scale than the rest.

METRICS FROM THIS PHASE ARE IN-SAMPLE. There is no held-out test set here, so
these numbers are NOT comparable to phase-2/3 results -- a lower MAE partly just
means the model has seen the data. Each model's metadata records the year->role
map so this is checkable rather than remembered.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Console output is UTF-8 regardless of redirection; see esb/_encoding.py.
# Must run before anything prints.
from esb._encoding import force_utf8
force_utf8()

# The tuner modules use bare imports (`from base_tuner import ...`), so both the
# repo root and the tune-experiment dir must be importable. These inserts also
# propagate to spawned workers (multiprocessing copies the parent's sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUNE_DIR = _REPO_ROOT / "esb_03_2026_tune_experiment"
for _p in (_TUNE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_SELECTIONS = _REPO_ROOT / "results" / "phase5_final" / "final_selections_phase5.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "results" / "phase5_final"
DEFAULT_DATA = _REPO_ROOT / "data" / "ESB_datasets"

# The cycle-1 year->role map, recorded into every model's metadata. Derived from
# splittingData's rotation over sorted(data/ESB_datasets/*.csv) with csv_files[0]
# held out; asserted against the real rotation by _assert_year_roles() below.
CYCLE1_YEAR_ROLES = {
    "esb_2023_2024": "train",
    "esb_2024_2025": "train",
    "esb_2022_2023": "train",   # was test in phases 1-4
    "esb_2020_2021": "train",   # was the independent (Uri) year
    "esb_2021_2022": "val",
}

# Column spellings accepted in the selections CSV, so a sheet exported from
# Excel or copied from the phase-2 tidy file both load without hand-editing.
_ALIASES = {
    "leadtime": "lead_time", "lead_time": "lead_time", "lt": "lead_time",
    "cycle": "cycle", "rotation": "cycle",
    "neurons": "neurons", "n": "neurons",
    "layers": "num_layers", "num_layers": "num_layers", "l": "num_layers",
    "activation": "activation",
    "dropout": "dropout",
}


def derive_year_roles(data_path: Path, cycle: int) -> dict:
    """Return the year->role map for `cycle`, addressed by NAME.

    Phase 5 folds the old test year and the old independent year into training;
    only the validation year (which drives early stopping) rotates with cycle.

    The rotation comes from ESB_ROTATION_YEARS -- the same constant
    readingData() uses -- NOT from a positional glob. It used to be
    sorted(*.csv) with [0] as independent and [1:] as the rotation, so a 6th
    file in the folder shifted every year into the wrong role: once
    esb_2025_2026.csv landed there, cycle 0 made the one true holdout the
    VALIDATION year. Naming the years is what makes an unrelated file inert.

    Pure filename arithmetic -- no pandas, no data read.
    """
    # esb_years, not utils_mse_crps: that module imports TensorFlow at line 1
    # and --dry-run promises not to.
    from src.helper.esb_years import ESB_ROTATION_YEARS, ESB_INDEPENDENT_YEARS

    present = {q.stem for q in Path(data_path).glob("*.csv")}
    required = set(ESB_ROTATION_YEARS) | set(ESB_INDEPENDENT_YEARS)
    missing = sorted(required - present)
    if missing:
        raise SystemExit(
            f"year CSV(s) missing from {data_path}: {missing}\n"
            f"  present: {sorted(present)}"
        )
    if not 0 <= cycle < len(ESB_ROTATION_YEARS):
        raise SystemExit(
            f"--cycle must be 0..{len(ESB_ROTATION_YEARS) - 1}, got {cycle}"
        )

    independent = ESB_INDEPENDENT_YEARS[0]
    year_list = list(ESB_ROTATION_YEARS)
    for j in range(cycle + 1):
        for i in range(len(year_list)):
            if i > 0:
                year_list = [year_list[-1]] + year_list[:-1]
        if j == cycle:
            break
    n = len(year_list)
    derived = {y: "train" for y in year_list[: n - 2]}
    derived[year_list[n - 2]] = "val"
    derived[year_list[n - 1]] = "train"   # old test, folded in
    derived[independent] = "train"        # old independent, folded in

    # Cycle 1 is the phase-5 baseline and its map is written down. Keep checking
    # it: if this ever stops matching, the rotation logic moved under us and
    # every other cycle derived here is suspect too.
    if cycle == 1 and derived != CYCLE1_YEAR_ROLES:
        raise SystemExit(
            "year->role map disagrees with splittingData's actual rotation.\n"
            f"  constant: {CYCLE1_YEAR_ROLES}\n"
            f"  derived : {derived}\n"
            "The data folder or the split logic changed. Fix CYCLE1_YEAR_ROLES "
            "(and re-check the phase5 README) before training."
        )
    return derived


def load_selections(csv_path: Path, cycle: int) -> list[dict]:
    """Read the hand-edited selections sheet into normalized combo dicts.

    Normalizes two things that are silently destructive if passed through raw:
      * activation is lowercased -- Keras' activation lookup is case-sensitive,
        and mape_tuner picks AlphaDropout on ``activation == 'selu'``, so 'SELU'
        would take the wrong branch.
      * dropout becomes a float -- job_base_name formats it straight into the
        filename, so '0' yields ``_d0`` where every other campaign has ``_d0.0``.
        That mismatch is invisible until something tries to match filenames.
    """
    with open(csv_path, newline="") as fh:
        raw_rows = list(csv.DictReader(fh))
    if not raw_rows:
        raise SystemExit(f"{csv_path} has no rows")

    combos, problems = [], []
    for i, raw in enumerate(raw_rows, start=2):   # start=2 -> spreadsheet row
        row = {}
        for key, val in raw.items():
            if key is None:
                continue
            canon = _ALIASES.get(key.strip().lower())
            if canon:
                row[canon] = (val or "").strip()
        if not any(row.get(k) for k in ("lead_time", "neurons", "num_layers")):
            continue                                  # fully blank spacer row
        missing = [k for k in ("lead_time", "neurons", "num_layers", "activation")
                   if not row.get(k)]
        if missing:
            problems.append(f"  row {i}: missing {', '.join(missing)}")
            continue
        row_cycle = int(row["cycle"]) if row.get("cycle") else cycle
        if row_cycle != cycle:
            problems.append(
                f"  row {i} (LT {row['lead_time']}): cycle={row_cycle} but this "
                f"run is cycle {cycle} -- fix the sheet or pass --cycle {row_cycle}"
            )
            continue
        combos.append({
            # 'model_type' and 'run_num' are normally injected by
            # GridSearchConfig.generate_configs; an explicit config list must
            # supply them itself. base_tuner overwrites run_num per repetition.
            "model_type": "MAPE",
            "run_num": 0,
            "lead_time": int(row["lead_time"]),
            "cycle": cycle,
            "neurons": int(row["neurons"]),
            "num_layers": int(row["num_layers"]),
            "activation": row["activation"].lower(),
            "dropout": float(row.get("dropout") or 0.0),
        })

    if problems:
        raise SystemExit(
            f"{csv_path} has {len(problems)} unusable row(s):\n" + "\n".join(problems)
        )
    seen = [c["lead_time"] for c in combos]
    dupes = sorted({lt for lt in seen if seen.count(lt) > 1})
    if dupes:
        raise SystemExit(
            f"lead time(s) {dupes} appear more than once in {csv_path}. "
            "Phase 5 trains ONE architecture per lead time."
        )
    return sorted(combos, key=lambda c: c["lead_time"])


class _ExplicitGrid:
    """Duck-type of GridSearchConfig carrying an explicit config list.

    The grid tuners take a cartesian product; phase 5 needs one specific
    architecture per lead time, which no product expresses. base_tuner already
    accepts an externally supplied ``grid_config``, so this fits the existing
    seam rather than adding one.
    """

    def __init__(self, configs: list[dict]):
        self._configs = configs

    def generate_configs(self) -> list[dict]:
        return [dict(c) for c in self._configs]

    def count_configs(self) -> int:
        return len(self._configs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m esb.retrain_final",
        description="Phase 5: retrain the chosen architectures on all available data.",
    )
    ap.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS,
                    help="hand-edited sheet, one row per lead time (default: %(default)s)")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT,
                    help="campaign folder (default: %(default)s)")
    ap.add_argument("--data_path", type=Path, default=DEFAULT_DATA,
                    help="folder of yearly ESB CSVs (default: %(default)s)")
    ap.add_argument("--reps", type=int, default=5,
                    help="models per lead time; saved as run_num 1..N (default: %(default)s)")
    ap.add_argument("--cycle", type=int, default=1,
                    help="rotation. Phase 5 is cycle 1 per the review (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=200000,
                    help="max epochs; early stopping ends runs long before this (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel workers (default: %(default)s)")
    ap.add_argument("--only-leadtimes", type=str, default=None,
                    help="comma-separated subset, e.g. 3,6,12 -- shard knob, lead times are independent")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + print the plan, import no TensorFlow, train nothing")
    args = ap.parse_args(argv)

    if not args.selections.is_file():
        raise SystemExit(f"selections sheet not found: {args.selections}")
    if args.reps < 1:
        raise SystemExit(f"--reps must be >= 1, got {args.reps}")

    year_roles = derive_year_roles(args.data_path, args.cycle)
    combos = load_selections(args.selections, args.cycle)

    if args.only_leadtimes:
        want = {int(x) for x in args.only_leadtimes.split(",") if x.strip()}
        have = {c["lead_time"] for c in combos}
        unknown = want - have
        if unknown:
            # Loud, not silent: an empty shard is indistinguishable from a
            # crashed one once the run is over.
            raise SystemExit(
                f"--only-leadtimes names lead time(s) not in the sheet: "
                f"{sorted(unknown)} (available: {sorted(have)})"
            )
        combos = [c for c in combos if c["lead_time"] in want]

    train_years = sorted(y for y, r in year_roles.items() if r == "train")
    val_years = sorted(y for y, r in year_roles.items() if r == "val")

    print("=" * 78)
    print("PHASE 5 -- final operational models, trained on all available data")
    print("=" * 78)
    print(f"  selections : {args.selections}")
    print(f"  output     : {args.output_dir}")
    print(f"  cycle      : {args.cycle}")
    print(f"  reps       : {args.reps}  (saved as run_num 1..{args.reps})")
    print(f"  epochs     : {args.epochs}   workers: {args.workers}")
    print(f"  train on   : {', '.join(train_years)}")
    print(f"  val (ES)   : {', '.join(val_years)}")
    print(f"  lead times : {[c['lead_time'] for c in combos]}")
    print(f"  models     : {len(combos)} combos x {args.reps} reps "
          f"= {len(combos) * args.reps}")
    print()
    print("  NOTE: no held-out test set in this phase. Metrics reported here are")
    print("        IN-SAMPLE and are not comparable to phase-2/3 numbers.")
    print("=" * 78)
    for c in combos:
        print(f"    LT {c['lead_time']:>3}  {c['neurons']}N  {c['num_layers']}L  "
              f"{c['activation']}  d{c['dropout']}")
    print()

    if args.dry_run:
        print("[dry-run] validated; nothing trained.")
        return 0

    from esb.config import Config, add_run_arguments
    from esb.pipeline import _build_config_overrides

    # Build a Config through the SAME schema the CLI uses, so phase 5 inherits
    # every default (learning_rate, output_units, callback patience, ...) and
    # cannot drift from `esb run`. Only the handful of fields phase 5 actually
    # pins are passed; the rest come from the schema defaults.
    _schema_parser = argparse.ArgumentParser(add_help=False)
    add_run_arguments(_schema_parser)
    ns = _schema_parser.parse_args([
        "--loss", "mape",
        "--scale",
        "--data_path", str(args.data_path),
        "--epochs", str(args.epochs),
        "--workers", str(args.workers),
        "--output_dir", str(args.output_dir),
    ])
    config = Config.from_namespace(ns).validate()
    overrides = _build_config_overrides(config)
    # Phase 5 assembles its own training set inside Phase5Tuner._prepare_arrays;
    # 'cycle' keeps preparingData's rotation as the source of the base split.
    overrides["independent_year"] = "cycle"
    # THIS cycle's roles. Was dict(CYCLE1_YEAR_ROLES), which stamped cycle-1
    # provenance onto every model no matter what --cycle said.
    overrides["dataset_roles"] = dict(year_roles)
    overrides["training_datasets"] = train_years
    overrides["phase"] = "phase5_final"
    overrides["metrics_are_in_sample"] = True

    keras_save_dir = args.output_dir / "keras_files"
    keras_save_dir.mkdir(parents=True, exist_ok=True)

    # Imported here, not at module scope: this pulls in TensorFlow, and
    # --dry-run must stay importable on a machine without it.
    from esb._phase5_tuner import Phase5Tuner

    TunerClass = Phase5Tuner
    tuner = TunerClass(
        args.output_dir,
        max_workers=args.workers,
        keras_save_dir=keras_save_dir,
        verbose=config.verbose,
    )
    tuner.grid_config = _ExplicitGrid(combos)
    tuner.config_overrides = overrides
    tuner.default_epochs = args.epochs

    for rep in range(1, args.reps + 1):
        print(f"\n[phase5] === rep {rep}/{args.reps} ===")
        tuner.run_tuning(run_num=rep)

    # base_tuner catches per-job exceptions and keeps going, so a run in which
    # EVERY worker died (a BrokenProcessPool, an import error in the child)
    # still returns normally. Count the artifacts actually on disk and fail
    # loudly if they are missing -- "done" must mean done.
    expected = len(combos) * args.reps
    produced = len(list(keras_save_dir.glob("*.keras")))
    print(f"\n[phase5] models on disk: {produced}/{expected} in {keras_save_dir}")
    if produced < expected:
        print(
            f"[phase5] INCOMPLETE -- {expected - produced} model(s) missing.\n"
            f"[phase5] Check the traceback above and "
            f"{args.output_dir / 'mape_progress.csv'} for per-job status.\n"
            f"[phase5] Re-running skips finished jobs, so it resumes where it stopped.",
            file=sys.stderr,
        )
        return 1

    print(f"[phase5] done -> {args.output_dir}")
    # --datasets val, NOT the 2021,test default: phase 5 trained on test and
    # 2021, so wide CSVs for those are in-sample and cannot show an outlier rep.
    # val is the only year phase 5 withheld (it drove early stopping, so it is
    # mildly optimistic, but it is the only honest signal until 2025-26 lands).
    print("[phase5] next: python -m esb.predict "
          f"{args.output_dir} --split-by-leadtime --datasets val "
          f"--max-models {expected + 10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
