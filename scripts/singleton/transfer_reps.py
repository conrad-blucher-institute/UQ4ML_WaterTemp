#!/usr/bin/env python
"""Collect a combo set's .keras reps on the OTHER machine.

Run this ON the PC that holds the missing reps, FROM THE CAMPAIGN FOLDER --
the directory that contains keras_files/. It scans for every rep of each combo
you ask for, copies each .keras plus its _scaler.joblib / _history.json
sidecars into ./esb_reps_transfer/, and writes a manifest. Then zip that
folder and move it to the laptop.

    <campaign folder>/          <- cd here, run the script here
      keras_files/              <- scanned (recursively, so nested layouts work)
      esb_reps_transfer/        <- created here

Which combos
------------
Default is the 36 phase-3 combos embedded below. ``--combos <csv>`` reads them
from a file instead -- point it at ``top_20_combos.csv`` for the full top-20 set
(1680 combos: 21 lead times x 4 cycles x 20). Both that file's column spelling
and the tidy selection CSV's are accepted.

Standalone by design: no imports from `esb` or the tuner package, no pandas,
no repo required. Only the stdlib, so it runs anywhere Python 3.8+ exists --
copy this single file (plus the combos CSV, if you use one) over if that is
easier than syncing the repo.

USAGE
    cd /path/to/<campaign folder>
    python transfer_reps.py                     # DRY RUN (default)
    python transfer_reps.py --apply             # actually copy
    python transfer_reps.py --root D:/other     # scan somewhere else instead
    python transfer_reps.py --runs 4-10         # limit reps copied

    # the full top-20 set, reps 4-10 only:
    python transfer_reps.py --combos top_20_combos.csv --runs 4-10 --apply

By default it takes EVERY rep it finds (not just 4-10): this laptop already has
runs 1-3, and duplicates are cheap next to another cross-machine round trip.
Use --runs to narrow if space is tight -- and on the top-20 set you should, as
it is ~11,800 models rather than a couple hundred.
"""
import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# (leadtime, selection, cycle, neurons, activation, layers, dropout)
# Embedded on purpose: this script must run on a machine where the tidy CSV
# may not exist. Regenerate from results/phase2_selection_for_phase3_tidy.csv
# if the selection changes.
COMBOS = [
    (3, 1, 2, 256, 'selu', 1, 0),
    (3, 2, 2, 16, 'selu', 2, 0),
    (6, 1, 2, 256, 'selu', 2, 0),
    (6, 2, 0, 256, 'selu', 1, 0),
    (12, 1, 0, 256, 'selu', 1, 0),
    (12, 2, 0, 256, 'selu', 3, 0),
    (18, 1, 0, 256, 'selu', 2, 0),
    (18, 2, 0, 256, 'selu', 3, 0),
    (18, 3, 0, 256, 'relu', 2, 0),
    (24, 1, 0, 256, 'selu', 2, 0),
    (24, 2, 0, 256, 'relu', 2, 0),
    (30, 1, 0, 64, 'selu', 2, 0),
    (30, 2, 0, 256, 'selu', 1, 0),
    (36, 1, 0, 256, 'selu', 1, 0),
    (36, 2, 0, 128, 'relu', 2, 0),
    (42, 1, 0, 256, 'selu', 1, 0),
    (42, 2, 0, 32, 'selu', 2, 0),
    (48, 1, 0, 256, 'selu', 1, 0),
    (48, 1, 1, 256, 'selu', 1, 0),
    (48, 2, 0, 128, 'selu', 3, 0),
    (54, 1, 0, 128, 'selu', 1, 0),
    (54, 1, 1, 128, 'selu', 1, 0),
    (54, 2, 0, 256, 'selu', 1, 0),
    (60, 1, 0, 16, 'selu', 2, 0),
    (60, 2, 0, 256, 'selu', 1, 0),
    (66, 1, 0, 16, 'selu', 2, 0),
    (66, 2, 0, 256, 'selu', 1, 0),
    (66, 3, 0, 256, 'selu', 2, 0),
    (72, 1, 0, 256, 'selu', 2, 0),
    (72, 2, 0, 256, 'selu', 3, 0),
    (78, 1, 0, 256, 'selu', 3, 0),
    (78, 2, 0, 256, 'selu', 1, 0),
    (96, 1, 0, 256, 'selu', 3, 0),
    (96, 2, 0, 256, 'selu', 2, 0),
    (120, 1, 0, 256, 'selu', 3, 0),
    (120, 2, 0, 256, 'selu', 2, 0),
]


# Column spellings seen in the two CSVs that describe a combo set: the tidy
# phase-3 selection (leadtime/layers) and phase 2's top_20_combos.csv
# (lead_time/num_layers). Accepting both means neither file has to be reshaped
# before it can drive a transfer.
_ALIASES = {
    "leadtime": ("leadtime", "lead_time"),
    "cycle": ("cycle",),
    "neurons": ("neurons",),
    "activation": ("activation",),
    "layers": ("layers", "num_layers"),
    "dropout": ("dropout",),
    "selection": ("selection",),
}


def _pick(row, field):
    for name in _ALIASES[field]:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return None


def load_combos(path):
    """Read a combo set from CSV -> the same tuple shape as COMBOS.

    Why a file and not a bigger embedded list: the top-20 set is 1680 combos
    (21 lead times x 4 cycles x 20). Pasting that in here would make the script
    unreadable and would go stale the moment the selection is re-cut. The script
    stays stdlib-only and self-contained, so copying it plus one CSV to the other
    machine is still a two-file move.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")

    combos, seen, placeholders = [], set(), 0
    for i, row in enumerate(rows, start=2):
        # The tidy selection CSV carries "NO SELECTION YET" rows: a lead time
        # with every combo field blank. That is a real, expected state, not a
        # malformed file, so skip it and count it rather than dying on it.
        if all(_pick(row, f) is None
               for f in ("cycle", "neurons", "activation", "layers")):
            placeholders += 1
            continue
        try:
            lt = int(float(_pick(row, "leadtime")))
            cyc = int(float(_pick(row, "cycle")))
            neu = int(float(_pick(row, "neurons")))
            lay = int(float(_pick(row, "layers")))
            act = str(_pick(row, "activation")).strip().lower()
            drop = float(_pick(row, "dropout") or 0)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{path} line {i}: cannot read a combo from this row ({e}).\n"
                f"  columns present: {sorted(row)}\n"
                f"  need lead_time/leadtime, cycle, neurons, "
                f"num_layers/layers, activation, dropout"
            )
        sel_raw = _pick(row, "selection")
        sel = int(float(sel_raw)) if sel_raw is not None else 0
        key = (lt, cyc, neu, act, lay, drop)
        # top_20_combos.csv is one row per combo already, but a hand-edited file
        # can repeat one. Copying the same reps twice is harmless; reporting the
        # same combo twice in the matched/missing tally is not.
        if key in seen:
            continue
        seen.add(key)
        combos.append((lt, sel, cyc, neu, act, lay, drop))

    if placeholders:
        print(f"  note: skipped {placeholders} row(s) with no combo filled in")
    if not combos:
        raise ValueError(f"{path} named no usable combos")
    return combos


NAME_RE = re.compile(
    r"^(?P<model>[A-Za-z]+)_(?P<lt>\d+)h_cycle(?P<cycle>\d+)_(?P<act>[a-z_]+)"
    r"_(?P<layers>\d+)L_(?P<neurons>\d+)N(?:_d(?P<dropout>[\d.]+))?"
    r"_run(?P<run>\d+)$"
)
SIDECARS = ("_scaler.joblib", "_history.json")


def parse_runs(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def default_search_root(cwd):
    """Where to scan when --root is not given.

    Prefers ./keras_files (the campaign layout) but falls back to the whole
    CWD, so running from one level up -- or from a flattened dump of .keras
    files -- still works. The walk is recursive either way.
    """
    kf = cwd / "keras_files"
    return kf if kf.is_dir() else cwd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually copy (default is a dry run)")
    ap.add_argument("--root", action="append", default=[],
                    help="directory to search instead of ./keras_files (repeatable)")
    ap.add_argument("--out", default=None, help="output dir (default ./esb_reps_transfer)")
    ap.add_argument("--runs", default=None,
                    help="only these runs, e.g. '4-10' or '4,5,6' (default: all found)")
    ap.add_argument("--combos", default=None,
                    help="CSV naming the combos to collect (e.g. "
                         "top_20_combos.csv). Default is the 36 phase-3 combos "
                         "embedded in this file.")
    args = ap.parse_args()

    if args.combos:
        try:
            combo_set = load_combos(args.combos)
        except (OSError, ValueError) as e:
            print(f"ERROR reading --combos: {e}")
            return 1
        combo_source = f"{args.combos} ({len(combo_set)} combos)"
    else:
        combo_set = COMBOS
        combo_source = f"embedded phase-3 selection ({len(combo_set)} combos)"

    cwd = Path.cwd().resolve()
    out_dir = Path(args.out) if args.out else cwd / "esb_reps_transfer"
    want_runs = parse_runs(args.runs)

    roots = [Path(r) for r in args.root] or [default_search_root(cwd)]
    missing_roots = [r for r in roots if not r.exists()]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("ERROR: nothing to search.")
        for r in missing_roots:
            print(f"  does not exist: {r}")
        print(f"\nRun this from the campaign folder (the one containing keras_files/),")
        print(f"or pass --root <dir>. Current directory: {cwd}")
        return 1

    print(f"cwd       : {cwd}")
    print(f"searching : {', '.join(str(r) for r in roots)}")
    print(f"output    : {out_dir}")
    print(f"combos    : {combo_source}")
    print(f"runs      : {'all found' if not want_runs else sorted(want_runs)}")
    print(f"mode      : {'APPLY (copying)' if args.apply else 'DRY RUN (nothing copied)'}\n")

    # Index every .keras under the search roots, once.
    index = defaultdict(list)
    scanned = 0
    for r in roots:
        for dirpath, _dirnames, filenames in os.walk(r):
            for fn in filenames:
                if not fn.endswith(".keras"):
                    continue
                scanned += 1
                m = NAME_RE.match(fn[:-len(".keras")])
                if m:
                    g = m.groupdict()
                    key = (int(g["lt"]), int(g["cycle"]), int(g["neurons"]),
                           g["act"], int(g["layers"]), float(g["dropout"] or 0))
                    index[key].append((Path(dirpath) / fn, int(g["run"])))
    print(f"scanned {scanned} .keras files, {len(index)} distinct combos\n")

    # Loud, not silent: zero files almost always means "wrong directory", and
    # without this the run would look like a clean 36/36 MISSING report.
    if scanned == 0:
        print("ERROR: no .keras files found anywhere under the search root(s).")
        print("       You are probably not in the campaign folder.")
        print(f"       Expected a keras_files/ directory under: {cwd}")
        print("       Fix: cd to the folder containing keras_files/, or pass --root <dir>.")
        return 1
    if not index:
        print(f"ERROR: found {scanned} .keras files but NONE matched the expected")
        print("       naming pattern, so no combo can be resolved. Example expected name:")
        print("         MAPE_18h_cycle0_selu_2L_256N_d0.0_run3.keras")
        print("       Sample of what was actually found:")
        for r in roots:
            for dirpath, _dn, fns in os.walk(r):
                for fn in fns:
                    if fn.endswith(".keras"):
                        print(f"         {fn}")
                        break
                break
        return 1

    manifest, missing, total_files = [], [], 0
    for lt, sel, cyc, neu, act, lay, drop in combo_set:
        key = (lt, cyc, neu, act, lay, float(drop))
        found = sorted(index.get(key, []), key=lambda t: t[1])
        if want_runs:
            found = [f for f in found if f[1] in want_runs]
        label = f"LT{lt:>3} sel{sel} c{cyc} {lay}L {neu}N {act} d{drop}"
        if not found:
            print(f"  MISSING  {label}")
            missing.append(label)
            continue
        runs = [r for _, r in found]
        print(f"  {label}: {len(found)} reps runs={runs}")
        for src, run in found:
            targets = [src] + [src.with_name(src.stem + s) for s in SIDECARS]
            for t in targets:
                if not t.exists():
                    continue
                total_files += 1
                if args.apply:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    dst = out_dir / t.name
                    if not dst.exists():
                        shutil.copy2(t, dst)
            manifest.append({"leadtime": lt, "selection": sel, "cycle": cyc,
                             "neurons": neu, "layers": lay, "activation": act,
                             "dropout": drop, "run_num": run,
                             "model_name": src.stem, "source_path": str(src)})

    print(f"\n{'=' * 70}")
    print(f"combos matched : {len(combo_set) - len(missing)} / {len(combo_set)}")
    print(f"reps found     : {len(manifest)}")
    print(f"files to copy  : {total_files}  (.keras + sidecars)")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for m in missing:
            print("   " + m)

    if args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "transfer_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()) if manifest else
                               ["leadtime", "selection", "cycle", "neurons", "layers",
                                "activation", "dropout", "run_num", "model_name", "source_path"])
            w.writeheader()
            w.writerows(manifest)
        print(f"\nWROTE {out_dir}  (+ transfer_manifest.csv)")
        print("Next: zip that folder and move it to the laptop.")
    else:
        print("\nDry run only. Re-run with --apply to copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
