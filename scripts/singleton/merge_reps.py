#!/usr/bin/env python
"""Ingest transferred reps into the 21lt campaign and rebuild its manifest.

Two jobs, either of which can run alone:

  1. MERGE  (--from <dir>): copy .keras + sidecars from a transfer folder into
     results/phase3_selected/keras_files/, skipping anything already
     present. Content-checked, so a re-run is a no-op rather than a corruption.

  2. REPORT (always): recompute combo/rep coverage by SCANNING THE FOLDER and
     joining the tidy selection CSV, then print it.

The manifest is DERIVED, so it is NOT written to disk by default -- a stale
index file is worse than no index file. Pass --write-manifest if some tool
genuinely needs it as a CSV. Once *_metadata.json lands this computation moves
into load_campaign() and the filename regex here goes away.

Merge BEFORE wrapping
---------------------
Run this on RAW ``.keras`` files, not on ``_rawinput`` wrapped ones. Two reasons:
the filename regex here anchors on ``_run<N>`` and does not match a wrapped
name, and more importantly the byte-comparison below is only meaningful on
originals. Wrapping re-serializes the archive, so two wraps of the identical
inner model can differ byte-wise from zip ordering alone -- spurious conflicts
that would mask the real ones this exists to catch.

USAGE
    python scripts/merge_reps.py                          # rebuild manifest only
    python scripts/merge_reps.py --from ../esb_reps_transfer          # dry run
    python scripts/merge_reps.py --from ../esb_reps_transfer --apply  # do it
    python scripts/merge_reps.py --campaign results/phase4_top20 \
        --from ../esb_reps_transfer                       # a different campaign
"""
import argparse
import csv
import filecmp
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Three levels: singleton/ -> scripts/ -> repo root. This had only two, which
# resolved RESULTS to scripts/results and silently pointed every default at a
# folder that does not exist; it broke when the script moved into singleton/.
# Matches build_phase1.py and consolidate_progress.py.
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

# Defaults target the phase-3 campaign this script was written for; --campaign
# rebinds all of them together (see set_campaign). They are module-level rather
# than threaded through every function because do_merge/rebuild_manifest already
# read them as globals, and rebinding once in main() is a smaller, more
# reviewable change than adding a path argument to each.
CAMPAIGN = RESULTS / "phase3_selected"
KERAS_DIR = CAMPAIGN / "keras_files"
TIDY = RESULTS / "phase2_selection_for_phase3_tidy.csv"
MANIFEST = CAMPAIGN / "selection_manifest.csv"

NAME_RE = re.compile(
    r"^(?P<model>[A-Za-z]+)_(?P<lt>\d+)h_cycle(?P<cycle>\d+)_(?P<act>[a-z_]+)"
    r"_(?P<layers>\d+)L_(?P<neurons>\d+)N(?:_d(?P<dropout>[\d.]+))?"
    r"_run(?P<run>\d+)$"
)
SIDECARS = ("_scaler.joblib", "_history.json", "_metadata.json")

MANIFEST_COLS = ["leadtime", "selection", "cycle", "neurons", "layers", "activation",
                 "dropout", "run_num", "modeler_favorite", "notes", "model_name",
                 "source", "has_scaler", "has_history"]


def combo_key(lt, cycle, neurons, layers, act, dropout):
    """Identity of a hyperparameter combo. Rep (run_num) is deliberately NOT
    part of it -- that is what fans one combo out to N manifest rows."""
    return (int(lt), int(cycle), int(neurons), int(layers),
            str(act).strip().lower(), float(dropout or 0))


def load_tidy():
    """combo_key -> human columns. Rows without a selection are skipped."""
    out = {}
    if not TIDY.exists():
        print(f"WARNING: no tidy CSV at {TIDY}; manifest will have no human columns.")
        return out
    with TIDY.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("selection") or "").strip():
                continue
            k = combo_key(r["leadtime"], r["cycle"], r["neurons"],
                          r["layers"], r["activation"], r["dropout"])
            fav = (r.get("modeler_favorite") or "").strip().lower()
            out[k] = {
                "selection": r["selection"].strip(),
                # Normalize to a real boolean string; blank means false, and the
                # placeholder text used on unselected rows must never read true.
                "modeler_favorite": "true" if fav in ("true", "1", "yes", "y") else "false",
                "notes": (r.get("notes") or "").strip(),
            }
    return out


def scan(dirpath):
    """dirpath -> {model_name: (path, parsed dict)} for parseable .keras."""
    found = {}
    if not dirpath.is_dir():
        return found
    for p in sorted(dirpath.rglob("*.keras")):
        m = NAME_RE.match(p.stem)
        if m:
            found[p.stem] = (p, m.groupdict())
    return found


PROVENANCE = CAMPAIGN / "provenance.csv"


def set_campaign(folder):
    """Point every campaign-relative path at ``folder``.

    The selection CSV is looked for INSIDE the campaign first, falling back to
    the results root, because the reorganisation moved it into
    ``phase3_selected/`` while the original default still pointed at the root.
    A campaign without one (phase 4 has no hand-edited selection) is fine --
    load_tidy() already warns and returns no human columns.
    """
    global CAMPAIGN, KERAS_DIR, TIDY, MANIFEST, PROVENANCE
    CAMPAIGN = Path(folder).resolve()
    KERAS_DIR = CAMPAIGN / "keras_files"
    MANIFEST = CAMPAIGN / "selection_manifest.csv"
    PROVENANCE = CAMPAIGN / "provenance.csv"
    name = "phase2_selection_for_phase3_tidy.csv"
    local = CAMPAIGN / name
    TIDY = local if local.exists() else RESULTS / name


def load_provenance():
    if not PROVENANCE.exists():
        return {}
    with PROVENANCE.open(newline="", encoding="utf-8") as fh:
        return {r["model_name"]: r["source"] for r in csv.DictReader(fh)}


def save_provenance(mapping):
    with PROVENANCE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model_name", "source"])
        w.writeheader()
        for k in sorted(mapping):
            w.writerow({"model_name": k, "source": mapping[k]})


def do_merge(src_dir, apply, on_conflict="stop"):
    src = scan(src_dir)
    if not src:
        print(f"ERROR: no parseable .keras under {src_dir}")
        return None
    existing = scan(KERAS_DIR)
    print(f"transfer folder : {len(src)} models")
    print(f"campaign folder : {len(existing)} models already present\n")

    new, dupe, conflict = [], [], []
    for name, (p, _g) in src.items():
        dst = KERAS_DIR / p.name
        if not dst.exists():
            new.append((p, dst))
        elif filecmp.cmp(p, dst, shallow=True):
            dupe.append(name)
        else:
            # Same name, different bytes -- do not guess which is authoritative.
            conflict.append(name)

    print(f"  new      : {len(new)}")
    print(f"  duplicate: {len(dupe)}  (already present, identical -- skipped)")
    print(f"  CONFLICT : {len(conflict)}  (same name, DIFFERENT content)")
    for c in conflict[:10]:
        print(f"      {c}")
    if conflict and on_conflict == "stop":
        print("\nERROR: refusing to overwrite. Same filename with different bytes means")
        print("       two different trainings share an identity.")
        print("       Choose explicitly: --on-conflict keep (existing wins)")
        print("                          --on-conflict take (transfer wins)")
        return None
    if conflict:
        if on_conflict == "keep":
            print(f"\n  --on-conflict keep : leaving {len(conflict)} existing files untouched")
        else:
            print(f"\n  --on-conflict take : OVERWRITING {len(conflict)} existing files")
            for name in conflict:
                p = src[name][0]
                for s in [p] + [p.with_name(p.stem + x) for x in SIDECARS]:
                    if s.exists() and apply:
                        shutil.copy2(s, KERAS_DIR / s.name)
            new.extend((src[n][0], KERAS_DIR / src[n][0].name) for n in conflict)

    prov = load_provenance()
    tag = src_dir.name
    copied = 0
    for p, dst in new:
        prov[p.stem] = tag
        srcs = [p] + [p.with_name(p.stem + s) for s in SIDECARS]
        for s in srcs:
            if not s.exists():
                continue
            d = KERAS_DIR / s.name
            if d.exists():
                continue
            if apply:
                KERAS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
            copied += 1
    print(f"\n  files {'copied' if apply else 'to copy'}: {copied}")
    if apply:
        save_provenance(prov)
        print(f"  provenance -> {PROVENANCE.name}")
    return copied


def rebuild_manifest(apply, write_manifest=False):
    tidy = load_tidy()
    prov = load_provenance()
    models = scan(KERAS_DIR)
    if not models:
        print(f"ERROR: no .keras found under {KERAS_DIR}")
        return None

    rows, orphans = [], []
    per_combo = defaultdict(list)
    for name, (p, g) in models.items():
        k = combo_key(g["lt"], g["cycle"], g["neurons"], g["layers"],
                      g["act"], g["dropout"])
        human = tidy.get(k)
        if human is None:
            orphans.append(name)
            continue
        per_combo[k].append(int(g["run"]))
        rows.append({
            "leadtime": int(g["lt"]), "selection": human["selection"],
            "cycle": int(g["cycle"]), "neurons": int(g["neurons"]),
            "layers": int(g["layers"]), "activation": g["act"],
            "dropout": float(g["dropout"] or 0), "run_num": int(g["run"]),
            "modeler_favorite": human["modeler_favorite"], "notes": human["notes"],
            "model_name": name, "source": prov.get(name, "original"),
            "has_scaler": p.with_name(name + "_scaler.joblib").exists(),
            "has_history": p.with_name(name + "_history.json").exists(),
        })

    rows.sort(key=lambda r: (r["leadtime"], r["selection"], r["cycle"], r["run_num"]))

    if apply and write_manifest:
        with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
            w.writeheader()
            w.writerows(rows)

    # Report: rep coverage per combo is the number that decides the TS plan.
    print(f"\n{'=' * 72}")
    print(f"manifest rows (one per rep) : {len(rows)}")
    print(f"distinct combos             : {len(per_combo)}")
    print(f"missing from tidy CSV       : {len(orphans)}")
    for o in orphans[:5]:
        print(f"    orphan: {o}")

    counts = defaultdict(list)
    for k, runs in per_combo.items():
        counts[len(runs)].append(k)
    print("\nrep coverage:")
    for n in sorted(counts, reverse=True):
        print(f"  {n:>2} reps : {len(counts[n]):>3} combos")

    no_scaler = [r["model_name"] for r in rows if not r["has_scaler"]]
    if no_scaler:
        print(f"\nWARNING: {len(no_scaler)} models have NO _scaler.joblib.")
        print("         On a scaled campaign these silently predict garbage.")
        for m in no_scaler[:5]:
            print(f"    {m}")

    favs = {r["model_name"] for r in rows if r["modeler_favorite"] == "true"}
    print(f"\nmodeler_favorite reps: {len(favs)}")
    if write_manifest:
        print(f"{'WROTE' if apply else 'DRY RUN, would write'}: {MANIFEST}")
    else:
        print("manifest: not written (derived on demand; --write-manifest to force)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=None,
                    help="transfer folder to ingest (default: rebuild manifest only)")
    ap.add_argument("--apply", action="store_true", help="write changes (default dry run)")
    ap.add_argument("--write-manifest", action="store_true",
                    help="also write selection_manifest.csv (derived; off by default)")
    ap.add_argument("--on-conflict", choices=["stop", "keep", "take"], default="stop",
                    help="same name + different bytes: stop (default), keep existing, or take incoming")
    ap.add_argument("--campaign", default=None,
                    help="campaign folder to merge INTO (default: "
                         "results/phase3_selected)")
    args = ap.parse_args()

    if args.campaign:
        set_campaign(args.campaign)
        if not CAMPAIGN.is_dir():
            print(f"ERROR: no such campaign folder: {CAMPAIGN}")
            return 1

    print(f"campaign : {CAMPAIGN}")
    print(f"mode     : {'APPLY' if args.apply else 'DRY RUN'}\n")

    if args.src:
        if do_merge(Path(args.src).resolve(), args.apply, args.on_conflict) is None:
            return 1
    rebuild_manifest(args.apply, args.write_manifest)
    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
