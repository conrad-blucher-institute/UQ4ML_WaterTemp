#!/usr/bin/env python
"""Copy the two phase-1 grid campaigns into one phase1_grid_21lt folder.

The two campaigns partition the 21 lead times with no overlap:
    esb_tuner_scaled       LT 12, 48, 96, 120       13,209 models
    esb_tuner_scaled_17lt  LT 3, 6, 18 ... 114      36,720 models

Sources are LEFT INTACT (copy, not move). Any filename collision is reported
and skipped rather than overwritten -- a collision would mean the "no overlap"
assumption is wrong, which is worth knowing loudly.

    python scripts/singleton/build_phase1.py --apply
"""
import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
DST = RESULTS / "phase1_grid_21lt"
SOURCES = ["esb_tuner_scaled", "esb_tuner_scaled_17lt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    kdst = DST / "keras_files"
    if args.apply:
        kdst.mkdir(parents=True, exist_ok=True)

    seen, collisions, copied, total_bytes = {}, [], 0, 0
    t0 = time.time()

    for camp in SOURCES:
        src = RESULTS / camp / "keras_files"
        if not src.is_dir():
            print(f"ERROR: missing {src}")
            return 1
        n = 0
        with os.scandir(src) as it:
            for e in it:
                if not e.is_file():
                    continue
                if e.name in seen:
                    collisions.append((e.name, seen[e.name], camp))
                    continue
                seen[e.name] = camp
                n += 1
                total_bytes += e.stat().st_size
                if args.apply:
                    d = kdst / e.name
                    if not d.exists():
                        shutil.copy2(e.path, d)
                        copied += 1
                        if copied % 5000 == 0:
                            print(f"   ... {copied} files ({time.time()-t0:.0f}s)", flush=True)
        print(f"{camp}: {n} files", flush=True)

        # progress shards, prefixed so their origin survives
        for f in (RESULTS / camp).glob("mape_progress_*.csv"):
            if args.apply:
                shutil.copy2(f, DST / f"{camp}__{f.name}")

    print(f"\nunique files : {len(seen)}")
    print(f"total size   : {total_bytes/1e9:.1f} GB")
    print(f"collisions   : {len(collisions)}")
    for c in collisions[:10]:
        print(f"   {c[0]}: in both {c[1]} and {c[2]}")

    if args.apply:
        with (DST / "source_campaigns.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file_name", "source_campaign"])
            for k in sorted(seen):
                w.writerow([k, seen[k]])
        print(f"\ncopied {copied} files in {time.time()-t0:.0f}s -> {DST}")
    else:
        print("\nDRY RUN. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
