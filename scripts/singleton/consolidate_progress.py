#!/usr/bin/env python
"""Fold the 58 copied progress shards into one progress_all.csv.

The shards are a SHARDING ARTIFACT, not a data distinction: each campaign was
split across parallel workers and every worker appended to its own CSV. One row
per trained model either way, so the split carries no meaning once the campaign
is finished.

Schemas differ slightly between campaigns (duration_sec, combo), so this takes
the UNION of columns and records which campaign each row came from in a new
source_campaign column. Nothing is dropped or coerced.

Verifies the output row count equals the sum of inputs BEFORE deleting anything.

    python scripts/singleton/consolidate_progress.py <campaign>
    python scripts/singleton/consolidate_progress.py <campaign> --apply
"""
import argparse
import csv
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
KEEP = {"selection_manifest.csv", "provenance.csv", "progress_all.csv",
        "progress_selected.csv", "source_campaigns.csv"}


def campaign_of(fname):
    """'esb_tuner_scaled_17lt__mape_progress_shard10of17.csv' -> campaign, shard."""
    if "__" in fname:
        camp, rest = fname.split("__", 1)
    else:
        camp, rest = "unknown", fname
    m = re.search(r"shard(\d+)of(\d+)", rest)
    return camp, (f"{m.group(1)}of{m.group(2)}" if m else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", help="campaign folder name under results/")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    CAMPAIGN = RESULTS / args.campaign
    OUT = CAMPAIGN / "progress_all.csv"
    if not CAMPAIGN.is_dir():
        print(f"ERROR: no such campaign folder: {CAMPAIGN}")
        return 1

    files = sorted(f for f in CAMPAIGN.glob("*.csv") if f.name not in KEEP)
    if not files:
        print(f"nothing to consolidate under {CAMPAIGN}")
        return 1

    cols, rows, per_campaign = [], [], {}
    for f in files:
        camp, shard = campaign_of(f.name)
        with f.open(newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                r["source_campaign"] = camp
                r["source_shard"] = shard
                for k in r:
                    if k not in cols:
                        cols.append(k)
                rows.append(r)
                per_campaign[camp] = per_campaign.get(camp, 0) + 1

    print(f"input files : {len(files)}")
    for c, n in sorted(per_campaign.items()):
        print(f"   {c:28} {n:>6} rows")
    print(f"total rows  : {len(rows)}")
    print(f"union cols  : {len(cols)}")

    if not args.apply:
        print(f"\nDRY RUN. Would write {OUT} and delete {len(files)} shard files.")
        return 0

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)

    # Verify before deleting: read back and count.
    with OUT.open(newline="", encoding="utf-8") as fh:
        written = sum(1 for _ in csv.DictReader(fh))
    if written != len(rows):
        print(f"ERROR: wrote {written} rows, expected {len(rows)}. NOT deleting shards.")
        return 1
    print(f"\nwrote {OUT.name}: {written} rows (verified)")

    for f in files:
        f.unlink()
    print(f"deleted {len(files)} shard files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
