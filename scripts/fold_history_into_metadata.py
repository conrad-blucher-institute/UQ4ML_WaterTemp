"""Fold each <base>_history.json into its <base>_metadata.json, then delete it.

One job should be one .keras plus ONE sidecar. The tuner now writes the training
history under the metadata file's top-level "history" key; this script migrates
campaigns produced before that change.

Pairing note
------------
Metadata is named after the ARTIFACT, so wrapped models carry the ``_rawinput``
suffix (``..._run1_rawinput_metadata.json``) while their history was written by
the tuner under the pre-wrap name (``..._run1_history.json``). This script tries
the exact name first and then the ``_rawinput`` variant, and reports any history
file it cannot pair rather than guessing.

Usage
-----
    python scripts/fold_history_into_metadata.py results/phase3_selected
    python scripts/fold_history_into_metadata.py <folder> --apply
    python scripts/fold_history_into_metadata.py <folder> --apply --keep-history

Dry-run is the default: nothing is written or deleted without ``--apply``.
Stdlib only, so it runs anywhere. Re-runnable: a history already folded in is
reported as up-to-date and skipped (its file is still removed under --apply).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HISTORY_SUFFIX = "_history.json"
META_SUFFIX = "_metadata.json"


def find_metadata_for(history_path: Path) -> Path | None:
    """Locate the metadata file belonging to a history file, or None."""
    base = history_path.name[: -len(HISTORY_SUFFIX)]
    for candidate in (f"{base}{META_SUFFIX}", f"{base}_rawinput{META_SUFFIX}"):
        path = history_path.with_name(candidate)
        if path.exists():
            return path
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path,
                    help="campaign folder to walk (recursive)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write metadata and delete history files")
    ap.add_argument("--keep-history", action="store_true",
                    help="write the metadata but leave the _history.json in place")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process shard I of N (1-based), e.g. --shard 3/8. "
                         "Run N processes with I=1..N to fold in parallel; "
                         "together they cover every history file exactly once.")
    args = ap.parse_args()

    shard = None
    if args.shard:
        try:
            i_str, n_str = args.shard.split("/")
            shard = (int(i_str), int(n_str))
        except ValueError:
            print(f"ERROR: --shard must look like I/N, got {args.shard!r}")
            return 2
        i, n = shard
        # A silently-empty shard would look like a clean run and leave histories
        # unfolded that nothing downstream would flag.
        if n < 1 or not (1 <= i <= n):
            print(f"ERROR: --shard {i}/{n} is out of range (need 1 <= I <= N)")
            return 2

    if not args.folder.is_dir():
        print(f"ERROR: not a directory: {args.folder}")
        return 2

    histories = sorted(args.folder.rglob(f"*{HISTORY_SUFFIX}"))
    if not histories:
        print(f"No *{HISTORY_SUFFIX} found under {args.folder} - nothing to do.")
        return 0
    print(f"Found {len(histories)} history files under {args.folder}")
    if shard:
        i, n = shard
        # Sharding is safe here for a reason worth stating: each history file
        # pairs with its OWN metadata file and nothing is shared between pairs.
        # (Contrast merge_reps.py, which rewrites one provenance.csv and so
        # cannot be sharded without locking.) Strided, so every shard gets the
        # same mix of large and small histories.
        histories = histories[i - 1::n]
        print(f"--shard {i}/{n}: processing {len(histories)}")
    print()

    folded = already = orphan = failed = 0
    orphans: list[Path] = []

    for hist_path in histories:
        meta_path = find_metadata_for(hist_path)
        if meta_path is None:
            orphan += 1
            orphans.append(hist_path)
            continue

        try:
            history = json.loads(hist_path.read_text())
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"  FAIL    {hist_path.name}: {e}")
            failed += 1
            continue

        if "history" in meta:
            # Idempotent re-run. Only treat it as done if the content matches,
            # so a genuine mismatch is reported instead of silently dropped.
            if meta["history"] == history:
                already += 1
                if args.apply and not args.keep_history:
                    hist_path.unlink()
                continue
            print(f"  FAIL    {meta_path.name}: existing 'history' key differs "
                  f"from {hist_path.name}; refusing to overwrite")
            failed += 1
            continue

        if args.apply:
            meta["history"] = history
            # Write to a temp file then replace, so an interrupted run cannot
            # leave a truncated metadata file behind.
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(meta, indent=2))
            tmp.replace(meta_path)
            if not args.keep_history:
                hist_path.unlink()
        folded += 1

    if orphans:
        print(f"\n{len(orphans)} history file(s) with NO matching metadata:")
        for path in orphans[:20]:
            print(f"    {path.name}")
        if len(orphans) > 20:
            print(f"    ... and {len(orphans) - 20} more")
        print("  (left untouched - resolve these before relying on the campaign)")

    verb = "folded" if args.apply else "would fold"
    print(f"\n{verb}={folded} already-present={already} "
          f"orphan={orphan} failed={failed}")
    if not args.apply:
        print("DRY RUN - nothing written. Re-run with --apply.")
    return 1 if (failed or orphan) else 0


if __name__ == "__main__":
    raise SystemExit(main())
