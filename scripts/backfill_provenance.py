#!/usr/bin/env python
"""Fill in the provenance block of wrapped models' metadata, from the filename.

Why this step exists
--------------------
``wrap_models_with_scaler.py`` can only record what it can MEASURE from the
artifacts it is handed: the input width (from the scaler) and the fact that
scaling is now in-graph. It cannot recover WHICH model it just wrapped -- lead
time, cycle, activation, layers, neurons, dropout, run number were never stored
inside the ``.keras``. They exist only in the filename. So it writes a
provenance block of nulls and leaves this to be filled in.

That block is not decoration. Every consumer resolves a model's identity through
``_metadata.json`` and never through its name (see esb/campaign.py). With nulls,
``esb.predict`` groups every model into one bogus ``(None, None)`` bucket,
``--only-leadtimes`` cannot filter, and the time-series page's combo keys all
collapse together.

This is the one place where parsing a filename is correct: the point of the step
is to PROMOTE what the name knows into metadata, exactly once, so that nothing
downstream ever has to look at a name again. Models whose name does not parse
are reported and left alone rather than guessed at.

What it writes
--------------
Grid axes come from the name. The feature knobs (input_structure, lag hours,
sampling interval) are not in the name and are taken from the campaign defaults
in ``esb/config.py``, so they cannot drift from the pipeline that produced the
models. ``input_hours_forecast`` mirrors the lead time. A ``provenance_source``
string records that these were derived rather than observed.

Everything else in the file -- the folded ``history``, ``feature_names``,
``input_width`` -- is preserved untouched. Writes are atomic (tmp + replace), so
an interrupted run leaves whole files, never half-written JSON.

Idempotent: a model whose provenance is already filled is reported as
up-to-date and skipped, unless ``--overwrite``.

Usage
-----
    python scripts/backfill_provenance.py results/phase4_top20            # dry run
    python scripts/backfill_provenance.py results/phase4_top20 --apply
    python scripts/backfill_provenance.py <folder> --shard 3/8 --apply    # parallel
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

META_SUFFIX = "_metadata.json"
WRAP_SUFFIX = "_rawinput"

# Same shape the rest of the toolchain expects, e.g.
#   MAPE_102h_cycle0_leaky_relu_2L_256N_d0.0_run3
NAME_RE = re.compile(
    r"^(?P<model_type>[A-Za-z]+)_(?P<lead_time>\d+)h_cycle(?P<cycle>\d+)"
    r"_(?P<activation>[a-z_]+)_(?P<num_layers>\d+)L_(?P<neurons>\d+)N"
    r"(?:_d(?P<dropout>[\d.]+))?_run(?P<run_num>\d+)$"
)

# Axes that come from the NAME, with the type each must land as.
GRID_FIELDS = {
    "model_type": str, "lead_time": int, "cycle": int, "activation": str,
    "num_layers": int, "neurons": int, "dropout": float, "run_num": int,
}
# Axes that are NOT in the name; read from esb/config.py so one source of truth
# governs both the runs and this backfill.
FEATURE_FIELDS = ("input_structure", "atp_hours_back", "wtp_hours_back",
                  "pred_atp_interval")


def campaign_defaults() -> dict:
    from esb.config import FIELDS_BY_NAME
    return {name: FIELDS_BY_NAME[name].default for name in FEATURE_FIELDS}


def parse_name(base_name: str):
    """Grid axes from a model's base name, or None if it does not parse."""
    stem = base_name
    if stem.endswith(WRAP_SUFFIX):
        stem = stem[: -len(WRAP_SUFFIX)]
    m = NAME_RE.match(stem)
    if not m:
        return None
    out = {}
    for field, cast in GRID_FIELDS.items():
        raw = m.group(field)
        if raw is None:
            # Only dropout is optional in the name; absent means no dropout.
            out[field] = 0.0 if cast is float else None
        else:
            out[field] = cast(raw)
    return out


def is_filled(provenance) -> bool:
    """True when the grid axes are already populated."""
    return provenance is not None and all(
        provenance.get(k) is not None for k in GRID_FIELDS)


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="campaign folder (recursive)")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default is a dry run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite provenance even where it is already filled")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process shard I of N (1-based), e.g. --shard 3/8")
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
        if n < 1 or not (1 <= i <= n):
            print(f"ERROR: --shard {i}/{n} is out of range (need 1 <= I <= N)")
            return 2

    if not args.folder.is_dir():
        print(f"ERROR: not a directory: {args.folder}")
        return 2

    metas = sorted(args.folder.rglob("*" + META_SUFFIX))
    if not metas:
        print("No *" + META_SUFFIX + f" under {args.folder} - nothing to do.")
        return 0
    print(f"Found {len(metas)} metadata files under {args.folder}")
    if shard:
        i, n = shard
        # Strided: each metadata file is independent (no shared state), so N
        # processes cover the set exactly once with an even mix of sizes.
        metas = metas[i - 1::n]
        print(f"--shard {i}/{n}: processing {len(metas)}")

    defaults = campaign_defaults()
    print(f"feature defaults from esb/config.py: {defaults}\n")

    filled = already = unparsed = failed = 0
    unparsed_names = []

    for meta_path in metas:
        try:
            payload = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  FAIL     {meta_path.name}: {e}")
            failed += 1
            continue

        base_name = payload.get("base_name", meta_path.name[: -len(META_SUFFIX)])
        if is_filled(payload.get("provenance")) and not args.overwrite:
            already += 1
            continue

        axes = parse_name(base_name)
        if axes is None:
            unparsed += 1
            if len(unparsed_names) < 5:
                unparsed_names.append(base_name)
            continue

        provenance = dict(payload.get("provenance") or {})
        provenance.update(axes)
        provenance.update(defaults)
        # Mirrors the lead time: the forecast horizon and the lead time are the
        # same quantity in this pipeline.
        provenance["input_hours_forecast"] = axes["lead_time"]
        payload["provenance"] = provenance
        payload["provenance_source"] = (
            "parsed from filename + campaign defaults (esb/config.py) "
            "by scripts/backfill_provenance.py")

        if args.apply:
            try:
                write_atomic(meta_path, payload)
            except OSError as e:
                print(f"  FAIL     {meta_path.name}: {e}")
                failed += 1
                continue
        filled += 1

    verb = "filled" if args.apply else "would fill"
    print(f"{verb}={filled} already-filled={already} "
          f"unparsed={unparsed} failed={failed}")
    if unparsed_names:
        print("\nUNPARSED (left untouched):")
        for name in unparsed_names:
            print(f"   {name}")
        print("   These keep null provenance and will be rejected downstream.")
    if not args.apply:
        print("\nDry run. Re-run with --apply to write.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
