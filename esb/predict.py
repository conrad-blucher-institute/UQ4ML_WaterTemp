"""Generate ONE WIDE prediction CSV per dataset for a campaign.

Replaces the per-model CSV explosion. One file per dataset, shaped:

    datetime, observation, <model_1>, <model_2>, ... <model_N>

The datetime is the TARGET time (when the forecast verifies), so ``observation``
is a property of the row alone and is stored ONCE instead of being repeated in
every model's file.

Why a cap
---------
This is deliberately unusable on a screening campaign. Phase 1 is judged on
metrics, not curves; a wide file over its ~50k models would be a multi-GB CSV
with a 2MB header that nothing can open. ``--max-models`` (default 500) makes
that fail immediately and loudly instead of after an hour of prediction.

Ragged coverage is expected
---------------------------
Models with different ``cycle`` values were tested on different year rotations,
so within one dataset file their columns cover different date ranges. The join
is an OUTER join and the gaps are blank. That is truthful, not a bug - a blank
means "this model was never evaluated at this timestamp", and plotting simply
skips it.

Inputs are RAW
--------------
This targets ``_rawinput.keras`` models, which carry their scaler inside as a
frozen Normalization layer. The arrays built here are raw and are fed as-is; no
scaler is loaded or applied. Feeding raw values to a NON-wrapped model would be
silently wrong, so the script checks metadata says ``scaling: in_graph`` and
refuses otherwise.

Where this lives
----------------
This is a data-production step, not a plot: it loads TensorFlow and runs every
model in a campaign. It sits in ``esb/`` rather than ``scripts/`` because it is
load-bearing (the whole time-series view is downstream of it) and because it
imports from ``esb`` - which ``scripts/`` is not allowed to do. Run it only when
the models change; the viz suite then reads the CSVs it leaves behind.

One file, or one per lead time
------------------------------
The default is one wide CSV per dataset spanning the whole campaign, which is
right for a hand-picked selection of tens of models. Past a few hundred it is
not: a campaign carrying the top 20 combos at every lead time is ~16,800 models,
and a single file would be multi-GB with a column for every model at every lead
time - of which any one chart uses a twentieth. ``--split-by-leadtime`` writes
``predictions_wide/LT<n>/<dataset>_wide.csv`` instead, one folder per lead time.

Lead times are fully independent, so ``--only-leadtimes`` shards the run across
processes; disjoint subsets produce byte-identical files to one serial run. In
split mode each lead time is written and dropped as soon as it finishes, so peak
memory is one lead time's columns rather than the whole campaign's.

Usage
-----
    python -m esb.predict results/phase3_selected
    python -m esb.predict <folder> --datasets 2021,test,val,train
    python -m esb.predict <folder> --limit 4    # smoke test

    # A full top-20 campaign, sharded four ways:
    python -m esb.predict <folder> --split-by-leadtime --max-models 20000 \
        --only-leadtimes 3,6,12,18,24,30
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Tuner defaults - must match what produced the models.
_INPUT_STRUCTURE = "descending"
_ATP_HOURS_BACK = 24
_WTP_HOURS_BACK = 24
_PRED_ATP_INTERVAL = 1
_IPP_OFFSET = 0.0

ALL_DATASETS = ("train", "test", "val", "2021")


def _data_path() -> str:
    return str(_REPO_ROOT / "data" / "ESB_datasets")


def prepare_splits(lead_time: int, cycle: int):
    """Return {dataset: (X, y, dates)} for train/test/val at one lead_time+cycle."""
    from src.helper.utils_mse_crps import (
        readingData, creatingAdditionalColumns, splittingData,
        deletingMissingValues, dateTimeRetriever,
    )

    year_dfs = readingData(_data_path())
    years = [
        creatingAdditionalColumns(
            df, _INPUT_STRUCTURE, lead_time,
            _ATP_HOURS_BACK, _WTP_HOURS_BACK,
            _PRED_ATP_INTERVAL, _IPP_OFFSET,
        )
        for df in year_dfs
    ]
    # readingData() already drops csv_files[0] (the 2021 independent year), so
    # years[0:4] are the four rotation years. independent_year="cycle" is the
    # regular-rotation branch: the test set comes from the rotated cycle year.
    # Passing the int instead leaves the test set EMPTY (see the R1 fix note in
    # utils_mse_crps.preparingData).
    training_df, testing_df, validation_df = splittingData(
        years[0], years[1], years[2], years[3], "cycle", cycle)

    col_start = 1 if _INPUT_STRUCTURE == "descending" else 3
    out = {}
    for name, frame in (("train", training_df), ("test", testing_df),
                        ("val", validation_df)):
        clean = deletingMissingValues(frame)
        dates = dateTimeRetriever(clean.copy(), lead_time)
        X = clean.iloc[:, col_start:-1].values.astype(float)
        y = clean.iloc[:, -1].values.astype(float)
        out[name] = (X, y, dates)
    return out


def prepare_2021(lead_time: int):
    """Return (X, y, dates) for the independent year."""
    import glob
    import os
    from src.helper.utils_mse_crps import prepare_independent_year

    csv_files = sorted(glob.glob(os.path.join(_data_path(), "*.csv")))
    # TODO: the independent set is hardcoded to the first file. Parameterizing
    # this is a separate todo item (needed for the esb_2025_2026 set).
    return prepare_independent_year(
        csv_path=csv_files[0],
        input_structure=_INPUT_STRUCTURE,
        lead_time=lead_time,
        atp_hours_back=_ATP_HOURS_BACK,
        wtp_hours_back=_WTP_HOURS_BACK,
        pred_atp_interval=_PRED_ATP_INTERVAL,
        IPPOffset=_IPP_OFFSET,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="campaign folder")
    ap.add_argument("--datasets", default="2021,test",
                    help=f"comma-separated subset of {','.join(ALL_DATASETS)} "
                         f"(default: 2021,test)")
    ap.add_argument("--max-models", type=int, default=500,
                    help="refuse to run beyond this many models (default 500)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N models (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default <folder>/predictions_wide)")
    ap.add_argument("--split-by-leadtime", action="store_true",
                    help="write one wide CSV per lead time into LT<n>/ "
                         "subfolders instead of one file spanning every lead "
                         "time. Required past a few hundred models: a single "
                         "file over a full top-20 campaign is multi-GB and has "
                         "a column for every model at every lead time, of which "
                         "any one chart uses a twentieth.")
    ap.add_argument("--only-leadtimes", default=None,
                    help="comma-separated lead times to process (e.g. 3,6,12). "
                         "The shard knob: lead times are independent, so N "
                         "processes over disjoint subsets produce exactly the "
                         "same files as one long serial run.")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        print(f"ERROR: unknown dataset(s) {unknown}; choose from {ALL_DATASETS}")
        return 2

    from esb.campaign import load_campaign

    models = load_campaign(args.folder)
    if args.limit:
        models = models[: args.limit]
    print(f"Loaded {len(models)} models from {args.folder}")

    if args.only_leadtimes:
        wanted = {float(v) for v in args.only_leadtimes.split(",") if v.strip()}
        models = [m for m in models if float(m["lead_time"]) in wanted]
        missing = wanted - {float(m["lead_time"]) for m in models}
        if missing:
            # A typo'd lead time would otherwise shard into an empty run that
            # looks like a success and leaves a hole in the 21 files.
            print(f"ERROR: no models at lead time(s) "
                  f"{sorted(missing)} in {args.folder}")
            return 2
        print(f"  --only-leadtimes {sorted(wanted)}: {len(models)} models")

    if len(models) > args.max_models:
        print(
            f"\nERROR: {len(models)} models exceeds --max-models {args.max_models}.\n"
            f"A wide CSV over this many models is not a usable artifact - this is\n"
            f"a screening campaign, judged on metrics rather than curves. Promote a\n"
            f"selection into its own phase folder first (esb.campaign.promote), or\n"
            f"raise --max-models deliberately if you really mean it."
        )
        return 2

    # Wrapped models take RAW inputs. A model still expecting SCALED inputs would
    # produce plausible-looking garbage here rather than failing, so check.
    not_wrapped = [m["base_name"] for m in models
                   if m["metadata"].get("scaling") != "in_graph"]
    if not_wrapped:
        print(
            f"\nERROR: {len(not_wrapped)} model(s) do not have scaling='in_graph' "
            f"and so expect PRE-SCALED inputs, which this script does not build:\n"
            f"  " + "\n  ".join(not_wrapped[:5])
        )
        return 2

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    out_dir = args.out_dir or (Path(args.folder) / "predictions_wide")
    out_dir.mkdir(parents=True, exist_ok=True)

    # One prediction column per model, accumulated per dataset as
    # {dataset: {model_base_name: Series indexed by datetime}}.
    columns: dict[str, dict[str, "pd.Series"]] = {d: {} for d in datasets}
    observations: dict[str, "pd.Series"] = {}

    def flush(destination: Path) -> None:
        """Write the accumulated columns out and reset the accumulators."""
        destination.mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            obs = observations[dataset].sort_index()
            frame = pd.DataFrame({"observation": obs})
            # Outer join: ragged coverage across cycles is expected (docstring).
            frame = frame.join(pd.DataFrame(columns[dataset]), how="outer")
            frame.index.name = "datetime"
            out_path = destination / f"{dataset}_wide.csv"
            frame.to_csv(out_path, float_format="%.4f")
            size_mb = out_path.stat().st_size / 1e6
            print(f"\n{dataset}: {len(frame)} rows x {len(frame.columns)} cols "
                  f"-> {out_path} ({size_mb:.1f} MB)")
        columns.clear()
        columns.update({d: {} for d in datasets})
        # Resetting observations also restarts the cross-group agreement check
        # below, so in split mode it compares cycles within one lead time rather
        # than across all of them. That is the comparison that matters: the
        # observation at a target time does not depend on lead time, but the
        # cycles of one lead time are what share a file and must line up in it.
        observations.clear()

    groups = defaultdict(list)
    for m in models:
        groups[(m["lead_time"], m["cycle"])].append(m)

    # Sorted by (lead_time, cycle), so every cycle of one lead time is
    # contiguous. In split mode that lets each lead time be written and dropped
    # as soon as it completes, which is what keeps peak memory at one lead time's
    # worth of columns rather than the whole campaign's.
    current_lt = None
    for (lt, cy), group in sorted(groups.items()):
        if args.split_by_leadtime and current_lt is not None and lt != current_lt:
            flush(out_dir / f"LT{current_lt:g}")
        current_lt = lt
        print(f"\n--- lead_time {lt}h, cycle {cy} ({len(group)} models) ---")
        prepared: dict[str, tuple] = {}
        if {"train", "test", "val"} & set(datasets):
            print("  preparing train/test/val ...")
            prepared.update(prepare_splits(lt, cy))
        if "2021" in datasets:
            print("  preparing 2021 ...")
            prepared["2021"] = prepare_2021(lt)

        tensors = {d: tf.constant(prepared[d][0], dtype=tf.float32)
                   for d in datasets}

        for dataset in datasets:
            _, y, dates = prepared[dataset]
            index = pd.to_datetime(pd.Index(dates), errors="coerce")
            obs = pd.Series(np.asarray(y, dtype=float), index=index)
            obs = obs[~obs.index.duplicated(keep="first")]
            if dataset not in observations:
                observations[dataset] = obs
            else:
                # The observation at a target timestamp cannot depend on which
                # model predicted it. If it does, the date alignment is wrong and
                # every curve drawn from this file would be subtly misplaced.
                shared = observations[dataset].index.intersection(obs.index)
                if len(shared):
                    delta = (observations[dataset].loc[shared]
                             - obs.loc[shared]).abs().max()
                    if delta > 1e-6:
                        raise RuntimeError(
                            f"observation mismatch in '{dataset}' at lead_time "
                            f"{lt}h cycle {cy}: max |diff| = {delta:.6g}. The "
                            f"target-time alignment disagrees between groups."
                        )
                observations[dataset] = observations[dataset].combine_first(obs)

        for model in group:
            name = model["base_name"]
            print(f"  predicting {name}")
            net = tf.keras.models.load_model(model["keras_path"], compile=False)
            for dataset in datasets:
                _, y, dates = prepared[dataset]
                pred = net(tensors[dataset], training=False).numpy().flatten()
                index = pd.to_datetime(pd.Index(dates), errors="coerce")
                series = pd.Series(pred.astype(float), index=index)
                columns[dataset][name] = series[~series.index.duplicated(
                    keep="first")]
            tf.keras.backend.clear_session()

    if current_lt is None:
        print("ERROR: no models to predict with")
        return 2
    flush(out_dir / f"LT{current_lt:g}" if args.split_by_leadtime else out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
