"""Bake a model's sidecar scaler INTO the model as a frozen Normalization layer.

Path A: for models that are ALREADY TRAINED. Retraining is not needed and does
not happen here — a ``Normalization`` layer is non-trainable and deterministic,
so prepending it to a trained model and feeding raw inputs is arithmetically the
same as feeding ``scaler.transform(x)`` to the original model. This script does
that rewrite and then PROVES it on random inputs before writing anything.

Input  : <base>.keras  +  <base>_scaler.joblib   (as produced by the campaigns)
Output : <base>_rawinput.keras  +  <base>_metadata.json

The suffix is deliberate. The wrapped model takes RAW inputs; the original takes
SCALED ones. Two files that look alike and silently disagree is exactly the
failure this whole change exists to remove, so the names must differ.

Running it in parallel
----------------------
The work is per-model and shares nothing between models, so ``--shard I/N``
splits it across processes: run I=1..N and together they cover every pair
exactly once. Each shard writes a disjoint set of output files into the same
folder, so no locking is involved. Measured at ~0.3 s/model, which is ~1.4 h for
a 16,800-model campaign serially and minutes across 8 workers.

Usage
-----
    python scripts/wrap_models_with_scaler.py results/esb_tuner_scaled
    python scripts/wrap_models_with_scaler.py <folder> --dry-run
    python scripts/wrap_models_with_scaler.py <folder> --tolerance 1e-4
    python scripts/wrap_models_with_scaler.py <folder> --limit 20    # time it
    python scripts/wrap_models_with_scaler.py <folder> --shard 3/8   # parallel

Notes
-----
* Needs TensorFlow + sklearn + joblib, so run it on grendal, not the laptop.
* Existing ``*_rawinput.keras`` files are skipped unless ``--overwrite``.
* Any model whose verification fails is reported and NOT written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def find_pairs(folder: Path) -> list[tuple[Path, Path]]:
    """(.keras, _scaler.joblib) pairs, skipping already-wrapped models."""
    pairs = []
    for keras_path in sorted(folder.rglob("*.keras")):
        if keras_path.stem.endswith("_rawinput"):
            continue
        scaler_path = keras_path.with_name(f"{keras_path.stem}_scaler.joblib")
        if scaler_path.exists():
            pairs.append((keras_path, scaler_path))
    return pairs


def wrap_one(keras_path: Path, scaler_path: Path, tolerance: float):
    """Return (wrapped_model, max_abs_diff, n_features). Raises if inputs disagree."""
    import joblib
    import numpy as np
    from tensorflow import keras

    from esb.stages.scale import normalization_stats

    inner = keras.models.load_model(keras_path, compile=False)
    scaler = joblib.load(scaler_path)

    n_features = int(scaler.n_features_in_)
    model_width = int(inner.inputs[0].shape[-1])
    if model_width != n_features:
        raise ValueError(
            f"scaler width {n_features} != model input width {model_width} — "
            f"this scaler does not belong to this model"
        )

    mean, variance = normalization_stats(scaler)
    norm = keras.layers.Normalization(axis=-1, mean=mean, variance=variance)

    raw_in = keras.Input(shape=(n_features,), name="raw_inputs")
    wrapped = keras.Model(raw_in, inner(norm(raw_in)),
                          name=f"{keras_path.stem}_rawinput")

    # Prove equivalence BEFORE writing. Random inputs spanning a wide range so a
    # per-feature mean/scale mix-up cannot pass by luck.
    rng = np.random.RandomState(0)
    x = rng.uniform(-40.0, 60.0, size=(64, n_features)).astype("float32")
    got = wrapped.predict(x, verbose=0)
    want = inner.predict(scaler.transform(x), verbose=0)
    max_diff = float(np.max(np.abs(got - want)))
    if not (max_diff <= tolerance):
        raise AssertionError(
            f"wrapped output differs from scaler+model by {max_diff:.3e} "
            f"(tolerance {tolerance:.1e})"
        )
    return wrapped, max_diff, n_features


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="results folder to walk (recursive)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify equivalence but write nothing")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite existing *_rawinput.keras files")
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="max allowed |wrapped - (scaler+model)| (default 1e-4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N pairs. For timing a run "
                         "before committing to the whole campaign - the work is "
                         "per-model and uniform, so N models extrapolate.")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process shard I of N (1-based), e.g. --shard 3/8. "
                         "Run N processes with I=1..N to wrap in parallel; "
                         "together they cover every pair exactly once.")
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
        # A silently-empty shard would look like a clean run and leave a hole in
        # the campaign that only surfaces much later, at predict time.
        if n < 1 or not (1 <= i <= n):
            print(f"ERROR: --shard {i}/{n} is out of range (need 1 <= I <= N)")
            return 2

    if not args.folder.is_dir():
        print(f"ERROR: not a directory: {args.folder}")
        return 2

    pairs = find_pairs(args.folder)
    if not pairs:
        print(f"No (.keras + _scaler.joblib) pairs found under {args.folder}")
        return 1
    print(f"Found {len(pairs)} model/scaler pairs under {args.folder}")
    if shard:
        i, n = shard
        # STRIDED, not a contiguous block: pairs are sorted by filename, so
        # blocks would clump one architecture (and one cost) into one worker.
        # Striding interleaves them, so every shard gets the same mix.
        pairs = pairs[i - 1::n]
        print(f"--shard {i}/{n}: processing {len(pairs)}")
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"--limit {args.limit}: processing {len(pairs)}")
    print()

    from esb.io.results import save_model_metadata

    wrote = skipped = failed = verified = 0
    for keras_path, scaler_path in pairs:
        base = keras_path.stem
        out_path = keras_path.with_name(f"{base}_rawinput.keras")
        if out_path.exists() and not args.overwrite:
            print(f"  SKIP    {base} (already wrapped; --overwrite to redo)")
            skipped += 1
            continue
        try:
            wrapped, max_diff, n_features = wrap_one(
                keras_path, scaler_path, args.tolerance)
        except Exception as e:
            print(f"  FAIL    {base}: {e}")
            failed += 1
            continue

        if args.dry_run:
            print(f"  OK(dry) {base}  max_diff={max_diff:.2e}  n_features={n_features}")
            verified += 1
            continue

        wrapped.save(out_path)
        save_model_metadata(
            out_path.parent, out_path.stem,
            # Path A cannot recover the column NAMES from the .keras — the run's
            # config is not carried in it. The width IS known (measured from the
            # scaler and cross-checked against the model above), so record that;
            # names get backfilled from the filename separately.
            feature_names=None,
            input_width=n_features,
            config={"wrapped_from": keras_path.name},
            scaling="in_graph",
        )
        print(f"  WROTE   {out_path.name}  max_diff={max_diff:.2e}")
        wrote += 1

    if args.dry_run:
        print(f"\nverified={verified} failed={failed}  (dry run — nothing written)")
    else:
        print(f"\nwrote={wrote} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
