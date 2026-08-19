"""Campaign loading and phase promotion.

Two functions, both built on one idea: **a model's identity lives in its
``_metadata.json``, not in its filename.** Filenames are for humans. Every
consumer that re-parsed them had to be updated when the grid grew a new axis,
and the one that wasn't (``generate_predictions.py``) silently broke. Nothing
here parses a filename.

``load_campaign`` reads a folder once and hands back a list. Pass that list
around instead of making each function re-glob and re-parse the folder itself.

``promote`` is the phase step: given a hand-editable "passed" CSV naming the
combos that survived, copy those models (all their reps, plus every sidecar)
into the next phase's folder.

    models = load_campaign("results/phase3_selected")

    promote("results/esb_tuning_phase1_grid_21lt",
            "results/phase2_tuner",
            "results/phase1_passed.csv")

Stdlib only - no pandas - so these run in any environment.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

# The grid axes that identify a COMBO (everything except which repetition it is).
# A passed-CSV may name any subset of these; matching uses the ones it provides.
COMBO_FIELDS = (
    "model_type", "lead_time", "cycle", "activation",
    "num_layers", "neurons", "dropout",
)

METADATA_SUFFIX = "_metadata.json"


def _keras_dir(folder: str | Path) -> Path:
    """Accept either a campaign root or the keras_files/ dir itself."""
    folder = Path(folder)
    nested = folder / "keras_files"
    return nested if nested.is_dir() else folder


def _metadata_for(keras_path: Path) -> Path | None:
    """Metadata sidecar for a model, or None. Sidecar is named for the artifact."""
    path = keras_path.with_name(f"{keras_path.stem}{METADATA_SUFFIX}")
    return path if path.exists() else None


def load_campaign(folder: str | Path) -> list[dict[str, Any]]:
    """Load every model in a campaign once. Returns one dict per ``.keras``.

    Each entry carries the resolved paths plus the model's identity, read from
    its metadata sidecar:

        {"keras_path": Path, "base_name": str, "metadata": {...},
         "model_type": "MAPE", "lead_time": 120, "cycle": 0, ...}

    Raises if any model lacks metadata. That is deliberate: a model whose
    identity was never recorded cannot be safely matched, promoted or predicted
    with, and quietly skipping it would silently shrink the campaign.
    """
    keras_dir = _keras_dir(folder)
    if not keras_dir.is_dir():
        raise NotADirectoryError(f"no such campaign folder: {keras_dir}")

    models: list[dict[str, Any]] = []
    missing: list[str] = []

    for keras_path in sorted(keras_dir.glob("*.keras")):
        meta_path = _metadata_for(keras_path)
        if meta_path is None:
            missing.append(keras_path.name)
            continue
        meta = json.loads(meta_path.read_text())
        entry: dict[str, Any] = {
            "keras_path": keras_path,
            "metadata_path": meta_path,
            "base_name": meta.get("base_name", keras_path.stem),
            "metadata": meta,
        }
        # Hoist identity to the top level so callers write model["lead_time"]
        # rather than model["metadata"]["provenance"]["lead_time"].
        entry.update(meta.get("provenance", {}))
        models.append(entry)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} model(s) in {keras_dir} have no{METADATA_SUFFIX}: "
            f"{', '.join(missing[:5])}"
            + (f", ... (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        )
    if not models:
        raise FileNotFoundError(f"no .keras files found in {keras_dir}")
    return models


def _norm(value: Any) -> Any:
    """Normalize one identity value so CSV text matches metadata JSON types.

    '0.0', '0', 0 and 0.0 must all be the same dropout, and '120' the same lead
    time as 120. Numbers compare as floats; anything else as a stripped string.
    """
    if value is None:
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text


def _combo_key(source: dict[str, Any], fields: tuple[str, ...]) -> tuple:
    return tuple(_norm(source.get(f)) for f in fields)


def promote(src_folder: str | Path, dst_folder: str | Path,
            passed_csv: str | Path, *, dry_run: bool = False,
            overwrite: bool = False) -> list[dict[str, Any]]:
    """Copy the combos named in ``passed_csv`` from one phase into the next.

    ``passed_csv`` is meant to be hand-editable (or written by a top-N script).
    It needs a header row naming any subset of :data:`COMBO_FIELDS`; matching
    uses exactly the columns present, so a CSV listing only lead_time+activation
    legitimately promotes every combo sharing those. Extra columns (``notes``,
    ``modeler_favorite``, ...) are ignored here and carried by the CSV itself.

    A combo means ALL of its repetitions - reps are the sample, and promoting a
    subset of them would change the statistics the selection was made from.
    Every sidecar beside a promoted model is copied with it.

    Copies, never moves: the source phase stays intact as the record of what was
    screened. Returns the list of promoted model entries.
    """
    src_folder, dst_folder = Path(src_folder), Path(dst_folder)
    passed_csv = Path(passed_csv)

    models = load_campaign(src_folder)

    with open(passed_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{passed_csv} has no rows")

    fields = tuple(f for f in COMBO_FIELDS if f in rows[0])
    if not fields:
        raise ValueError(
            f"{passed_csv} names none of the combo columns {COMBO_FIELDS}; "
            f"it has {list(rows[0])}"
        )
    print(f"Matching on: {', '.join(fields)}")

    wanted = {_combo_key(row, fields) for row in rows}
    selected = [m for m in models if _combo_key(m, fields) in wanted]

    # A combo listed as passed but absent from the source is a real problem -
    # usually the wrong source phase, or a name typed by hand. Say so.
    found = {_combo_key(m, fields) for m in selected}
    for key in sorted(wanted - found, key=str):
        print(f"  WARNING: no models in {src_folder.name} match combo {key}")

    n_combos = len(found)
    print(f"{len(rows)} row(s) -> {n_combos} combo(s) -> "
          f"{len(selected)} model(s) of {len(models)} in source")
    if not selected:
        raise ValueError("nothing matched; refusing to create an empty phase")

    dst_keras = _keras_dir(dst_folder) if (Path(dst_folder) / "keras_files").is_dir() \
        else Path(dst_folder) / "keras_files"
    if not dst_keras.is_dir() and not dry_run:
        dst_keras.mkdir(parents=True)
        print(f"Created destination folder: {dst_keras}")

    src_keras = _keras_dir(src_folder)
    copied = skipped = 0
    for model in selected:
        # Every file sharing this model's stem: the .keras and all its sidecars.
        for path in sorted(src_keras.glob(f"{model['keras_path'].stem}*")):
            target = dst_keras / path.name
            if target.exists() and not overwrite:
                skipped += 1
                continue
            if not dry_run:
                shutil.copy2(path, target)
            copied += 1

    verb = "would copy" if dry_run else "copied"
    print(f"{verb}={copied} skipped(exists)={skipped} -> {dst_keras}")
    if dry_run:
        print("DRY RUN - nothing written.")
    return selected
