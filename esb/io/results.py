"""results IO — the single site that finalizes a run.

Provenance must be impossible to skip (design doc §5b). The orchestration calls
:func:`finalize_run` exactly once per run, and it ALWAYS writes
``run_provenance.json`` via ``src/helper/run_provenance.write_provenance``. The
tuner writes its own progress CSV / .keras files per worker; this module owns the
run-level record (git SHA + branch + dirty flag, timestamp, host, python, and the
fully-resolved config) that ties those artifacts to exactly what produced them.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


def _write_provenance():
    """Lazily resolve the provenance helper (either import layout)."""
    try:
        return import_module("src.helper.run_provenance").write_provenance
    except Exception:
        return import_module("helper.run_provenance").write_provenance


def finalize_run(output_dir: str | Path, resolved_config: dict[str, Any],
                 filename: str = "run_provenance.json") -> str:
    """Write the provenance record into ``output_dir``. Returns the path written.

    Called unconditionally by the orchestration's single result-writing site, so
    a run can never be produced without a provenance record. Sharded runs pass a
    per-shard ``filename`` so machines never collide on merge.
    """
    return _write_provenance()(str(output_dir), resolved_config, filename=filename)


def save_model_metadata(save_dir: str | Path, base_name: str, *,
                        feature_names: list[str] | None,
                        config: dict[str, Any],
                        scaling: str,
                        input_width: int | None = None,
                        history: dict[str, Any] | None = None) -> str:
    """Write ``<base_name>_metadata.json`` next to the ``.keras`` (single site).

    This is the artifact that replaces ``_scaler.joblib``. The scaler's NUMBERS
    now live inside the model (frozen Normalization layer), but two things still
    cannot live in the model and so must ship beside it:

      * ``feature_names`` - the input COLUMN ORDER. The Normalization layer
        applies mean/scale positionally and cannot detect a reordered matrix, so
        a wrong order is silently wrong. Consumers assert against this list.
      * ``input_width`` - the cheap count guard (wrong width fails loudly).

    Plus the per-model provenance needed to resolve this artifact back to the
    run that made it. NOTE this is per-MODEL scope; ``run_provenance*.json``
    stays, since the campaign-level grid fingerprint is a check ACROSS models
    that no per-model file can perform.

    ``history`` is the Keras training history (per-epoch metric lists). It used
    to be its own ``<base_name>_history.json``; it lives here now so one model is
    one model file plus one sidecar, not three. It sits under a single top-level
    ``"history"`` key so it stays trivially separable if a campaign ever grows
    large enough to want the curves in one columnar file instead.
    """
    import json

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "base_name": base_name,
        "scaling": scaling,               # "in_graph" | "none"
        "precision_note": "predictions are degrees C; sensor accuracy ~0.1 C",
        # Prefer a width measured from the artifact (the wrap path knows it from
        # scaler.n_features_in_) over one merely derived from the name list.
        "input_width": (input_width if input_width is not None
                        else (len(feature_names) if feature_names else None)),
        "feature_names": feature_names,
        "feature_order_note": (
            "Feed model inputs in EXACTLY this column order. The model scales "
            "internally; pass RAW values. Order is positional and unverifiable "
            "by the model - assert against this list before predicting."
        ),
        "provenance": {
            k: config.get(k) for k in (
                "model_type", "lead_time", "cycle", "activation", "num_layers",
                "neurons", "dropout", "run_num", "input_structure",
                "atp_hours_back", "wtp_hours_back", "pred_atp_interval",
                "input_hours_forecast",
            )
        },
    }
    if history is not None:
        payload["history"] = history

    out_path = save_dir / f"{base_name}_metadata.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return str(out_path)


# save_scaler() was DELETED here. The pipeline no longer persists a sklearn
# scaler beside the model: its statistics are frozen into the model's own
# Normalization layer, so the .keras is self-contained and there is no sidecar
# for a consumer to locate, pair and apply correctly. What still ships beside the
# model is save_model_metadata() above — the column ORDER, which the model cannot
# check for itself.
#
# Scalers already on disk from earlier campaigns are untouched;
# scripts/wrap_models_with_scaler.py reads them to upgrade those models.
