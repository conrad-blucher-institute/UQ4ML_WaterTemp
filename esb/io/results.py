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


def save_scaler(save_dir: str | Path, base_name: str, scaler: Any) -> str | None:
    """Persist a fitted scaler as ``<base_name>_scaler.joblib`` (single site).

    The ONE place a scaler is written, so it always lands next to the ``.keras``
    it belongs to (same ``base_name``) and can never drift from its model. A
    ``None`` scaler (scaling off) is a no-op and returns ``None`` — the contract
    stays uniform whether or not scaling ran. Returns the path written.
    """
    if scaler is None:
        return None
    import joblib

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{base_name}_scaler.joblib"
    joblib.dump(scaler, out_path)
    return str(out_path)
