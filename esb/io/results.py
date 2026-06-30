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


def finalize_run(output_dir: str | Path, resolved_config: dict[str, Any]) -> str:
    """Write run_provenance.json into ``output_dir``. Returns the path written.

    Called unconditionally by the orchestration's single result-writing site, so
    a run can never be produced without a provenance record.
    """
    return _write_provenance()(str(output_dir), resolved_config)
