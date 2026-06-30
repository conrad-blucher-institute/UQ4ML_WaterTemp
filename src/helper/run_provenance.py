"""Record reproducibility provenance alongside experiment results.

Writes a single ``run_provenance.json`` into a result folder so a training run
can never be produced without a record of *exactly* what made it:

  - git commit SHA, branch, and a dirty flag (uncommitted changes present?)
  - UTC timestamp, hostname, python version
  - the fully-resolved config actually used for the run

Design intent (see docs/ESB_PIPELINE_DESIGN.md): call ``write_provenance`` from
the single result-writing site in the orchestration layer. Wired in once, it
cannot be forgotten. Pipeline-agnostic: pass any config as a dict (e.g.
``vars(args)``). Non-serializable values are stringified rather than crashing.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _git(*args: str) -> str | None:
    """Run a git command, returning stripped stdout or None if git/repo is absent."""
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def capture_provenance(config: dict) -> dict:
    """Build the provenance record (does not write anything)."""
    dirty = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
        "git_dirty_files": dirty.splitlines() if dirty else [],
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "config": config,
    }


def write_provenance(result_dir: str, config: dict) -> str:
    """Capture provenance and write ``run_provenance.json`` into ``result_dir``.

    Returns the path written. ``default=str`` keeps it from crashing on
    non-JSON-serializable config values (e.g. callables, keras objects).
    """
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, "run_provenance.json")
    with open(path, "w") as f:
        json.dump(capture_provenance(config), f, indent=2, default=str)
    return path


if __name__ == "__main__":
    # Smoke test: write a sample record to the current directory.
    out = write_provenance(".", {"example": "config", "lead_time": 12})
    print(f"wrote {out}")
    print(open(out).read())
