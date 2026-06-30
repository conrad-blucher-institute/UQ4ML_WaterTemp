"""Pipeline stages — Stage B: THIN FAÇADES over existing functions.

Each module here wraps the corresponding function in
``src/helper/utils_mse_crps.py`` (or the grid-search tuner) **without moving any
logic**. They exist as the named seams the design doc describes
(read → features → clean → split → reshape → scale → model → train → evaluate →
infer) so that Stages C and D can replace internals one at a time.

In Stage B the LIVE path does not call these — orchestration delegates to the
existing tuner's ``run_tuning()`` (the real lead_times × rotations loop). These
façades are import-light: the heavy ``utils_mse_crps`` import happens lazily
inside each wrapper, so importing ``esb`` stays cheap and TF-free.
"""

from importlib import import_module
from typing import Any


def _utils():
    """Lazily import the canonical helper module (resolves either layout)."""
    try:
        return import_module("src.helper.utils_mse_crps")
    except Exception:
        return import_module("helper.utils_mse_crps")


def _fn(name: str):
    """Fetch a function from the canonical helper by name (lazy)."""
    return getattr(_utils(), name)
