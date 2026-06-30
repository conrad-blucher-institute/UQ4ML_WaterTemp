"""evaluate stage (façade) — delegates to utils metric helpers.

Owns: metric computation (mae, mae12, ...). No logic moved (Stage B): the live
path computes post-hoc metrics inside the tuner. Façade over utils.mae12, which
the tuner uses for val/2021 evaluation.
"""
from importlib import import_module


def _utils():
    try:
        return import_module("src.helper.utils")
    except Exception:
        return import_module("helper.utils")


def mae12(y_true, y_pred):
    """Cold-stunning (<12°C) MAE. Façade over utils.mae12()."""
    return _utils().mae12(y_true, y_pred)
