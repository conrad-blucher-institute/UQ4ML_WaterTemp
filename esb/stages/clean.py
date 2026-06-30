"""clean stage (façade) — delegates to utils_mse_crps missing-value helpers.

Owns: -999 + NaN rules, loud row-count reporting. No logic moved (Stage B).
"""
from esb.stages import _fn


def count_missing(df):
    """Façade over countingMissingValues()."""
    return _fn("countingMissingValues")(df)


def delete_missing(df):
    """Façade over deletingMissingValues()."""
    return _fn("deletingMissingValues")(df)
