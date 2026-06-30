"""read stage (façade) — delegates to utils_mse_crps.readingData.

Owns: sorted glob, column naming, which year is held out. No logic moved (Stage B).
"""
from esb.stages import _fn


def read(path_to_data):
    """Read ESB year CSVs. Façade over utils_mse_crps.readingData()."""
    return _fn("readingData")(path_to_data)
