"""split stage (façade) — delegates to utils_mse_crps.splittingData.

Owns: 4-year temporal rotation by `rotation` (a.k.a. cycle). No logic moved
(Stage B). Note: the underlying function's 6th arg is named `cycle`; the esb
interface calls it `rotation` (R6).
"""
from esb.stages import _fn


def split(year2, year3, year4, year5, year_independent, rotation):
    """Split into train/val/test for one rotation. Façade over splittingData()."""
    return _fn("splittingData")(year2, year3, year4, year5, year_independent, rotation)
