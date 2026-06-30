"""reshape stage (façade) — delegates to utils_mse_crps.reshaping.

Owns: the col_start invariant (1 descending / 3 ascending). No logic moved
(Stage B). This is where the duplicated col_start rule will collapse to one
place in Stage D.
"""
from esb.stages import _fn


def reshape(input_structure, training, testing, validation, model):
    """Slice frames into x/y arrays. Façade over reshaping()."""
    return _fn("reshaping")(input_structure, training, testing, validation, model)
