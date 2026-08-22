"""The phase-5 tuner subclass, in its own module so it is picklable.

``base_tuner.run_tuning`` submits ``self._tune_single_config`` to a
ProcessPoolExecutor. That pickles ``self``, and unpickling in the worker looks
the class up by ``module.qualname`` -- so the class must exist at module level
in a module the CHILD can import. A class created inside a function does not,
even if the parent injects it into globals(): the child re-imports the module
fresh, the factory never runs there, and unpickling dies with

    AttributeError: Can't get attribute 'Phase5Tuner' on <module ...>

Importing this module pulls in TensorFlow (via mape_tuner), which is why
``esb.retrain_final`` imports it lazily -- ``--dry-run`` stays TF-free.

See ``esb.retrain_final`` for what phase 5 is and why the arrays are assembled
this way.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The tuner modules use bare imports (`from base_tuner import ...`). This module
# can be imported directly by a spawned worker, before esb.retrain_final has run
# its own bootstrap, so it must set the path up itself.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUNE_DIR = _REPO_ROOT / "esb_03_2026_tune_experiment"
for _p in (_TUNE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mape_tuner import MAPETuner  # noqa: E402  (must follow the sys.path setup)

# The independent-year file that preparingData() deliberately skips
# (readingData drops sorted(*.csv)[0]) and that phase 5 folds into training.
INDEPENDENT_CSV_NAME = "esb_2020_2021.csv"

# One entry per lead time, not per (lead time, rep): the arrays are identical
# across reps, and feature engineering over 4 years costs far more than the fit.
# Module-level, so the cache is per worker PROCESS.
#
# NOTE base_tuner opens a NEW ProcessPoolExecutor per batch of workers*2 jobs,
# so processes are torn down regularly and this cache only helps within a batch.
# Hoisting data loading above the executor is the real fix (see the
# 'Load training data once' TODO in mape_tuner) and is deliberately not done
# here -- it changes shared tuner code for every campaign, not just phase 5.
_ARRAY_CACHE: dict = {}


def phase5_arrays(prep_kwargs: dict):
    """Return (x_train+test+2021, y..., x_val, y_val, x_test, y_test).

    Concatenates the ORIGINAL test split and the independent 2021 year into
    training. Validation is untouched and still drives early stopping.

    Safe without forking splittingData because reshaping() is pure column
    slicing with no interaction between rows, and prepare_independent_year()
    slices with the identical col_start -- so concatenating arrays afterwards
    equals concatenating dataframes beforehand.
    """
    import numpy as np
    from src.helper.utils_mse_crps import preparingData, prepare_independent_year

    lead_time = prep_kwargs["input_hours_forecast"]
    cache_key = (lead_time, prep_kwargs["cycle"], prep_kwargs["input_structure"])
    if cache_key in _ARRAY_CACHE:
        return _ARRAY_CACHE[cache_key]

    # Standard rotation: train / val / test for this cycle.
    data = preparingData(**prep_kwargs)
    if data is None or len(data) < 6:
        raise RuntimeError(
            "preparingData() must return the full 6 arrays for phase 5 "
            f"(got {None if data is None else len(data)})"
        )
    x_train, y_train, x_val, y_val, x_test, y_test = data[:6]

    indep_csv = Path(prep_kwargs["path_to_data"]) / INDEPENDENT_CSV_NAME
    if not indep_csv.is_file():
        raise RuntimeError(f"independent year CSV not found: {indep_csv}")
    x_2021, y_2021, _dates = prepare_independent_year(
        csv_path=str(indep_csv),
        input_structure=prep_kwargs["input_structure"],
        lead_time=lead_time,
        atp_hours_back=prep_kwargs["atp_hours_back"],
        wtp_hours_back=prep_kwargs["wtp_hours_back"],
        pred_atp_interval=prep_kwargs["pred_atp_interval"],
        IPPOffset=prep_kwargs["IPPOffset"],
        scaler=None,   # raw on purpose: the model scales in-graph
    )

    # Width guard. A mismatch means the independent year was engineered with
    # different lag settings; concatenating anyway would raise deep inside
    # numpy, or worse, broadcast.
    for name, arr in (("test", x_test), ("2021", x_2021)):
        if arr.shape[1] != x_train.shape[1]:
            raise RuntimeError(
                f"{name} feature width {arr.shape[1]} != train width "
                f"{x_train.shape[1]} at LT {lead_time}"
            )

    x_full = np.concatenate([x_train, x_test, x_2021], axis=0)
    y_full = np.concatenate([y_train, y_test, y_2021], axis=0)

    print(
        f"  [phase5] LT {lead_time}: train {x_train.shape[0]} "
        f"+ test {x_test.shape[0]} + 2021 {x_2021.shape[0]} "
        f"= {x_full.shape[0]} rows | val {x_val.shape[0]} (early stopping)",
        flush=True,
    )

    # x_test/y_test pass through as the ORIGINAL test arrays so the 6-tuple
    # contract holds and scale=True's len>=6 check passes. They are part of
    # training now too; any "test" metric phase 5 reports is in-sample.
    result = (x_full, y_full, x_val, y_val, x_test, y_test)
    _ARRAY_CACHE[cache_key] = result
    return result


class Phase5Tuner(MAPETuner):
    """MAPETuner that trains on train + test + 2021, validating on val."""

    def _prepare_arrays(self, prep_kwargs: dict, config: dict):
        return phase5_arrays(prep_kwargs)
