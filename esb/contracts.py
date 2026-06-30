"""Typed stage contracts — the only things passed between pipeline stages.

Stage C introduces the first two (design doc §4, decision C3): ``Arrays`` (the raw
x/y arrays out of ``reshape``) and ``ScaledArrays`` (post-``scale``, carrying the
fitted ``scaler`` — ``None`` when scaling is off). They REPLACE the legacy
variable-length return of ``preparingData`` (10 values unscaled / 11 when scaled):
the esb path is now always ``Arrays -> ScaledArrays``, with the scaler a *field*
rather than an extra positional that appears and disappears.

Behavior-preservation: ``Arrays.from_preparing_data`` accepts the existing 10-tuple
unchanged, so nothing in ``preparingData`` (or its other callers, or the Stage B
golden digest) has to move for these to exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class Arrays:
    """Raw model-ready arrays out of the reshape stage (one config/rotation).

    The six x/y arrays are the model.fit inputs. The four auxiliary fields
    (dates + test-year air temps) are carried verbatim so an ``Arrays`` can be
    built losslessly from ``preparingData``'s tuple and threaded onward.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    training_dates: Any = field(default_factory=list)
    validation_dates: Any = field(default_factory=list)
    testing_dates: Any = field(default_factory=list)
    testing_air_temps: Any = field(default_factory=list)

    @classmethod
    def from_preparing_data(cls, data: Sequence) -> "Arrays":
        """Build from ``preparingData(...)``'s existing 10-tuple (unscaled path).

        Order matches ``utils_mse_crps.preparingData``'s return:
        ``x_train, y_train, x_val, y_val, x_test, y_test, training_dates,
        validation_dates, testingDates, testingAirTemps``.
        Pandas frames are coerced to float numpy arrays (matching what the tuner
        already feeds ``model.fit``).
        """
        if len(data) < 6:
            raise ValueError(
                f"preparingData returned {len(data)} values; need >= 6 "
                "(x_train, y_train, x_val, y_val, x_test, y_test, ...)."
            )

        def _arr(a):
            return np.asarray(getattr(a, "values", a), dtype=float)

        aux = list(data[6:10]) + [[]] * (4 - len(data[6:10]))
        return cls(
            x_train=_arr(data[0]), y_train=_arr(data[1]),
            x_val=_arr(data[2]), y_val=_arr(data[3]),
            x_test=_arr(data[4]), y_test=_arr(data[5]),
            training_dates=aux[0], validation_dates=aux[1],
            testing_dates=aux[2], testing_air_temps=aux[3],
        )


@dataclass
class ScaledArrays:
    """Arrays after the scale stage. ``scaler`` is ``None`` when scaling is off.

    Same six x/y arrays as :class:`Arrays` (x already standardized when
    ``scaler`` is not None), plus the fitted scaler so the io site can persist it
    and the infer stage can apply the SAME transform to the independent year.
    The auxiliary date/air-temp fields pass through untouched.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: Any = None
    training_dates: Any = field(default_factory=list)
    validation_dates: Any = field(default_factory=list)
    testing_dates: Any = field(default_factory=list)
    testing_air_temps: Any = field(default_factory=list)

    @property
    def scaled(self) -> bool:
        """True when a scaler was fitted (scaling was on)."""
        return self.scaler is not None
