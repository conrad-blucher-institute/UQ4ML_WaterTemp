"""scale stage — the first truly cohesive stage (Stage C, decision C1/C2/C3).

Reconciles the two pre-existing StandardScaler implementations into one
leakage-safe, opt-in stage (design doc §6). The scaling LOGIC is lifted from
``origin/esb_dev_normalization`` (``StandardScaler`` fit-on-train-only); the
inference half lives in ``infer`` (``prepare_independent_year(scaler=, column_map=)``).

Contract: ``Arrays -> ScaledArrays + scaler``.

Guardrails (design doc §6):
  * **Leakage is now possible.** ``fit`` runs on ``x_train`` ONLY; ``x_val`` and
    ``x_test`` are transform-only. :func:`scale_arrays` asserts this explicitly
    (``scaler.mean_`` must equal the training-set column means), so a future edit
    that accidentally fits on more than the training data fails loudly.
  * **Opt-in, OFF by default (C1).** ``enabled=False`` returns the same arrays
    with ``scaler=None`` — byte-identical to the unscaled path, so the Stage B
    golden digest d80ae54… stays valid.
"""
from __future__ import annotations

import numpy as np

from esb.contracts import Arrays, ScaledArrays


def _assert_fit_on_train_only(scaler, x_train: np.ndarray) -> None:
    """Loud leakage check: prove the scaler saw ONLY the training rows.

    A StandardScaler fitted on ``x_train`` has ``mean_``/``var_`` equal to that
    array's per-column statistics. If val/test (or the full set) had leaked into
    ``fit``, these would differ. This is the explicit assert the design doc
    requires for the new leakage surface.
    """
    expected_mean = x_train.mean(axis=0)
    expected_var = x_train.var(axis=0)
    if not np.allclose(scaler.mean_, expected_mean, rtol=1e-9, atol=1e-9):
        raise AssertionError(
            "scale leakage check FAILED: scaler.mean_ != x_train column means — "
            "the scaler was fit on something other than the training set."
        )
    if not np.allclose(scaler.var_, expected_var, rtol=1e-9, atol=1e-9):
        raise AssertionError(
            "scale leakage check FAILED: scaler.var_ != x_train column variances — "
            "the scaler was fit on something other than the training set."
        )


def scale_arrays(arrays: Arrays, enabled: bool) -> ScaledArrays:
    """Fit a StandardScaler on ``x_train`` only, transform train/val/test.

    Args:
        arrays: reshape-stage output (raw, unscaled).
        enabled: when False (the default everywhere), this is an identity
            passthrough returning the same arrays with ``scaler=None``.

    Returns:
        ScaledArrays — x arrays standardized (when enabled), the fitted scaler
        (or None), and the auxiliary date/air-temp fields unchanged.
    """
    common = dict(
        y_train=arrays.y_train, y_val=arrays.y_val, y_test=arrays.y_test,
        training_dates=arrays.training_dates,
        validation_dates=arrays.validation_dates,
        testing_dates=arrays.testing_dates,
        testing_air_temps=arrays.testing_air_temps,
    )

    if not enabled:
        return ScaledArrays(
            x_train=arrays.x_train, x_val=arrays.x_val, x_test=arrays.x_test,
            scaler=None, **common,
        )

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(arrays.x_train)                 # TRAIN ONLY — never val/test
    _assert_fit_on_train_only(scaler, arrays.x_train)

    x_train = scaler.transform(arrays.x_train)
    # Empty val/test frames (e.g. some edge configs) have 0 columns; guard the
    # transform so an empty array doesn't trip sklearn's feature-count check.
    x_val = scaler.transform(arrays.x_val) if arrays.x_val.size else arrays.x_val
    x_test = scaler.transform(arrays.x_test) if arrays.x_test.size else arrays.x_test

    return ScaledArrays(
        x_train=x_train, x_val=x_val, x_test=x_test, scaler=scaler, **common,
    )
