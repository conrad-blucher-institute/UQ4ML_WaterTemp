"""scale stage — fits the scaler; the MODEL applies it (Stage C, decision C1/C2/C3).

Reconciles the two pre-existing StandardScaler implementations into one
leakage-safe, opt-in stage (design doc §6). The scaling LOGIC is lifted from
``origin/esb_dev_normalization`` (``StandardScaler`` fit-on-train-only).

Contract: ``Arrays -> scaler``. The stage no longer transforms anything. Its
statistics become a **frozen** ``keras.layers.Normalization`` layer inside the
model, so the arrays reaching ``model.fit`` are RAW and the saved ``.keras``
scales its own inputs. That is what makes the artifact self-contained: no
``_scaler.joblib`` for a consumer to locate, pair with the right model, and apply
in the right column order.

Because the layer is non-trainable and deterministic, this is a relocation of the
transform, not a change to the training math — same forward pass, same gradients.

Guardrails (design doc §6):
  * **Leakage is now possible.** ``fit`` runs on ``x_train`` ONLY. :func:`fit_scaler`
    asserts this explicitly (``scaler.mean_`` must equal the training-set column
    means), so a future edit that accidentally fits on more than the training data
    fails loudly.
  * **Opt-in, OFF by default (C1).** With scaling off no scaler is fitted and no
    layer is added — byte-identical to the unscaled path, so the Stage B golden
    digest d80ae54… stays valid.
  * **``variance = scale_ ** 2``, never ``var_``** — see :func:`normalization_stats`.
"""
from __future__ import annotations

import numpy as np


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


def fit_scaler(x_train: np.ndarray):
    """Fit a StandardScaler on ``x_train`` and leakage-check it. Single fit site.

    The ONE place a scaler is fitted. Callers (the tuner and the golden-digest
    verifier) get the leakage guard for free; nobody transforms arrays, because
    the model's Normalization layer does that in-graph.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)                        # TRAIN ONLY — never val/test
    _assert_fit_on_train_only(scaler, x_train)
    return scaler


def normalization_stats(scaler) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, variance)`` for a ``keras.layers.Normalization`` built from ``scaler``.

    Keras computes ``(x - mean) / max(sqrt(variance), epsilon)``. Squaring
    ``scale_`` — NOT returning ``var_`` — is what makes that identical to
    ``StandardScaler.transform``: ``scale_`` carries sklearn's zero-variance
    guard (a constant feature gets ``scale_ = 1.0``, while ``var_`` stays 0.0,
    which would divide by ~epsilon and blow the feature up).
    """
    mean = np.asarray(scaler.mean_, dtype="float32")
    variance = np.asarray(scaler.scale_, dtype="float32") ** 2
    return mean, variance


# scale_arrays() was DELETED. It fitted a scaler and transformed the arrays,
# returning ScaledArrays. Scaling now happens INSIDE the model (a frozen
# Normalization layer built from fit_scaler + normalization_stats above), so the
# arrays that reach model.fit are raw and there is nothing left to transform. The
# stage contract is therefore ``Arrays -> scaler`` rather than
# ``Arrays -> ScaledArrays``; ScaledArrays went with it.
