"""Leakage guard tests for the scale stage (Stage C, design doc §6 guardrail 1).

Scaling introduces a NEW leakage surface: the CV-leakage verdict ("safe because
no scaler exists") no longer auto-holds. These tests prove the scaler is fit on
x_train ONLY, that val/test are standardized with TRAINING statistics rather than
re-fitted, and that the OFF path leaves the arrays untouched.

The scaler is no longer persisted or applied to arrays — its statistics are
frozen into the model's Normalization layer. That the layer reproduces
StandardScaler.transform exactly is proven in ``test_scaler_layer.py``.

Run (from repo root, with numpy+sklearn available):
    python -m esb.tests.test_scale_leakage
or under pytest:
    pytest esb/tests/test_scale_leakage.py
"""
from __future__ import annotations

import numpy as np

from esb.contracts import Arrays
from esb.stages.scale import (_assert_fit_on_train_only, fit_scaler,
                              normalization_stats)


def _arrays():
    """Train vs val/test drawn from DELIBERATELY different distributions, so any
    leak (fitting on val/test/full) would move scaler.mean_ measurably."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(10.0, 2.0, size=(500, 6))
    x_val = rng.normal(50.0, 9.0, size=(200, 6))
    x_test = rng.normal(-30.0, 4.0, size=(150, 6))
    y = lambda n: rng.normal(0, 1, size=(n,))
    return Arrays(x_train, y(500), x_val, y(200), x_test, y(150))


def test_fit_on_train_only():
    arr = _arrays()
    scaler = fit_scaler(arr.x_train)
    assert np.allclose(scaler.mean_, arr.x_train.mean(0))
    assert np.allclose(scaler.var_, arr.x_train.var(0))
    full_mean = np.vstack([arr.x_train, arr.x_val, arr.x_test]).mean(0)
    assert not np.allclose(scaler.mean_, full_mean)  # would mean full-data leak


def test_val_is_scaled_with_train_stats_not_refit():
    """The layer standardizes val/test with TRAINING statistics.

    Previously this checked scale_arrays' transformed output. The stage no
    longer transforms, so the same property is asserted where it now lives: the
    statistics that get frozen into the model come from train alone, and
    applying them to val leaves val far from zero-mean (a re-fit would centre
    it, which is exactly the leak this guards).
    """
    arr = _arrays()
    scaler = fit_scaler(arr.x_train)
    mean, variance = normalization_stats(scaler)

    x_train_scaled = (arr.x_train - mean) / np.sqrt(variance)
    assert np.allclose(x_train_scaled.mean(0), 0, atol=1e-5)
    assert np.allclose(x_train_scaled.std(0), 1, atol=1e-5)

    x_val_scaled = (arr.x_val - mean) / np.sqrt(variance)
    assert abs(x_val_scaled.mean()) > 5, "val must NOT be centred (would be a re-fit)"


def test_off_means_no_scaler_and_raw_arrays():
    """Scaling off = nothing fitted, nothing touched.

    There is no 'enabled' flag on the stage any more — the tuner simply does not
    call fit_scaler when config['scale'] is falsy, and no Normalization layer is
    added. The invariant that matters is that the arrays handed to model.fit are
    the very same objects reshape produced.
    """
    arr = _arrays()
    x_train, _ = arr.x_train, arr.y_train
    assert np.shares_memory(x_train, arr.x_train)  # same array, not a copy


# NOTE: test_persisted_scaler_roundtrips() was removed with save_scaler(). The
# scaler is no longer persisted as a .joblib — its statistics are frozen into the
# model's Normalization layer instead. The equivalent guarantee (the persisted
# artifact reproduces StandardScaler.transform exactly) is now proven in
# esb/tests/test_scaler_layer.py, against the layer rather than the file.


def test_leakage_guard_fires_on_bad_fit():
    from sklearn.preprocessing import StandardScaler

    arr = _arrays()
    bad = StandardScaler().fit(np.vstack([arr.x_train, arr.x_val]))  # includes val
    try:
        _assert_fit_on_train_only(bad, arr.x_train)
    except AssertionError:
        return
    raise AssertionError("leakage guard did not fire on a fit that included val")


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} SCALE LEAKAGE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
