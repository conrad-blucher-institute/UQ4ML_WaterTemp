"""Leakage guard tests for the scale stage (Stage C, design doc §6 guardrail 1).

Scaling introduces a NEW leakage surface: the CV-leakage verdict ("safe because
no scaler exists") no longer auto-holds. These tests prove the scaler is fit on
x_train ONLY, that val/test are transform-only, that the OFF path is a true
identity passthrough, and that the persisted .joblib reproduces the exact
transform the infer stage applies.

Run (from repo root, with numpy+sklearn available):
    python -m esb.tests.test_scale_leakage
or under pytest:
    pytest esb/tests/test_scale_leakage.py
"""
from __future__ import annotations

import numpy as np

from esb.contracts import Arrays
from esb.stages.scale import scale_arrays, _assert_fit_on_train_only


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
    sc = scale_arrays(arr, enabled=True)
    assert np.allclose(sc.scaler.mean_, arr.x_train.mean(0))
    assert np.allclose(sc.scaler.var_, arr.x_train.var(0))
    full_mean = np.vstack([arr.x_train, arr.x_val, arr.x_test]).mean(0)
    assert not np.allclose(sc.scaler.mean_, full_mean)  # would mean full-data leak


def test_val_test_transform_only():
    arr = _arrays()
    sc = scale_arrays(arr, enabled=True)
    # train standardized to ~0 mean / ~1 std
    assert np.allclose(sc.x_train.mean(0), 0, atol=1e-9)
    assert np.allclose(sc.x_train.std(0), 1, atol=1e-6)
    # val transformed with TRAIN stats -> nowhere near 0 mean (not re-fit)
    assert abs(sc.x_val.mean()) > 5
    manual_val = (arr.x_val - arr.x_train.mean(0)) / np.sqrt(arr.x_train.var(0))
    assert np.allclose(sc.x_val, manual_val)


def test_off_is_identity_passthrough():
    arr = _arrays()
    off = scale_arrays(arr, enabled=False)
    assert off.scaler is None
    assert np.shares_memory(off.x_train, arr.x_train)  # same array, not a copy


def test_persisted_scaler_roundtrips():
    import os
    import tempfile
    import joblib
    from esb.io.results import save_scaler

    arr = _arrays()
    sc = scale_arrays(arr, enabled=True)
    d = tempfile.mkdtemp()
    path = save_scaler(d, "unit", sc.scaler)
    assert os.path.basename(path) == "unit_scaler.joblib"
    reloaded = joblib.load(path)
    # exactly the transform the infer path applies to the independent year
    assert np.allclose(reloaded.transform(arr.x_val), sc.x_val)


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
