"""In-graph scaling tests — the Normalization layer must BE the StandardScaler.

The whole design rests on one claim: moving the transform from the arrays into a
frozen first layer changes where scaling happens, not what it computes. If that
claim is false the failure is silent (a model that trains fine and predicts
plausible-but-wrong temperatures), so it gets proven here rather than assumed.

Covered:
  1. ``normalization_stats`` reproduces ``StandardScaler.transform`` exactly.
  2. The zero-variance feature — the one case where ``scale_**2`` and ``var_``
     diverge, and using ``var_`` would blow a constant column up by ~1e7.
  3. ``feature_names`` width matches the documented W+1+A+1+F/I formula and the
     documented ordering; a non-descending structure raises instead of guessing.
  4. Round-trip: a saved+reloaded model still scales (TF only; skipped without).

Run (from repo root):
    python -m esb.tests.test_scaler_layer
or under pytest:
    pytest esb/tests/test_scaler_layer.py
"""
from __future__ import annotations

import numpy as np

from esb.stages.features import feature_names
from esb.stages.scale import fit_scaler, normalization_stats


def _fit(x):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(x)


def test_stats_reproduce_transform():
    """(x - mean)/sqrt(variance) from our stats == scaler.transform(x)."""
    rng = np.random.default_rng(0)
    x = rng.normal(15.0, 6.0, size=(400, 9))
    scaler = _fit(x)
    mean, variance = normalization_stats(scaler)

    probe = rng.uniform(-40.0, 60.0, size=(50, 9))
    ours = (probe - mean) / np.sqrt(variance)
    theirs = scaler.transform(probe)
    assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-5), \
        f"max diff {np.max(np.abs(ours - theirs))}"


def test_zero_variance_feature_uses_scale_not_var():
    """A constant column must pass through unscaled, not explode.

    sklearn guards this by setting scale_ = 1.0 while var_ stays 0.0. Squaring
    scale_ preserves the guard; handing var_ to Keras would divide by ~1e-7.
    """
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(20.0, 3.0, size=300),
                         np.full(300, 7.5)])          # zero-variance column
    scaler = _fit(x)
    mean, variance = normalization_stats(scaler)

    assert scaler.var_[1] == 0.0, "precondition: sklearn reports zero variance"
    assert variance[1] == 1.0, "scale_**2 must restore the zero-variance guard"

    probe = np.column_stack([rng.normal(20.0, 3.0, size=10), np.full(10, 7.5)])
    ours = (probe - mean) / np.sqrt(variance)
    assert np.allclose(ours, scaler.transform(probe), atol=1e-6)
    assert np.all(np.abs(ours[:, 1]) < 1e-6), "constant column must map to ~0"


def test_fit_scaler_is_leakage_checked():
    """fit_scaler must reject a scaler that saw anything but the training rows."""
    rng = np.random.default_rng(2)
    x_train = rng.normal(10.0, 2.0, size=(200, 4))
    scaler = fit_scaler(x_train)
    assert np.allclose(scaler.mean_, x_train.mean(axis=0))

    from esb.stages.scale import _assert_fit_on_train_only
    leaked = _fit(np.vstack([x_train, rng.normal(80.0, 5.0, size=(50, 4))]))
    try:
        _assert_fit_on_train_only(leaked, x_train)
    except AssertionError:
        pass
    else:
        raise AssertionError("leakage check failed to fire on a leaked scaler")


def test_feature_names_width_and_order():
    """Width equals W+1+A+1+F/I and the documented ordering holds."""
    names = feature_names("descending", input_hours_forecast=12,
                          atp_hours_back=24, wtp_hours_back=24,
                          pred_atp_interval=1)
    assert len(names) == 24 + 1 + 24 + 1 + 12 == 62
    assert names[0] == "waterTemperature__24h_ago", "water lags largest-first"
    assert names[23] == "waterTemperature__1h_ago"
    assert names[24] == "Water Average"
    assert names[25] == "airTemperature__24h_ago", "air lags largest-first"
    assert names[49] == "Air Average"
    assert names[50] == "airTemperature_pred__1h_forecast", "forecasts ascend"
    assert names[-1] == "airTemperature_pred__12h_forecast"
    assert len(set(names)) == len(names), "names must be unique"

    # LT120 — the widest config in the grid.
    assert len(feature_names("descending", 120, 24, 24, 1)) == 170


def test_feature_names_refuses_to_guess():
    """A structure we have not verified must raise, not emit a plausible list."""
    for bad in ("ascending", "whatever"):
        try:
            feature_names(bad, 12, 24, 24, 1)
        except NotImplementedError:
            continue
        raise AssertionError(f"{bad!r} should have raised NotImplementedError")

    try:
        feature_names("descending", input_hours_forecast=13,
                      atp_hours_back=24, wtp_hours_back=24,
                      pred_atp_interval=6)      # 13 not a multiple of 6
    except ValueError:
        pass
    else:
        raise AssertionError("non-multiple forecast interval should raise")


def test_layer_round_trip_survives_save_load():
    """A saved+reloaded model still scales — the point of the whole exercise."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        print("  SKIP round-trip test (no TensorFlow here)")
        return

    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(3)
    x = rng.normal(18.0, 5.0, size=(300, 5))
    scaler = _fit(x)
    mean, variance = normalization_stats(scaler)

    inp = keras.Input(shape=(5,))
    out = keras.layers.Dense(1)(keras.layers.Normalization(
        axis=-1, mean=mean, variance=variance)(inp))
    model = keras.Model(inp, out)

    probe = rng.uniform(-30.0, 50.0, size=(16, 5)).astype("float32")
    before = model.predict(probe, verbose=0)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.keras"
        model.save(p)
        # compile=False is the incantation ops should use — no loss lookup.
        reloaded = keras.models.load_model(p, compile=False)
        after = reloaded.predict(probe, verbose=0)

    assert np.allclose(before, after, atol=1e-5), "scaling lost across save/load"


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    _main()
