"""
Benchmark: measure per-model prediction time with and without clear_session().
Runs on the first 10 models in keras_files/, times each step.
"""

import sys
import glob
import os
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = (_SCRIPT_DIR / '../..').resolve()
sys.path.insert(0, str(_REPO_ROOT))

_DATA_PATH = str(_REPO_ROOT / 'data' / 'ESB_datasets')

# Prepare 2021 data once (use leadtime=12 as representative)
from src.helper.utils_mse_crps import prepare_independent_year

csv_files = sorted(glob.glob(os.path.join(_DATA_PATH, '*.csv')))
print("Preparing 2021 data (leadtime=12)...")
t0 = time.perf_counter()
X_2021, y_2021, dates_2021 = prepare_independent_year(
    csv_path=csv_files[0],
    input_structure='descending',
    lead_time=12,
    atp_hours_back=24,
    wtp_hours_back=24,
    pred_atp_interval=1,
    IPPOffset=0.0,
)
print(f"Data prep: {time.perf_counter() - t0:.2f}s, X shape: {X_2021.shape}")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Find first 10 keras files
keras_dir = _REPO_ROOT / 'results' / 'esb_tuner' / 'mape_results_run1' / 'keras_files'
keras_files = sorted(glob.glob(str(keras_dir / '*12h*cycle0*.keras')))[:10]
print(f"\nBenchmarking {len(keras_files)} models\n")

# ── Run 1: WITH clear_session ──
print("=" * 60)
print("RUN 1: WITH clear_session()")
print("=" * 60)
times_with = []
for kf in keras_files:
    name = os.path.basename(kf)
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(kf, compile=False)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_2021, verbose=0)
    t_pred = time.perf_counter() - t0

    t0 = time.perf_counter()
    tf.keras.backend.clear_session()
    t_clear = time.perf_counter() - t0

    t_all = time.perf_counter() - t_total
    times_with.append({'total': t_all, 'load': t_load, 'predict': t_pred, 'clear': t_clear})
    print(f"  {name:55s}  total={t_all:.2f}s  load={t_load:.2f}s  predict={t_pred:.2f}s  clear={t_clear:.2f}s")

# ── Run 2: WITHOUT clear_session ──
print()
print("=" * 60)
print("RUN 2: WITHOUT clear_session()")
print("=" * 60)
# Do one clear before starting fresh
tf.keras.backend.clear_session()

times_without = []
for kf in keras_files:
    name = os.path.basename(kf)
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(kf, compile=False)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_2021, verbose=0)
    t_pred = time.perf_counter() - t0

    t_all = time.perf_counter() - t_total
    times_without.append({'total': t_all, 'load': t_load, 'predict': t_pred})
    print(f"  {name:55s}  total={t_all:.2f}s  load={t_load:.2f}s  predict={t_pred:.2f}s")

# ── Summary ──
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

avg_with = sum(t['total'] for t in times_with) / len(times_with)
avg_without = sum(t['total'] for t in times_without) / len(times_without)
avg_clear = sum(t['clear'] for t in times_with) / len(times_with)
avg_load_with = sum(t['load'] for t in times_with) / len(times_with)
avg_load_without = sum(t['load'] for t in times_without) / len(times_without)
avg_pred_with = sum(t['predict'] for t in times_with) / len(times_with)
avg_pred_without = sum(t['predict'] for t in times_without) / len(times_without)

# ── Run 3: Direct call model(X) instead of model.predict(X) ──
print()
print("=" * 60)
print("RUN 3: model(X, training=False) — no clear_session")
print("=" * 60)
tf.keras.backend.clear_session()

times_direct = []
for kf in keras_files:
    name = os.path.basename(kf)
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(kf, compile=False)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model(tf.constant(X_2021, dtype=tf.float32), training=False).numpy()
    t_pred = time.perf_counter() - t0

    t_all = time.perf_counter() - t_total
    times_direct.append({'total': t_all, 'load': t_load, 'predict': t_pred})
    print(f"  {name:55s}  total={t_all:.2f}s  load={t_load:.2f}s  predict={t_pred:.2f}s")

avg_direct = sum(t['total'] for t in times_direct) / len(times_direct)
avg_load_direct = sum(t['load'] for t in times_direct) / len(times_direct)
avg_pred_direct = sum(t['predict'] for t in times_direct) / len(times_direct)

# ── Run 4: Build architecture once, load_weights only ──
print()
print("=" * 60)
print("RUN 4: build arch once + load_weights + direct call")
print("=" * 60)
tf.keras.backend.clear_session()

import re
def parse_keras_filename(name):
    stem = Path(name).stem
    m = re.match(r'^(\w+?)_(\d+)h_cycle(\d+)_(.+?)_(\d+)L_(\d+)N_run(\d+)$', stem)
    if not m:
        return None
    return {
        'activation': m.group(4), 'num_layers': int(m.group(5)),
        'neurons': int(m.group(6)), 'leadtime': int(m.group(2)),
    }

def build_architecture(activation, num_layers, neurons, input_dim):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(Dense(units=neurons, activation=activation))
    model.add(Dense(1, activation='linear'))
    return model

# Group benchmark files by architecture
from collections import defaultdict
arch_groups = defaultdict(list)
for kf in keras_files:
    cfg = parse_keras_filename(os.path.basename(kf))
    key = (cfg['activation'], cfg['num_layers'], cfg['neurons'])
    arch_groups[key].append(kf)

times_weights = []
input_dim = X_2021.shape[1]
for (act, nl, nn), files in arch_groups.items():
    t0 = time.perf_counter()
    arch_model = build_architecture(act, nl, nn, input_dim)
    t_build = time.perf_counter() - t0
    print(f"  Built arch {act}/{nl}L/{nn}N in {t_build:.3f}s ({len(files)} models)")

    for kf in files:
        name = os.path.basename(kf)
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        arch_model.load_weights(kf)
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = arch_model(tf.constant(X_2021, dtype=tf.float32), training=False).numpy()
        t_pred = time.perf_counter() - t0

        t_all = time.perf_counter() - t_total
        times_weights.append({'total': t_all, 'load': t_load, 'predict': t_pred})
        print(f"    {name:53s}  total={t_all:.2f}s  load={t_load:.2f}s  predict={t_pred:.2f}s")

avg_weights = sum(t['total'] for t in times_weights) / len(times_weights)
avg_load_weights = sum(t['load'] for t in times_weights) / len(times_weights)
avg_pred_weights = sum(t['predict'] for t in times_weights) / len(times_weights)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"{'':30s}  {'1:WITH clr':>12s}  {'2:NO clr':>12s}  {'3:DIRECT':>12s}  {'4:WEIGHTS':>12s}")
print(f"{'Avg total/model':30s}  {avg_with:11.2f}s  {avg_without:11.2f}s  {avg_direct:11.2f}s  {avg_weights:11.2f}s")
print(f"{'Avg load':30s}  {avg_load_with:11.2f}s  {avg_load_without:11.2f}s  {avg_load_direct:11.2f}s  {avg_load_weights:11.2f}s")
print(f"{'Avg predict':30s}  {avg_pred_with:11.2f}s  {avg_pred_without:11.2f}s  {avg_pred_direct:11.2f}s  {avg_pred_weights:11.2f}s")
print(f"{'Avg clear_session':30s}  {avg_clear:11.2f}s  {'N/A':>12s}  {'N/A':>12s}  {'N/A':>12s}")
print()
best = avg_weights
print(f"Speedup vs current (Run 1):  {avg_with / best:.1f}x  ({avg_with - best:+.2f}s/model)")
print(f"\nFor 700 models:")
print(f"  1: WITH clear_session:         ~{avg_with * 700 / 60:.0f} min")
print(f"  2: NO clear_session:           ~{avg_without * 700 / 60:.0f} min")
print(f"  3: Direct call:                ~{avg_direct * 700 / 60:.0f} min")
print(f"  4: Build once + load_weights:  ~{avg_weights * 700 / 60:.0f} min")
