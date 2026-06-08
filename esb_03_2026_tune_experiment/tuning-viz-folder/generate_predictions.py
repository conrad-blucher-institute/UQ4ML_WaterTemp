"""
Incremental batch prediction generator.

Loads .keras models from a results folder, runs predictions on training,
testing, validation, and 2021 independent test data, saves per-model CSVs + an index manifest.

Usage:
    python generate_predictions.py          # GUI folder picker
    python generate_predictions.py --folder path/to/mape_results_run1

Skips models that already have predictions in the predictions/ subfolder.
"""

import argparse
import glob
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root so we can import src/ ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = (_SCRIPT_DIR / '../..').resolve()
sys.path.insert(0, str(_REPO_ROOT))

from src.helper.utils_mse_crps import (
    readingData,
    creatingAdditionalColumns,
    splittingData,
    deletingMissingValues,
    dateTimeRetriever,
    prepare_independent_year,
)

_DEFAULT_RESULTS = str((_SCRIPT_DIR / '../../results').resolve())
_DATA_PATH = str(_REPO_ROOT / 'data' / 'ESB_datasets')

# Tuner defaults (must match what the tuner uses)
_INPUT_STRUCTURE = 'descending'
_ATP_HOURS_BACK = 24
_WTP_HOURS_BACK = 24
_PRED_ATP_INTERVAL = 1
_IPP_OFFSET = 0.0


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select results folder (containing keras_files/)",
        initialdir=_DEFAULT_RESULTS,
    )
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)


def parse_keras_filename(name):
    """Parse model config from filename.

    Pattern: {type}_{lt}h_cycle{c}_{activation}_{layers}L_{neurons}N_run{r}.keras
    Example: MAPE_12h_cycle0_leaky_relu_1L_100N_run1.keras
    """
    stem = Path(name).stem  # strip .keras
    m = re.match(
        r'^(\w+?)_(\d+)h_cycle(\d+)_(.+?)_(\d+)L_(\d+)N_run(\d+)$',
        stem
    )
    if not m:
        return None
    return {
        'model_type': m.group(1),
        'leadtime': int(m.group(2)),
        'cycle': int(m.group(3)),
        'activation': m.group(4),
        'num_layers': int(m.group(5)),
        'neurons': int(m.group(6)),
        'run_num': int(m.group(7)),
        'basename': stem,
    }


def prepare_val_data(leadtime, cycle):
    """Run the full data pipeline and return (X_train, y_train, train_dates,
    X_test, y_test, test_dates, X_val, y_val, val_dates) for a given leadtime+cycle."""
    # Read years 2-5 (skipping year 1 = 2021 independent)
    year_dfs = readingData(_DATA_PATH)
    years = [
        creatingAdditionalColumns(
            df, _INPUT_STRUCTURE, leadtime,
            _ATP_HOURS_BACK, _WTP_HOURS_BACK,
            _PRED_ATP_INTERVAL, _IPP_OFFSET
        )
        for df in year_dfs
    ]
    # Split by cycle
    training_df, testing_df, validation_df = splittingData(years[0], years[1], years[2], years[3], cycle)
    # Clean
    training_clean = deletingMissingValues(training_df)
    testing_clean = deletingMissingValues(testing_df)
    validation_clean = deletingMissingValues(validation_df)
    # Dates
    train_dates = dateTimeRetriever(training_clean.copy(), leadtime)
    test_dates = dateTimeRetriever(testing_clean.copy(), leadtime)
    val_dates = dateTimeRetriever(validation_clean.copy(), leadtime)

    col_start = 1 if _INPUT_STRUCTURE == 'descending' else 3
    X_train = training_clean.iloc[:, col_start:-1].values.astype(float)
    y_train = training_clean.iloc[:, -1].values.astype(float)
    X_test = testing_clean.iloc[:, col_start:-1].values.astype(float)
    y_test = testing_clean.iloc[:, -1].values.astype(float)
    X_val = validation_clean.iloc[:, col_start:-1].values.astype(float)
    y_val = validation_clean.iloc[:, -1].values.astype(float)
    return X_train, y_train, train_dates, X_test, y_test, test_dates, X_val, y_val, val_dates


def prepare_2021_data(leadtime):
    """Return (X_2021, y_2021, dates_2021) for the independent test year."""
    csv_files = sorted(glob.glob(os.path.join(_DATA_PATH, '*.csv')))
    X, y, dates = prepare_independent_year(
        csv_path=csv_files[0],
        input_structure=_INPUT_STRUCTURE,
        lead_time=leadtime,
        atp_hours_back=_ATP_HOURS_BACK,
        wtp_hours_back=_WTP_HOURS_BACK,
        pred_atp_interval=_PRED_ATP_INTERVAL,
        IPPOffset=_IPP_OFFSET,
    )
    return X, y, dates


def main(folder):
    keras_dir = folder / 'keras_files'
    pred_dir = folder / 'predictions'
    pred_dir.mkdir(exist_ok=True)

    keras_files = sorted(glob.glob(str(keras_dir / '*.keras')))
    if not keras_files:
        print(f"No .keras files found in {keras_dir}")
        sys.exit(1)

    # Parse all model configs
    models = []
    for kf in keras_files:
        cfg = parse_keras_filename(os.path.basename(kf))
        if cfg is None:
            print(f"  SKIP (can't parse filename): {os.path.basename(kf)}")
            continue
        cfg['keras_path'] = kf
        cfg['pred_path'] = str(pred_dir / f"{cfg['basename']}_pred.csv")
        models.append(cfg)

    # Check which already have predictions (with train+test datasets)
    def _needs_regen(pred_path):
        """Return True if file is missing or lacks train/test datasets."""
        if not os.path.exists(pred_path):
            return True
        try:
            df = pd.read_csv(pred_path, usecols=['dataset'], nrows=1000)
            datasets = set(df['dataset'].unique())
            return not {'train', 'test'}.issubset(datasets)
        except Exception:
            return True

    todo = [m for m in models if _needs_regen(m['pred_path'])]
    skip_count = len(models) - len(todo)
    print(f"Found {len(models)} models, {skip_count} already up-to-date, {len(todo)} to process")

    if not todo:
        print("All predictions up to date.")
    else:
        # Group by (leadtime, cycle) to reuse data prep
        from collections import defaultdict
        groups = defaultdict(list)
        for m in todo:
            groups[(m['leadtime'], m['cycle'])].append(m)

        # Lazy import tensorflow
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        # Cache 2021 data per leadtime
        cache_2021 = {}
        # Cache val data per (leadtime, cycle)
        cache_val = {}

        total_done = 0
        for (lt, cy), group_models in groups.items():
            print(f"\n--- Leadtime {lt}h, Cycle {cy} ({len(group_models)} models) ---")

            # Prepare data (cached)
            if lt not in cache_2021:
                print(f"  Preparing 2021 data for leadtime={lt}h...")
                cache_2021[lt] = prepare_2021_data(lt)
            X_2021, y_2021, dates_2021 = cache_2021[lt]

            if (lt, cy) not in cache_val:
                print(f"  Preparing train/test/val data for leadtime={lt}h, cycle={cy}...")
                cache_val[(lt, cy)] = prepare_val_data(lt, cy)
            X_train, y_train, train_dates, X_test, y_test, test_dates, X_val, y_val, val_dates = cache_val[(lt, cy)]

            # Pre-convert to tensors once per group (avoids re-creating each model)
            X_train_t = tf.constant(X_train, dtype=tf.float32)
            X_test_t = tf.constant(X_test, dtype=tf.float32)
            X_val_t = tf.constant(X_val, dtype=tf.float32)
            X_2021_t = tf.constant(X_2021, dtype=tf.float32)

            for m in group_models:
                basename = m['basename']
                print(f"  Predicting: {basename}")

                try:
                    model = tf.keras.models.load_model(m['keras_path'], compile=False)
                    y_pred_train = model(X_train_t, training=False).numpy().flatten()
                    y_pred_test = model(X_test_t, training=False).numpy().flatten()
                    y_pred_val = model(X_val_t, training=False).numpy().flatten()
                    y_pred_2021 = model(X_2021_t, training=False).numpy().flatten()

                    # Build CSV
                    rows = []
                    for i in range(len(y_train)):
                        rows.append({
                            'date': str(train_dates[i]),
                            'actual': float(y_train[i]),
                            'predicted': float(y_pred_train[i]),
                            'dataset': 'train',
                        })
                    for i in range(len(y_test)):
                        rows.append({
                            'date': str(test_dates[i]),
                            'actual': float(y_test[i]),
                            'predicted': float(y_pred_test[i]),
                            'dataset': 'test',
                        })
                    for i in range(len(y_val)):
                        rows.append({
                            'date': str(val_dates[i]),
                            'actual': float(y_val[i]),
                            'predicted': float(y_pred_val[i]),
                            'dataset': 'val',
                        })
                    for i in range(len(y_2021)):
                        rows.append({
                            'date': str(dates_2021[i]),
                            'actual': float(y_2021[i]),
                            'predicted': float(y_pred_2021[i]),
                            'dataset': '2021',
                        })

                    df = pd.DataFrame(rows)
                    df.to_csv(m['pred_path'], index=False)
                    total_done += 1

                except Exception as e:
                    print(f"    ERROR: {e}")
                finally:
                    tf.keras.backend.clear_session()

        print(f"\nDone. Generated {total_done} prediction files.")

    # Generate air_temp.csv lookup from all raw CSVs
    print("\nBuilding air_temp.csv lookup...")
    csv_files = sorted(glob.glob(os.path.join(_DATA_PATH, '*.csv')))
    air_frames = []
    for cf in csv_files:
        raw = pd.read_csv(cf)
        raw['date'] = pd.to_datetime(raw['date'], utc=True)
        air_frames.append(raw[['date', 'Air Average']].copy())
    air_df = pd.concat(air_frames, ignore_index=True).drop_duplicates(subset='date').sort_values('date')
    air_df.columns = ['date', 'air_temp']
    air_df['date'] = air_df['date'].astype(str)
    air_path = str(pred_dir / 'air_temp.csv')
    air_df.to_csv(air_path, index=False)
    print(f"Air temp lookup: {len(air_df)} rows -> {air_path}")

    # Build/rebuild index manifest from ALL prediction files
    print("\nBuilding _index.csv manifest...")
    progress_csvs = glob.glob(str(folder / '*_progress.csv'))
    metrics_lookup = {}
    for pc in progress_csvs:
        try:
            pdf = pd.read_csv(pc)
            # Normalize column names
            col_map = {}
            for c in pdf.columns:
                cl = c.strip().lower()
                if cl == 'run_num':
                    col_map[c] = 'run_num'
                elif cl == 'lead_time':
                    col_map[c] = 'leadtime'
                elif cl == 'model_type':
                    col_map[c] = 'model_type'
            pdf.rename(columns=col_map, inplace=True)

            for _, row in pdf.iterrows():
                if str(row.get('status', '')).strip() != 'completed':
                    continue
                # Build the same basename the tuner uses
                key = (
                    f"{row.get('model_type', 'MAPE')}_{int(row.get('leadtime', 0))}h"
                    f"_cycle{int(row.get('cycle', 0))}_{row.get('activation', '')}"
                    f"_{int(row.get('num_layers', 0))}L_{int(row.get('neurons', 0))}N"
                    f"_run{int(row.get('run_num', 1))}"
                )
                metrics_lookup[key] = {
                    'val_mae': row.get('val_mae', None),
                    'val_mae12': row.get('val_mae12', None),
                    'mae_2021': row.get('mae_2021', None),
                    'mae12_2021': row.get('mae12_2021', None),
                }
        except Exception as e:
            print(f"  Warning: could not read {pc}: {e}")

    pred_files = sorted(glob.glob(str(pred_dir / '*_pred.csv')))
    index_rows = []
    for pf in pred_files:
        cfg = parse_keras_filename(os.path.basename(pf).replace('_pred.csv', '.keras'))
        if cfg is None:
            continue
        row = {
            'model_name': cfg['basename'],
            'activation': cfg['activation'],
            'num_layers': cfg['num_layers'],
            'neurons': cfg['neurons'],
            'leadtime': cfg['leadtime'],
            'cycle': cfg['cycle'],
            'run_num': cfg['run_num'],
        }
        m = metrics_lookup.get(cfg['basename'], {})
        row.update(m)
        index_rows.append(row)

    index_df = pd.DataFrame(index_rows)
    index_path = str(pred_dir / '_index.csv')
    index_df.to_csv(index_path, index=False)
    print(f"Index: {len(index_rows)} models → {index_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions from .keras models")
    parser.add_argument('--folder', type=str, default=None,
                        help="Path to results folder (skip GUI picker)")
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
    else:
        folder = pick_folder()

    print(f"Results folder: {folder}")
    main(folder)
