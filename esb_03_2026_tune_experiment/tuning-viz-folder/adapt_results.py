"""
Adapter: convert coworker's experiment results structure into the tuning-viz format.

Creates a copy of the data in a new folder with suffix '_adapted-to-tuneviz',
restructured to match the flat layout that run_v15.py and generate_predictions.py expect.

Coworker structure (input):
    {source}/
      {lt}h/
        mape-{layers}_layers-{act}-{neurons}_neurons-cycle_{c}-iteration_{i}/
          losses.csv                        (index, Loss, Val_Loss)
          model_*.keras                     (timestamped, possibly multiple)
          train_datetime_obsv_predictions.csv
          val_datetime_obsv_predictions.csv
          test_datetime_obsv_predictions.csv
          2021_datetime_obsv_predictions.csv

Tuning-viz structure (output):
    {source}_adapted-to-tuneviz/
      mape_progress.csv                     (grid-search summary)
      keras_files/
        MAPE_{lt}h_cycle{c}_{act}_{layers}L_{neurons}N_run{i}.keras
        MAPE_{lt}h_cycle{c}_{act}_{layers}L_{neurons}N_run{i}_history.json
      predictions/
        MAPE_{lt}h_cycle{c}_{act}_{layers}L_{neurons}N_run{i}_pred.csv
        _index.csv
        air_temp.csv

Usage:
    python adapt_results.py                          # GUI folder picker
    python adapt_results.py --source path/to/folder  # direct path
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np
import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = (_SCRIPT_DIR / '../..').resolve()
_DEFAULT_RESULTS = str((_REPO_ROOT / 'results').resolve())
_DATA_PATH = str(_REPO_ROOT / 'data' / 'ESB_datasets')
_ADAPTER_SUFFIX = '_adapted-to-HyperView'


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title="Select coworker's results folder to adapt",
        initialdir=_DEFAULT_RESULTS,
    )
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)


def parse_experiment_folder(folder_name):
    """Parse coworker folder name like 'mape-3_layers-relu-256_neurons-cycle_0-iteration_1'.

    Returns dict with layers, activation, neurons, cycle, iteration or None.
    """
    m = re.match(
        r'^mape-(\d+)_layers-(\w+)-(\d+)_neurons-cycle_(\d+)-iteration_(\d+)$',
        folder_name,
    )
    if not m:
        return None
    return {
        'num_layers': int(m.group(1)),
        'activation': m.group(2),
        'neurons': int(m.group(3)),
        'cycle': int(m.group(4)),
        'iteration': int(m.group(5)),
    }


def build_basename(leadtime, cycle, activation, num_layers, neurons, run_num):
    """Build the tuning-viz canonical model name."""
    return f"MAPE_{leadtime}h_cycle{cycle}_{activation}_{num_layers}L_{neurons}N_run{run_num}"


def pick_latest_keras(experiment_dir):
    """Pick the most recently modified .keras file from an experiment folder."""
    keras_files = sorted(
        glob.glob(str(experiment_dir / 'model_*.keras')),
        key=os.path.getmtime,
    )
    return Path(keras_files[-1]) if keras_files else None


def convert_losses_to_history(losses_csv_path):
    """Convert coworker's losses.csv (index, Loss, Val_Loss) to history dict."""
    df = pd.read_csv(losses_csv_path)
    return {
        'loss': df['Loss'].tolist(),
        'val_loss': df['Val_Loss'].tolist(),
    }


def merge_predictions(experiment_dir):
    """Merge coworker's separate val + 2021 prediction CSVs into one unified CSV.

    Input format:  (index), date_time, target, pred_1
    Output format: date, actual, predicted, dataset
    """
    rows = []

    for split_name, dataset_label in [('val', 'val'), ('2021', '2021')]:
        if split_name == '2021':
            csv_path = experiment_dir / '2021_datetime_obsv_predictions.csv'
        else:
            csv_path = experiment_dir / f'{split_name}_datetime_obsv_predictions.csv'

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rows.append({
                'date': str(row['date_time']),
                'actual': float(row['target']),
                'predicted': float(row['pred_1']),
                'dataset': dataset_label,
            })

    return pd.DataFrame(rows)


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def compute_mae12(y_true, y_pred):
    """MAE computed only on samples where y_true <= 12."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true <= 12.0
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_metrics_from_predictions(experiment_dir, losses_csv_path):
    """Compute val_mae, val_mae12, mae_2021, mae12_2021 from prediction CSVs."""
    metrics = {}

    # Validation
    val_csv = experiment_dir / 'val_datetime_obsv_predictions.csv'
    if val_csv.exists():
        df = pd.read_csv(val_csv)
        metrics['val_mae'] = compute_mae(df['target'], df['pred_1'])
        metrics['val_mae12'] = compute_mae12(df['target'], df['pred_1'])

    # 2021
    csv_2021 = experiment_dir / '2021_datetime_obsv_predictions.csv'
    if csv_2021.exists():
        df = pd.read_csv(csv_2021)
        metrics['mae_2021'] = compute_mae(df['target'], df['pred_1'])
        metrics['mae12_2021'] = compute_mae12(df['target'], df['pred_1'])

    # Last val_loss from losses.csv
    if losses_csv_path.exists():
        losses_df = pd.read_csv(losses_csv_path)
        if len(losses_df) > 0:
            metrics['loss_value'] = float(losses_df['Val_Loss'].iloc[-1])
            metrics['loss'] = float(losses_df['Loss'].iloc[-1])
            metrics['val_loss'] = float(losses_df['Val_Loss'].iloc[-1])
            metrics['epochs_trained'] = len(losses_df)

    return metrics


def build_air_temp_csv(pred_dir):
    """Build air_temp.csv from raw ESB data CSVs."""
    csv_files = sorted(glob.glob(os.path.join(_DATA_PATH, '*.csv')))
    if not csv_files:
        print("  WARNING: No ESB data CSVs found, skipping air_temp.csv")
        return

    air_frames = []
    for cf in csv_files:
        raw = pd.read_csv(cf)
        if 'Air Average' in raw.columns:
            raw['date'] = pd.to_datetime(raw['date'], utc=True)
            air_frames.append(raw[['date', 'Air Average']].copy())

    if not air_frames:
        print("  WARNING: No 'Air Average' column found, skipping air_temp.csv")
        return

    air_df = pd.concat(air_frames, ignore_index=True).drop_duplicates(subset='date').sort_values('date')
    air_df.columns = ['date', 'air_temp']
    air_df['date'] = air_df['date'].astype(str)
    air_path = str(pred_dir / 'air_temp.csv')
    air_df.to_csv(air_path, index=False)
    print(f"  Air temp lookup: {len(air_df)} rows -> {air_path}")


def adapt(source_dir):
    source = Path(source_dir)
    dest = source.parent / f"{source.name}{_ADAPTER_SUFFIX}"

    if dest.exists():
        print(f"Destination already exists: {dest}")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(dest)

    keras_dir = dest / 'keras_files'
    pred_dir = dest / 'predictions'
    keras_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)

    # Discover all lead time folders
    lt_dirs = [d for d in source.iterdir() if d.is_dir() and re.match(r'^\d+h$', d.name)]
    if not lt_dirs:
        print(f"No lead-time folders (e.g., 12h, 120h) found in {source}")
        sys.exit(1)

    print(f"Found lead times: {sorted([d.name for d in lt_dirs])}")

    progress_rows = []
    index_rows = []
    total_models = 0

    for lt_dir in sorted(lt_dirs):
        leadtime = int(lt_dir.name.replace('h', ''))
        exp_dirs = [d for d in lt_dir.iterdir() if d.is_dir() and parse_experiment_folder(d.name)]

        for exp_dir in sorted(exp_dirs):
            cfg = parse_experiment_folder(exp_dir.name)
            if cfg is None:
                continue

            basename = build_basename(
                leadtime, cfg['cycle'], cfg['activation'],
                cfg['num_layers'], cfg['neurons'], cfg['iteration'],
            )

            # Copy .keras file
            keras_src = pick_latest_keras(exp_dir)
            if keras_src:
                keras_dest = keras_dir / f"{basename}.keras"
                shutil.copy2(keras_src, keras_dest)
            else:
                print(f"  WARNING: No .keras file in {exp_dir.name}, skipping model copy")

            # Convert losses.csv -> history.json
            losses_csv = exp_dir / 'losses.csv'
            if losses_csv.exists():
                history = convert_losses_to_history(losses_csv)
                history_path = keras_dir / f"{basename}_history.json"
                with open(history_path, 'w') as f:
                    json.dump(history, f)

            # Merge predictions
            pred_df = merge_predictions(exp_dir)
            if len(pred_df) > 0:
                pred_path = pred_dir / f"{basename}_pred.csv"
                pred_df.to_csv(pred_path, index=False)

            # Compute metrics
            metrics = compute_metrics_from_predictions(exp_dir, losses_csv)

            # Build metrics JSON blob (for the progress CSV)
            metrics_json = json.dumps({
                'loss': metrics.get('loss', None),
                'val_loss': metrics.get('val_loss', None),
                'val_mae': metrics.get('val_mae', None),
                'val_mae12': metrics.get('val_mae12', None),
                'mae_2021': metrics.get('mae_2021', None),
                'mae12_2021': metrics.get('mae12_2021', None),
                'epochs_trained': metrics.get('epochs_trained', None),
            })

            progress_rows.append({
                'run_num': cfg['iteration'],
                'model_type': 'MAPE',
                'lead_time': leadtime,
                'cycle': cfg['cycle'],
                'activation': cfg['activation'],
                'num_layers': cfg['num_layers'],
                'neurons': cfg['neurons'],
                'loss_value': metrics.get('loss_value', None),
                'val_mae': metrics.get('val_mae', None),
                'val_mae12': metrics.get('val_mae12', None),
                'mae_2021': metrics.get('mae_2021', None),
                'mae12_2021': metrics.get('mae12_2021', None),
                'metrics': metrics_json,
                'timestamp': '',
                'status': 'completed',
            })

            index_rows.append({
                'model_name': basename,
                'activation': cfg['activation'],
                'num_layers': cfg['num_layers'],
                'neurons': cfg['neurons'],
                'leadtime': leadtime,
                'cycle': cfg['cycle'],
                'run_num': cfg['iteration'],
                'val_mae': metrics.get('val_mae', None),
                'val_mae12': metrics.get('val_mae12', None),
                'mae_2021': metrics.get('mae_2021', None),
                'mae12_2021': metrics.get('mae12_2021', None),
            })

            total_models += 1

    # Write mape_progress.csv
    progress_df = pd.DataFrame(progress_rows)
    progress_path = dest / 'mape_progress.csv'
    progress_df.to_csv(progress_path, index=False)
    print(f"\nProgress CSV: {len(progress_rows)} rows -> {progress_path}")

    # Write _index.csv
    index_df = pd.DataFrame(index_rows)
    index_path = pred_dir / '_index.csv'
    index_df.to_csv(index_path, index=False)
    print(f"Index CSV: {len(index_rows)} models -> {index_path}")

    # Write mape_timing.csv (empty, just header — coworker data doesn't have equivalent)
    timing_path = dest / 'mape_timing.csv'
    with open(timing_path, 'w') as f:
        f.write('timestamp,event_type,component,section,duration_sec,notes\n')

    # Build air_temp.csv
    print("\nBuilding air_temp.csv...")
    build_air_temp_csv(pred_dir)

    print(f"\n{'='*60}")
    print(f"Adapted {total_models} models -> {dest}")
    print(f"{'='*60}")
    print(f"\nTo visualize:")
    print(f"  python run_v15.py  (then select {dest})")
    print(f"\nTo serve time series:")
    print(f"  python serve_timeseries.py --folder {dest / 'predictions'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adapt coworker results to tuning-viz format")
    parser.add_argument('--source', type=str, default=None,
                        help="Path to coworker's results folder (skip GUI picker)")
    args = parser.parse_args()

    if args.source:
        source = Path(args.source)
    else:
        source = pick_folder()

    print(f"Source: {source}")
    adapt(source)
