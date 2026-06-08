"""
One-time experiment: run trained CRPS models on the ESB-2020-2021 dataset.

Usage:
    python run_esb2020_2021_crps_experiment.py

Setup:
    1. Update MODELS_DIR to point to the folder containing your .keras files.
    2. Verify LEAD_TIME matches the lead time used during training.
    3. Adjust other config params if they differ from the training defaults.

Output:
    A CSV per model + one aggregated CSV are written to OUTPUT_DIR.
    Each CSV has columns: date_time, target, pred_mean, pred_std, member_0 … member_99
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import traceback
from datetime import datetime

# Scripts live in esb2020-2021_experiment/; repo root is one level up.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))   # enables: from src.helper.utils_mse_crps import ...

from src.helper.utils_mse_crps import (
    creatingAdditionalColumns,
    deletingMissingValues,
    dateTimeRetriever,
    crps_loss,
    crps,
)

# =============================================================================
# CONFIGURATION — review and update before running
# =============================================================================

# Models live in esb2020-2021_experiment/crps_results/ (sibling to this script).
# The script searches recursively, so nested subdirectories are fine.
MODELS_DIR = str(_SCRIPT_DIR / "crps_results")

# Path to the ESB-2020-2021 dataset (resolved from repo root)
DATA_PATH = str(_REPO_ROOT / "data" / "ESB_datasets" / "esb_2020_2021.csv")

# Where to write the prediction CSVs (all results live under experiment_results/)
OUTPUT_DIR = str(_SCRIPT_DIR / "experiment_results" / "predictions")
# Logging / debug output directory
LOG_DIR = _SCRIPT_DIR / "experiment_results" / "logs"


def _ensure_dirs():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(msg: str):
    ts = datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}\n"
    print(line, end='')
    with open(LOG_DIR / "output.log", "a", encoding="utf-8") as f:
        f.write(line)


def save_df(df: pd.DataFrame, name: str):
    path = LOG_DIR / f"{name}.csv"
    try:
        df.to_csv(path, index=False)
        write_log(f"Saved DataFrame '{name}' shape={df.shape} -> {path}")
    except Exception as e:
        write_log(f"Failed to save DataFrame '{name}': {e}")

# --- Training configuration ---
# TODO: Verify these match the values used when training your CRPS models.
LEAD_TIME         = 12   # Lead time in hours (e.g. 12, 48, 96, 120)
ATP_HOURS_BACK    = 24   # Hours of air-temperature history used as model input
WTP_HOURS_BACK    = 24   # Hours of water-temperature history used as model input
PRED_ATP_INTERVAL = 1    # Step interval for future air-temp forecast columns
INPUT_STRUCTURE   = "descending"  # Column ordering used during training
IPP_OFFSET        = 0.0  # Ideal-prog perturbation offset (0.0 = no perturbation)

# =============================================================================
# DATA PREPARATION
# =============================================================================

print(f"Loading {DATA_PATH} ...")
df_raw = pd.read_csv(DATA_PATH)
print(f"  {len(df_raw)} rows loaded.")
_ensure_dirs()
write_log(f"Loaded data from {DATA_PATH} rows={len(df_raw)}")
try:
    save_df(df_raw, "df_raw")
except Exception:
    write_log("Warning: failed to save df_raw (see traceback below)")
    write_log(traceback.format_exc())

print("Constructing input features ...")
df_features = creatingAdditionalColumns(
    df_raw.copy(),
    input_structure=INPUT_STRUCTURE,
    input_hours_forecast=LEAD_TIME,
    atp_hours_back=ATP_HOURS_BACK,
    wtp_hours_back=WTP_HOURS_BACK,
    pred_atp_interval=PRED_ATP_INTERVAL,
    IPPOffset=IPP_OFFSET,
)

print("Removing rows with missing values (-999) ...")
df_clean = deletingMissingValues(df_features)
print(f"  {len(df_clean)} valid rows remaining.")
write_log(f"Applied creatingAdditionalColumns -> df_features shape={df_features.shape}")
try:
    save_df(df_features, "df_features")
except Exception:
    write_log("Warning: failed to save df_features (see traceback below)")
    write_log(traceback.format_exc())
write_log(f"Applied deletingMissingValues -> df_clean shape={df_clean.shape}")
try:
    save_df(df_clean, "df_clean")
except Exception:
    write_log("Warning: failed to save df_clean (see traceback below)")
    write_log(traceback.format_exc())

# Extract dates shifted forward by lead time (mirrors training convention)
dates = dateTimeRetriever(df_clean.copy(), LEAD_TIME)
write_log(f"Extracted dates (count={len(dates)}) using dateTimeRetriever")
try:
    pd.Series(dates).to_csv(LOG_DIR / "dates.csv", index=False)
    write_log(f"Saved dates -> {LOG_DIR / 'dates.csv'}")
except Exception:
    write_log("Warning: failed to save dates (see traceback below)")
    write_log(traceback.format_exc())

# Extract feature matrix X and target vector y.
# "descending" layout: col 0 = date, cols 1:-1 = features, col -1 = target.
col_start = 1 if INPUT_STRUCTURE == "descending" else 3
X = df_clean.iloc[:, col_start:-1].values.astype(float)
y = df_clean.iloc[:, -1].values.astype(float)

print(f"  X shape: {X.shape}  |  y shape: {y.shape}")
write_log(f"Feature matrix X shape: {X.shape}; y shape: {y.shape}")

# =============================================================================
# MODEL LOADING & PREDICTION
# =============================================================================

keras_files = sorted(
    glob.glob(os.path.join(MODELS_DIR, "**", "*.keras"), recursive=True)
)

if not keras_files:
    raise FileNotFoundError(
        f"No .keras files found under '{MODELS_DIR}'.\n"
        "Please update MODELS_DIR at the top of this script."
    )

print(f"\nFound {len(keras_files)} .keras model(s) in '{MODELS_DIR}'.")
os.makedirs(OUTPUT_DIR, exist_ok=True)
write_log(f"Found {len(keras_files)} .keras model(s) in '{MODELS_DIR}'")

all_preds = []

for idx, model_path in enumerate(keras_files, start=1):
    write_log(f"\n[{idx}/{len(keras_files)}] {model_path}")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=crps_loss,
            metrics=[crps],
        )
        write_log(f"Loaded model: {model_path} layers={len(model.layers)} params={model.count_params()}")
    except Exception:
        write_log(f"ERROR loading model {model_path} (see traceback)")
        write_log(traceback.format_exc())
        raise

    preds = model.predict(X, verbose=0)   # shape: (n_samples, 100)
    all_preds.append(preds)
    write_log(f"Predicted with model {model_path} -> preds shape: {preds.shape}")
    try:
        pd.DataFrame(preds).to_csv(LOG_DIR / f"preds_model_{idx}.csv", index=False)
        write_log(f"Saved raw predictions -> {LOG_DIR / f'preds_model_{idx}.csv'}")
    except Exception:
        write_log("Warning: failed to save raw predictions (see traceback below)")
        write_log(traceback.format_exc())

    # Build output DataFrame
    member_cols = {f"member_{j}": preds[:, j] for j in range(preds.shape[1])}
    df_out = pd.DataFrame({
        "date_time": dates,
        "target":    y,
        "pred_mean": preds.mean(axis=1),
        "pred_std":  preds.std(axis=1),
        **member_cols,
    })

    # Save the per-model CSV inside the experiment folder so no outputs
    # are written outside this experiment workspace. We mirror the model's
    # relative path under MODELS_DIR into the local crps_results/ folder.
    model_p = Path(model_path).resolve()
    exp_models_root = Path(__file__).resolve().parent / "experiment_results" / "crps_results"
    try:
        rel = model_p.relative_to(Path(MODELS_DIR).resolve())
        target_dir = exp_models_root / rel.parent
    except Exception:
        target_dir = exp_models_root / f"model_{idx}"

    target_dir.mkdir(parents=True, exist_ok=True)
    label = target_dir.name or f"model_{idx}"
    out_path = target_dir / f"{label}_predictions.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

# =============================================================================
# AGGREGATED PREDICTIONS (mean across all loaded models)
# =============================================================================

if len(all_preds) > 1:
    stacked   = np.stack(all_preds, axis=0)   # (n_models, n_samples, 100)
    agg_preds = stacked.mean(axis=0)           # (n_samples, 100)

    member_cols = {f"member_{j}": agg_preds[:, j] for j in range(agg_preds.shape[1])}
    df_agg = pd.DataFrame({
        "date_time": dates,
        "target":    y,
        "pred_mean": agg_preds.mean(axis=1),
        "pred_std":  agg_preds.std(axis=1),
        **member_cols,
    })

    # Write aggregated CSV into the experiment OUTPUT_DIR (predictions/) only
    agg_path = os.path.join(OUTPUT_DIR, "aggregated_predictions.csv")
    df_agg.to_csv(agg_path, index=False)
    print(f"\nAggregated ({len(all_preds)} models) -> {agg_path}")

print("\nDone.")
