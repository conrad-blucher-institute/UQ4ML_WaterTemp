"""
ESB-2020-2021 Inference Runner (Standalone)
============================================

Loads pre-trained .keras models and generates predictions on the ESB-2020-2021
independent year dataset. Saves 2021_datetime_obsv_predictions.csv files into
experiment_results/crps_results/ matching the directory layout of the trained models.

This script handles ONLY inference — visualization passes are separate.

Run from the repo root:
    python esb2020-2021_experiment/run_esb2020_2021_inference.py [--model crps|mape|mse]

Examples:
    # CRPS (default)
    python esb2020-2021_experiment/run_esb2020_2021_inference.py

    # MAPE models
    python esb2020-2021_experiment/run_esb2020_2021_inference.py --model mape

    # MSE models, restrict iterations
    python esb2020-2021_experiment/run_esb2020_2021_inference.py --model mse --start-iteration 3 --end-iteration 5

Output locations:
    - All inference CSVs (2021_datetime_obsv_predictions.csv) are saved under
      esb2020-2021_experiment/experiment_results/crps_results/{lead_time}h/...

References:
    Training logic  -> src/driver/operational_mse_crps_driver.py
"""

import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import tensorflow.keras.backend as K

# --- Path setup ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))          # enables: from src.helper.utils_mse_crps import ...

from src.helper.utils_mse_crps import (
    crps_loss,
    crps,
    # creatingAdditionalColumns,
    deletingMissingValues,
    dateTimeRetriever,
)

from benchmark_creatingAdditionalColumns import creatingAdditionalColumns_OPTIMIZED

# =============================================================================
# CONFIGURATION  — mirrors operational_mse_crps_driver.py
# =============================================================================

# command-line arguments make it easy to run inference for CRPS, MAPE or MSE
parser = argparse.ArgumentParser(description="Run ESB-2020-2021 inference for a given model type")
parser.add_argument("--model", choices=["crps", "mape", "mse"], default="crps",
                    help="model type / folder suffix (lowercase)")
parser.add_argument("--start-iteration", type=int, default=1,
                    help="first iteration to process (inclusive)")
parser.add_argument("--end-iteration", type=int, default=10,
                    help="last iteration to process (inclusive)")
parser.add_argument("--esb-path", type=Path, default=None,
                    help="optional path to ESB CSV to use instead of the default esb_2020_2021.csv")
parser.add_argument("--air-source", type=str, default=None,
                    help="suffix for model results directory (e.g., '_abs_air' -> model_results_abs_air)")
args = parser.parse_args()

model_name = args.model.upper()  # e.g. 'CRPS', 'MAPE', 'MSE'

# TODO: Verify these match the values used when training your models.
# Set defaults based on model type: CRPS uses cycle [8], while MAPE/MSE use cycles [0,1,2,3]
# and have 30 iterations by default.
if model_name in ("MAPE", "MSE"):
    cycle_list = [0, 1, 2, 3]
    start_iteration = args.start_iteration
    # if user kept the CLI default, override to 30 iterations for MAPE/MSE
    end_iteration = 30 if args.end_iteration == 10 else args.end_iteration
else:
    cycle_list = [8]
    start_iteration = args.start_iteration
    end_iteration = args.end_iteration           # inclusive

lead_time_list   = [12, 48, 96, 120]
hours_back       = 24           # atp_hours_back and wtp_hours_back
temperature_list = [0.0]        # IPP offset; 0.0 = perfect prognosis, no perturbation
input_structure  = "descending"
path_to_data     = "data/ESB_datasets"

# MODELS_DIR and RESULTS_DIR now derive from the requested model type.
# For example, running with --model mape will look in "mape_results" and
# write outputs to "experiment_results/mape_results".
# If --air-source is provided, append it as a suffix (e.g., "mape_results_abs_air").
model_suffix = args.air_source if args.air_source else ""
MODELS_DIR = Path(__file__).resolve().parent / f"{model_name.lower()}_results{model_suffix}"

# Results directory for saving inference CSVs (2021_datetime_obsv_predictions_esb.csv)
RESULTS_DIR = Path(__file__).resolve().parent / "experiment_results" / f"{model_name.lower()}_results{model_suffix}"

# Path to the ESB-2020-2021 independent year CSV.
# readingData() intentionally skips this file; here we load it directly.
# Allow overriding via `--esb-path` CLI argument (if provided)
if args.esb_path is not None:
    ESB_2020_2021_PATH = Path(args.esb_path)
else:
    ESB_2020_2021_PATH = _REPO_ROOT / "data" / "ESB_datasets" / "esb_2020_2021.csv"

# =============================================================================
# MODEL-SPECIFIC ARCHITECTURE + OUTPUT CONFIG
# Each model function returns (arch_tuple, output_units, loss_function, metrics_list)
# For CRPS the combination choice is encoded by lead_time: use combo1 for 12h,
# and combo2 for 48/96/120h (per your request).
# =============================================================================

def _crps_info(lead_time: int):
    # select architecture according to requested mapping (combo1 for 12h, combo2 for others)
    if lead_time == 12:
        arch = (3, 'relu', 32)           # combo 1
    elif lead_time == 48:
        arch = (3, 'selu', 64)           # combo 2
    elif lead_time == 96:
        arch = (3, 'relu', 100)          # combo 2
    elif lead_time == 120:
        arch = (3, 'relu', 100)          # combo 2 (only combo2 exists)
    else:
        arch = None

    output_units = 100
    loss_function = crps_loss
    metrics_list = [crps]
    return arch, output_units, loss_function, metrics_list


def _mse_info(lead_time: int):
    # architectures for MSE (lead_time -> arch)
    if lead_time == 12:
        arch = (2, 'leaky_relu', 16)
    elif lead_time == 48:
        arch = (3, 'leaky_relu', 16)
    elif lead_time == 96:
        arch = (2, 'leaky_relu', 32)
    elif lead_time == 120:
        arch = (2, 'leaky_relu', 16)
    else:
        arch = None

    output_units = 1
    loss_function = 'mse'
    metrics_list = ['mae']
    return arch, output_units, loss_function, metrics_list


def _mape_info(lead_time: int):
    # architectures for MAPE (lead_time -> arch)
    if lead_time == 12:
        arch = (1, 'leaky_relu', 100)
    elif lead_time == 48:
        arch = (3, 'leaky_relu', 32)
    elif lead_time == 96:
        arch = (1, 'leaky_relu', 256)
    elif lead_time == 120:
        arch = (3, 'relu', 256)
    else:
        arch = None

    output_units = 1
    loss_function = 'mape'
    metrics_list = ['mape']
    return arch, output_units, loss_function, metrics_list



def get_combo_arch(model_name: str, lead_time: int):
    """Return (num_layers, act_func, neurons) for a given model + lead time.

    Dispatches to the appropriate model-specific function based on `model_name`.
    """
    if model_name == "CRPS":
        arch, *_ = _crps_info(lead_time)
        return arch
    elif model_name == "MSE":
        arch, *_ = _mse_info(lead_time)
        return arch
    elif model_name == "MAPE":
        arch, *_ = _mape_info(lead_time)
        return arch
    return None


# Provide global model output config (used when constructing CSV column names).
# These are model-level (not lead-time dependent) but obtained from the model
# info functions for consistency.
if model_name == "CRPS":
    _, output_units, loss_function, metrics_list = _crps_info(lead_time=12)
elif model_name == "MSE":
    _, output_units, loss_function, metrics_list = _mse_info(lead_time=12)
elif model_name == "MAPE":
    _, output_units, loss_function, metrics_list = _mape_info(lead_time=12)
else:
    output_units = 0
    loss_function = None
    metrics_list = []

prediction_column_names = [f'pred_{k+1}' for k in range(output_units)]

# =============================================================================
# HELPERS
# =============================================================================

def prepare_independent_year(lead_time: int, df_raw: pd.DataFrame | None = None):
    """Return (X, y, dates) for the ESB-2020-2021 independent year.

    Calls the same sub-functions that preparingData() uses internally, but
    applied directly to esb_2020_2021.csv, which readingData() skips.
    The same parameters (hours_back, input_structure, IPPOffset) are used
    so the feature layout exactly matches what the models were trained on.

    If `df_raw` is provided the function uses that DataFrame instead of
    re-reading the CSV; this enables callers to load the file once and reuse
    it for multiple lead times (useful when caching outside the loop).
    """
    if df_raw is None:
        df_raw = pd.read_csv(ESB_2020_2021_PATH)

    df_features = creatingAdditionalColumns_OPTIMIZED(
        df_raw.copy(),
        input_structure=input_structure,
        input_hours_forecast=lead_time,
        atp_hours_back=hours_back,
        wtp_hours_back=hours_back,
        pred_atp_interval=1,
        IPPOffset=temperature_list[0],
    )

    df_clean = deletingMissingValues(df_features)

    # dateTimeRetriever mutates the 'date' column, so pass a copy
    dates = dateTimeRetriever(df_clean.copy(), lead_time)

    # Mirrors reshaping() col_start logic:
    #   "descending" layout -> col 0 = date, cols 1:-1 = features, col -1 = target
    col_start = 1 if input_structure == "descending" else 3
    X = df_clean.iloc[:, col_start:-1].values.astype(float)
    y = df_clean.iloc[:, -1].values.astype(float)

    return X, y, dates



# =============================================================================
# INFERENCE LOOP  — same loop nesting as operational_mse_crps_driver.py
#                   Training block replaced with: load model → predict → save CSV
# =============================================================================

if start_iteration > end_iteration:
    up_down = -1
    _end = end_iteration - 1
else:
    up_down = 1
    _end = end_iteration + 1

print("\n========== INFERENCE: ESB-2020-2021 ==========\n")


# -------------------------------------------------------------
# Prepare features once per lead_time instead of once per iteration
# -------------------------------------------------------------
# read ESB dataset a single time
_df_esb = pd.read_csv(ESB_2020_2021_PATH)
# dictionary caches keyed by lead_time
_feature_cache = {}
for iteration in range(start_iteration, _end, up_down):

    for lead_time in lead_time_list:

        if lead_time not in _feature_cache:
            X_2021, y_2021, dates_2021 = prepare_independent_year(lead_time, df_raw=_df_esb)
            _feature_cache[lead_time] = (X_2021, y_2021, dates_2021)
            print(f"\n  lead_time={lead_time}h  X shape: {X_2021.shape} (cached)")
        else:
            X_2021, y_2021, dates_2021 = _feature_cache[lead_time]
            print(f"\n  lead_time={lead_time}h  X shape: {X_2021.shape} (reused)")

        # Determine architecture for this model + lead_time (combination logic removed)
        combo_arch = get_combo_arch(model_name, lead_time)
        if combo_arch is None:
            print(f"  WARNING: No architecture defined for {model_name} lead_time={lead_time}h — skipping.")
            continue

        num_layers, act_func, neurons = combo_arch
        combo_name = f"{model_name.lower()}-{num_layers}_layers-{act_func}-{neurons}_neurons"

        for cycle in cycle_list:

            label = f"{lead_time}h | {combo_name}-cycle_{cycle}-iteration_{iteration}"
            print(f"  {label}")

            # Model directory — where .keras files are located
            model_dir = (
                MODELS_DIR
                / f"{lead_time}h"
                / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"
            )

            # Results directory — where inference CSVs will be saved (mirrors model_dir layout)
            result_dir = (
                RESULTS_DIR
                / f"{lead_time}h"
                / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"
            )

            # If the inference output already exists, skip this model to save time
            out_path = result_dir / "2021_datetime_obsv_predictions_esb.csv"
            if out_path.exists():
                print(f"    Skipping: output already exists -> {out_path}")
                continue

            keras_files = glob.glob(str(model_dir / "*.keras"))
            if not keras_files:
                print(f"    WARNING: no .keras file in {model_dir} — skipping.")
                continue

            model_path = keras_files[-1]
            print(f"    Loading: {Path(model_path).name}")

            K.clear_session()  # mirrors operational_mse_crps_driver.py

            result_dir.mkdir(parents=True, exist_ok=True)  # ensure output dir exists

            # load without compiling (only needed for training)
            model = tf.keras.models.load_model(model_path, compile=False)

            preds = model.predict(X_2021, verbose=0)

            # Build CSV in the exact same format as operational_mse_crps_driver.py:
            #   df = pd.DataFrame(columns=prediction_column_names, data=predictions)
            #   df.insert(0, 'date_time', dates); df.insert(1, 'target', y)
            df_out = pd.DataFrame(columns=prediction_column_names, data=preds)
            df_out.insert(loc=0, column='date_time', value=dates_2021)
            df_out.insert(loc=1, column='target',    value=y_2021)

            # File named  2021_datetime_obsv_predictions.csv  — matches the convention in
            # operational_mse_crps_driver.py:  f"{independent_year}_datetime_obsv_predictions.csv"
            # All inference outputs are saved under experiment_results/
            df_out.to_csv(out_path, index=False)
            print(f"    Saved  -> {out_path.name}")

print("\n\nInference Complete.")
