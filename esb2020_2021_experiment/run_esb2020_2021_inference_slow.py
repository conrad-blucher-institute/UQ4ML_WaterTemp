"""
ESB-2020-2021 Inference Runner (Standalone) - SLOW VERSION
===========================================================

This is the unoptimized copy of the inference script.  It deliberately retains
all of the original inefficiencies so it can be used for comparison or demo
purposes.  In particular:

* `creatingAdditionalColumns` (non-vectorized) is used.
* The ESB CSV is re-read on every call to `prepare_independent_year`.
* Feature preparation is done inside the iteration loop instead of being
  cached across iterations.
* The model is compiled after loading (not required for inference).
* The first `.keras` file found (`keras_files[0]`) is used instead of the
  last.

Run exactly the same way as the normal script:
    python run_esb2020_2021_inference_slow.py [--model ...]

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
    creatingAdditionalColumns,
    deletingMissingValues,
    dateTimeRetriever,
)

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
MODELS_DIR = Path(__file__).resolve().parent / f"{model_name.lower()}_results"

# Results directory for saving inference CSVs (2021_datetime_obsv_predictions_esb.csv)
RESULTS_DIR = Path(__file__).resolve().parent / "experiment_results" / f"{model_name.lower()}_results"

# Path to the ESB-2020-2021 independent year CSV.
# readingData() intentionally skips this file; here we load it directly.
ESB_2020_2021_PATH = _REPO_ROOT / "data" / "ESB_datasets" / "esb_2020_2021.csv"

# =============================================================================
# MODEL OUTPUT CONFIG  — set per model type (CRPS, MSE, MAPE)
# =============================================================================

if model_name == "CRPS":
    output_units = 100
    loss_function = crps_loss
    metrics_list = [crps]

elif model_name == "MSE":
    output_units = 1
    loss_function = 'mse'
    metrics_list = ['mae']

elif model_name == "MAPE":
    output_units = 1
    loss_function = 'mape'
    metrics_list = ['mape']

prediction_column_names = [f'pred_{k+1}' for k in range(output_units)]

# =============================================================================
# HELPERS
# =============================================================================

def prepare_independent_year(lead_time: int):
    """Return (X, y, dates) for the ESB-2020-2021 independent year.

    This version re-reads the CSV every time and uses the un-optimized
    feature-creation function.  It is intentionally inefficient.
    """
    df_raw = pd.read_csv(ESB_2020_2021_PATH)

    df_features = creatingAdditionalColumns(
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

    col_start = 1 if input_structure == "descending" else 3
    X = df_clean.iloc[:, col_start:-1].values.astype(float)
    y = df_clean.iloc[:, -1].values.astype(float)

    return X, y, dates


def get_combo_arch(lead_time: int, combination: int):
    """Same as the optimized script"""
    # ... copy same architecture selection as main script ...
    if model_name == "CRPS":
        if lead_time == 12:
            if combination == 1:
                return 3, 'relu', 32
            elif combination == 2:
                return 2, 'leaky_relu', 256

        elif lead_time == 48:
            if combination == 1:
                return 3, 'relu', 32
            elif combination == 2:
                return 3, 'selu', 64

        elif lead_time == 96:
            if combination == 1:
                return 3, 'selu', 32
            elif combination == 2:
                return 3, 'relu', 100

        elif lead_time == 120:
            if combination == 2:
                return 3, 'relu', 100

    elif model_name == "MSE":
        if lead_time == 12:
            return 2, 'leaky_relu', 16
        elif lead_time == 48:
            return 3, 'leaky_relu', 16
        elif lead_time == 96:
            return 2, 'leaky_relu', 32
        elif lead_time == 120:
            return 2, 'leaky_relu', 16

    elif model_name == "MAPE":
        if lead_time == 12:
            return 1, 'leaky_relu', 100
        elif lead_time == 48:
            return 3, 'leaky_relu', 32
        elif lead_time == 96:
            return 1, 'leaky_relu', 256
        elif lead_time == 120:
            return 3, 'relu', 256

    return None


def get_cross_val_combinations(lead_time: int):
    if lead_time == 12:
        return [1]
    elif lead_time in (48, 96, 120):
        return [2]

# =============================================================================
# INFERENCE LOOP (slow version)
# =============================================================================

if start_iteration > end_iteration:
    up_down = -1
    _end = end_iteration - 1
else:
    up_down = 1
    _end = end_iteration + 1

print("\n========== INFERENCE: ESB-2020-2021 (SLOW) ==========")

for iteration in range(start_iteration, _end, up_down):
    for lead_time in lead_time_list:
        cross_val_combinations = get_cross_val_combinations(lead_time)
        # **no caching** – recompute every time
        X_2021, y_2021, dates_2021 = prepare_independent_year(lead_time)
        print(f"\n  lead_time={lead_time}h  X shape: {X_2021.shape}")

        for combination in cross_val_combinations:
            combo_arch = get_combo_arch(lead_time, combination)
            if combo_arch is None:
                print(f"  WARNING: No architecture defined for {model_name} lead_time={lead_time}h combination={combination} — skipping.")
                continue
            num_layers, act_func, neurons = combo_arch
            combo_name = f"{model_name.lower()}-{num_layers}_layers-{act_func}-{neurons}_neurons"

            for cycle in cycle_list:
                label = f"{lead_time}h | {combo_name}-cycle_{cycle}-iteration_{iteration}"
                print(f"  {label}")

                model_dir = (
                    MODELS_DIR
                    / f"{lead_time}h"
                    / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"
                )
                result_dir = (
                    RESULTS_DIR
                    / f"{lead_time}h"
                    / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"
                )

                out_path = result_dir / "2021_datetime_obsv_predictions_esb.csv"
                if out_path.exists():
                    print(f"    Skipping: output already exists -> {out_path}")
                    continue

                keras_files = glob.glob(str(model_dir / "*.keras"))
                if not keras_files:
                    print(f"    WARNING: no .keras file in {model_dir} — skipping.")
                    continue

                model_path = keras_files[0]  # first file, not last
                print(f"    Loading: {Path(model_path).name}")

                K.clear_session()

                result_dir.mkdir(parents=True, exist_ok=True)

                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss=loss_function,
                    metrics=metrics_list,
                )

                preds = model.predict(X_2021, verbose=0)

                df_out = pd.DataFrame(columns=prediction_column_names, data=preds)
                df_out.insert(loc=0, column='date_time', value=dates_2021)
                df_out.insert(loc=1, column='target',    value=y_2021)

                df_out.to_csv(out_path, index=False)
                print(f"    Saved  -> {out_path.name}")

print("\n\nInference Complete (SLOW).")