"""
MAPE Scaled Training Driver

Trains MAPE models with StandardScaler normalization on features (X only).
Runs 30 iterations across all ESB lead times and cycles, persists fitted
scalers alongside .keras models, and performs post-hoc inference on both
the ESB 2021 and Laguna Madre 2021 independent test years.

Run from UQ4ML_WaterTemp as CWD:
    python -m src.driver.mape_scaled_driver
"""

import sys
import os
from pathlib import Path

# Ensure repo root is on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dense
from keras.models import Sequential
from src.helper.utils_mse_crps import preparingData, prepare_independent_year, TrainingLogger

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ── Configuration ───────────────────────────────────────────────────────
model_name = "MAPE"
input_structure = "descending"
path_to_data = os.path.join("data", "ESB_datasets")
independent_year = "cycle"

lead_time_list = [12, 48, 96, 120]
cycle_list = [0, 1, 2, 3]
hours_back = 24

start_iteration = 1
end_iteration = 30

epochs = 20000
learning_rate = 0.01
output_units = 1
loss_function = "mape"
metrics = ["mape"]
output_activation = "linear"
optimizer = "adam"
kernel_regularizer = "l2"
call_back_monitor = "val_loss"

# Hyperparameters per lead time (from operational_mse_crps_driver.py MAPE blocks
# and tuner grid — using 3 layers / leaky_relu / 32 neurons as the baseline
# for all lead times since only 12h had an explicit MAPE config)
HYPERPARAMS = {
    12:  {"num_layers": 3, "act_func": "leaky_relu", "neurons": 32},
    48:  {"num_layers": 3, "act_func": "leaky_relu", "neurons": 32},
    96:  {"num_layers": 3, "act_func": "leaky_relu", "neurons": 32},
    120: {"num_layers": 3, "act_func": "leaky_relu", "neurons": 32},
}

# Independent test year paths
ESB_2021_CSV = os.path.join("data", "ESB_datasets", "esb_2020_2021.csv")
LM_2021_CSV = os.path.join(
    "data", "June_May_Datasets",
    "june_atp_and_wtp_2020_2021_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv",
)

# Column mapping for Laguna Madre data → ESB column names
LM_COLUMN_MAP = {
    "dateAndTime": "date",
    "packeryATP_lighthouse": "Air Average",
    "npsbiWTP_lighthouse": "Water Average",
}

RESULTS_ROOT = Path("src") / "results" / "MAPE_scaled_results"


def train_and_infer():
    prediction_column_names = [f"pred_{k+1}" for k in range(output_units)]

    for iteration in range(start_iteration, end_iteration + 1):
        for lead_time in lead_time_list:
            hp = HYPERPARAMS[lead_time]
            num_layers = hp["num_layers"]
            act_func = hp["act_func"]
            neurons = hp["neurons"]
            combo_name = f"mape-{num_layers}_layers-{act_func}-{neurons}_neurons"

            for cycle in cycle_list:
                print(f"\n{'='*60}")
                print(f"RUNNING {lead_time}h, {combo_name}-cycle_{cycle}-iteration_{iteration}")
                print(f"{'='*60}\n")

                cycle_time_start = datetime.now()

                tf.keras.backend.clear_session()

                # ── Data preparation with scaling ───────────────────────
                data_prep_start = datetime.now()
                (
                    x_train, y_train,
                    x_val, y_val,
                    x_test, y_test,
                    training_dates, validation_dates,
                    testingDates, testingAir,
                    scaler,
                ) = preparingData(
                    path_to_data,
                    input_structure,
                    independent_year,
                    lead_time,
                    hours_back,
                    hours_back,
                    1,  # pred_atp_interval
                    IPPOffset=0.0,
                    cycle=cycle,
                    model=model_name,
                    scale=True,
                )
                data_prep_end = datetime.now()

                # ── Output directory ────────────────────────────────────
                save_path = RESULTS_ROOT / f"{lead_time}h" / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"
                save_path.mkdir(parents=True, exist_ok=True)

                with open(save_path / "data_prep_compute_time.txt", "w") as f:
                    f.write(f"preparingData() compute time: {data_prep_end - data_prep_start}")

                # ── Build model ─────────────────────────────────────────
                batch_size = x_train.shape[0]
                inputShape = x_train[0].shape

                model = Sequential()
                model.add(Input(shape=(inputShape)))
                for _ in range(num_layers):
                    model.add(Dense(units=neurons, activation=act_func,
                                    kernel_regularizer=kernel_regularizer))
                model.add(Dense(output_units, activation=output_activation))

                model.compile(
                    optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                    loss=loss_function,
                    metrics=metrics,
                )

                # ── Callbacks ───────────────────────────────────────────
                logger = TrainingLogger(save_path / "std_output.txt")
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=call_back_monitor, min_delta=0.001,
                    factor=0.1, patience=15, min_lr=0.00001,
                )
                early_stopping = EarlyStopping(
                    monitor=call_back_monitor, min_delta=0.001,
                    patience=25, verbose=2, mode="auto",
                    restore_best_weights=True,
                )
                log_dir = save_path / "tensorboard_logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                model_callbacks = [early_stopping, reduce_lr, tensorboard_callback, logger]

                # ── Train ───────────────────────────────────────────────
                train_time_start = datetime.now()
                history = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=model_callbacks,
                    verbose=2,
                )
                train_time_end = datetime.now()

                with open(save_path / "train_compute_time.txt", "w") as f:
                    f.write(f"Model Train Time: {train_time_end - train_time_start}")

                # ── Save losses ─────────────────────────────────────────
                losses = pd.DataFrame({
                    "Loss": history.history["loss"],
                    "Val_Loss": history.history["val_loss"],
                })
                losses.to_csv(save_path / "losses.csv")

                # ── Save model and scaler ───────────────────────────────
                model_filename = f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}_.keras"
                model.save(save_path / model_filename)

                scaler_filename = f"scaler_{model_name}_{lead_time}h_cycle{cycle}_iter{iteration}.joblib"
                joblib.dump(scaler, save_path / scaler_filename)

                # ── Save train/val/test predictions ─────────────────────
                train_preds = model.predict(x_train)
                val_preds = model.predict(x_val)
                test_preds = model.predict(x_test)

                for split_name, preds, dates, targets in [
                    ("train", train_preds, training_dates, y_train),
                    ("val", val_preds, validation_dates, y_val),
                    ("test", test_preds, testingDates, y_test),
                ]:
                    df_out = pd.DataFrame(columns=prediction_column_names, data=preds)
                    df_out.insert(0, "date_time", dates)
                    df_out.insert(1, "target", targets)
                    df_out.to_csv(save_path / f"{split_name}_datetime_obsv_predictions.csv", index=False)

                # ── Post-hoc inference: ESB 2021 ────────────────────────
                esb_inf_path = RESULTS_ROOT / f"{lead_time}h" / "inference_esb2021" / f"cycle_{cycle}-iteration_{iteration}"
                esb_inf_path.mkdir(parents=True, exist_ok=True)

                X_esb, y_esb, dates_esb = prepare_independent_year(
                    ESB_2021_CSV, input_structure, lead_time,
                    hours_back, hours_back,
                    pred_atp_interval=1, IPPOffset=0.0,
                    scaler=scaler,
                )
                preds_esb = model.predict(X_esb)
                df_esb = pd.DataFrame({
                    "date_time": dates_esb,
                    "target": y_esb,
                    "prediction": preds_esb.flatten(),
                })
                df_esb.to_csv(esb_inf_path / "predictions.csv", index=False)

                # ── Post-hoc inference: Laguna Madre 2021 ───────────────
                lm_inf_path = RESULTS_ROOT / f"{lead_time}h" / "inference_lm2021" / f"cycle_{cycle}-iteration_{iteration}"
                lm_inf_path.mkdir(parents=True, exist_ok=True)

                X_lm, y_lm, dates_lm = prepare_independent_year(
                    LM_2021_CSV, input_structure, lead_time,
                    hours_back, hours_back,
                    pred_atp_interval=1, IPPOffset=0.0,
                    scaler=scaler,
                    column_map=LM_COLUMN_MAP,
                )
                preds_lm = model.predict(X_lm)
                df_lm = pd.DataFrame({
                    "date_time": dates_lm,
                    "target": y_lm,
                    "prediction": preds_lm.flatten(),
                })
                df_lm.to_csv(lm_inf_path / "predictions.csv", index=False)

                cycle_time_end = datetime.now()
                with open(save_path / "cycle_compute_time.txt", "w") as f:
                    f.write(f"Total Cycle Time: {cycle_time_end - cycle_time_start}")

                print(f"Completed {lead_time}h cycle_{cycle} iter_{iteration} "
                      f"in {cycle_time_end - cycle_time_start}")


if __name__ == "__main__":
    train_and_infer()
