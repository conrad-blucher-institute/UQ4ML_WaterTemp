import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress 1:INFO, 2:INFO and WARNING 3:Info, Warning, and ERROR logs

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import absl.logging
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity(absl.logging.ERROR)

import keras 
import tensorflow as tf
from keras.callbacks import TensorBoard
import keras_tuner as kt 
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
import pandas as pd

import sys
sys.path.append('./src') # need this to import functinos from other files 
# from utils import preparingData
# from src.helper.utils import crps, ssrat_avg, get_spread_skill_points, get_pit_points, mf, di, mae, mae12
from src.helper.utils import crps, ssrat_avg, mae, mae12
from src.helper.utils_mse_crps import crps_loss, crps
from src.helper.utils_mse_crps import preparingData
from datetime import datetime
import json

# this file is used to produce all trials of hyperparameter tuning on a csv
# the csvs this file produces is used in hyperparameter_bycombo_bycycle to make 
# the aggregate tables that make analysis easier

# function to get the hyperparameters based on the specified trial
def load_trial_hyperparameters(trial_folder):
    import os
    import json

    trial_json_path = os.path.join(trial_folder, "trial.json")
    if os.path.exists(trial_json_path):
        with open(trial_json_path, "r") as f:
            trial_data = json.load(f)
        # expecting the json to have a structure like {"hyperparameters": {"values": { ... }}}
        return trial_data.get("hyperparameters", {}).get("values", {})
    else:
        return None

#  load in a saved tuned model
def load_tuned_model(tuner_directory, trial_id, hypermodel):
    import os
    import tensorflow as tf

    # construct the trial folder path
    trial_folder = os.path.join(tuner_directory, f"trial_{trial_id}")

    # load trial-specific hyperparameters if available; otherwise, use global defaults
    trial_hparams = load_trial_hyperparameters(trial_folder)
    if trial_hparams is None:
        trial_hparams = global_hparams
        print(f"No trial.json found for trial id {trial_id}. Using global hyperparameters.")
    else:
        print(f"Loaded hyperparameters for trial id {trial_id} from trial.json.")

    # build the model using the provided hypermodel and hyperparameters
    model = hypermodel.build(trial_hparams)

    # locate the checkpoint file; tf.train.latest_checkpoint finds the most recent checkpoint file in the directory
    latest_checkpoint = tf.train.latest_checkpoint(trial_folder)
    if latest_checkpoint is None:
        print(f"No checkpoint found in {trial_folder} for trial id {trial_id}.")
        return None
    else:
        # loading in the model weights from the trial 
        print(f"Loading weights from {latest_checkpoint} for trial id {trial_id}.")
        model.load_weights(latest_checkpoint)
    
    
    return model, trial_hparams

class MyHyperModel(kt.HyperModel):
    def build(self, hp):

        # If hp is a dict (loaded from oracle.json), use its values directly.
        if isinstance(hp, dict):
            neurons = hp.get("neurons_", 128)
            act_func = hp.get("act_", "leaky_relu")
            layers = hp.get("layers", 3)
            learning_rate = hp.get("learning_rate_", 0.01)
        else:
            # Otherwise, assume hp is a HyperParameters object and use its methods.
            neurons = hp.Choice("neurons_", values=[16, 32, 64, 100, 128, 256], default=128, ordered=False)
            act_func = hp.Choice("act_", values=["relu", "selu", "leaky_relu"], default="leaky_relu")
            layers = hp.Int("layers", min_value=1, max_value=3, step=1, default=3)
            learning_rate = hp.Choice("learning_rate_", values=[0.01], default=0.01)
        

        model = Sequential()

        # first layer = input layer
        model.add(Input(shape=(inputShape)))

        # hidden layer(s)
        for i in range(layers):
            model.add(Dense(units=neurons, 
                            activation=act_func, 
                            kernel_regularizer=kernel_regularizer))
        

        # last layer = output layer
        model.add(Dense(output_units, activation=output_activation))


        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                        loss=loss_function, metrics=metrics)
        return model
    
    def fit(self, hp, model, *args, **kwargs):
                batch = batch_size
                history = model.fit(*args, batch_size=batch, **kwargs)
                
                return history

path_to_data = "./data/ESB_Datasets"

# model_name = "CRPS"
model_name = "MSE"


obj = "val_loss"
# max_trials = 18
max_trials = 30
execution_per_trial = 2
kernel_regularizer = 'l2'
learning_rate = 0.01
unit_list = [16, 32, 64, 100, 128, 256]       
# activation_list = ['relu', 'selu', 'leaky_relu']
activation_list = ['tanh', 'sigmoid']


output_units = 1
output_activation = 'linear'

# save_path = "../Models/CRPS_MME_TUNER_RESULTS"
save_path = "../Models/MSE__TUNER_RESULTS"

# loss_function = crps_loss
# metrics = [crps]
loss_function = 'mse'
metrics = ["mae", "mae12", "me", "me12"]

# lead_times = [12] #,48,96]
# lead_times = [12, 48, 96, 120] #,48,96]
lead_times = [12, 48, 96, 120]
cycles = [0, 1, 2, 3]
iterations = [1]

# column names for the saving of the model predictions later within "train"
prediction_column_names = []
for k in range(output_units):
    prediction_column_names.append(f'pred_{k+1}')  


for lead_time in lead_times:

    # to store results by leadtime into a dataframe
    df_rows = []
    predictions = []

    for cycle in cycles:
        for iteration in iterations:

            print()
            print("RUNNING...")
            print("Iteration: ", iteration) 
            print("Lead Time: ", lead_time)
            print("Cycle: ", cycle)

            """ Model Input Variables """
            input_hours_forecast = lead_time
            atp_hours_back = 24
            wtp_hours_back = 24
            pred_atp_interval = 1 # hour intervals (3 hrs for operational team currently)
            # input_structure = "ascending"
            input_structure = "descending"


            """ Manipulating data for AI Model """
            # "model" variable only mattered for when we used lstm; lstm resuired a transofmration of dimensions of input shape
            temporary = preparingData(path_to_data, input_structure, "cycle", lead_time, atp_hours_back,wtp_hours_back, pred_atp_interval, IPPOffset = 0.0, cycle=cycle,model=model_name)

            # same for c++ i think
            

            x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = temporary  

            y_train = y_train.reshape(y_train.shape[0], 1).astype("float32")
            y_val = y_val.reshape(y_val.shape[0], 1).astype("float32")
            y_test = y_test.reshape(y_test.shape[0], 1).astype("float32")

            inputShape = x_train.shape[1]

            # path to results; this directory has the "trial_00", "trial_01", etc folders inside
            # as well as the oracle.json file which is where the saved models can be retrieved
            tuner_directory = save_path + f"/Iteration_{iteration}_{lead_time}h_Lead_Time_Cycle_{cycle}"
            oracle_path = os.path.join(tuner_directory, "oracle.json")

            with open(oracle_path, "r") as f:
                oracle_data = json.load(f)

            global_hparams = oracle_data.get("hyperparameters", {}).get("values", {})
            # print(global_hparams)
            
            # the model weights can be accesssed through the hash corresponding to the trial number
            id_to_hash = oracle_data.get("id_to_hash", {})

            # loop through all trials, load models, and make predictions
            for trial_id in id_to_hash.keys():
                
                print(f"\nProcessing trial id: {trial_id}")

                # grabbing the model weights to make predictions to then run metrics against
                # also the hyperparameters to store in the dataframe ot signify which hparams produced 
                # which results specifically
                model, hparams = load_tuned_model(tuner_directory, trial_id, MyHyperModel())

                if model is not None:
                    # getting the predictions 
                    train_preds = model.predict(x_train).astype("float32")
                    val_preds = model.predict(x_val).astype("float32")

                    # if lead_time == 12:
                    #     if  (hparams['layers'] == 1 and hparams['act_'] == 'selu' and hparams['neurons_'] == 128) or\
                    #         (hparams['layers'] == 1 and hparams['act_'] == 'relu' and hparams['neurons_'] == 128) or\
                    #         (hparams['layers'] == 1 and hparams['act_'] == 'leaky_relu' and hparams['neurons_'] == 128) or\
                    #         (hparams['layers'] == 1 and hparams['act_'] == 'leaky_relu' and hparams['neurons_'] == 100) or\
                    #         (hparams['layers'] == 2 and hparams['act_'] == 'leaky_relu' and hparams['neurons_'] == 100) or\
                    #         (hparams['layers'] == 2 and hparams['act_'] == 'leaky_relu' and hparams['neurons_'] == 256) or\
                    #         (hparams['layers'] == 3 and hparams['act_'] == 'relu' and hparams['neurons_'] == 32) or\
                    #         (hparams['layers'] == 3 and hparams['act_'] == 'leaky_relu' and hparams['neurons_'] == 128):

                    #             # saving the predictions
                    #             df_val = pd.DataFrame(columns=prediction_column_names, data=val_preds)
                    #             df_train = pd.DataFrame(columns=prediction_column_names, data=train_preds)

                    #             df_train.insert(loc=0, column='date_time', value=training_dates)
                    #             df_val.insert(loc=0, column='date_time', value=validation_dates)

                    #             df_train.insert(loc=1, column='target', value=y_train)
                    #             df_val.insert(loc=1, column='target', value=y_val)

                                
                    #             df_val.to_csv(f"./VAL_{lead_time}h_cycle{cycle}_{hparams['layers']}_layers-{hparams['act_']}-{hparams['neurons_']}_neurons.csv")
                    #             df_train.to_csv(f"./TRAIN_{lead_time}h_cycle{cycle}_{hparams['layers']}_layers-{hparams['act_']}-{hparams['neurons_']}_neurons.csv")

                    # og_val_metrics = model.evaluate(x_val, y_val, batch_size=512)
                    # og_train_metrics = model.evaluate(x_train, y_train, batch_size=512)

                    # print(f"Val Predictions for trial {trial_id}:\n{val_preds}")
                    # print(f"Train Predictions for trial {trial_id}:\n{train_preds}")

                    # taking the mean of the predictions to be passed into 
                    # the ssrel function
                    val_preds_mean = val_preds.mean(axis=-1)
                    train_preds_mean = train_preds.mean(axis=-1)
                    val_preds_mean = val_preds_mean.reshape(val_preds_mean.shape[0], 1).astype("float32")
                    train_preds_mean = train_preds_mean.reshape(train_preds_mean.shape[0], 1).astype("float32")

                    # getting the standard deviations to then be passed
                    # into the ssrat, ssrel, and pitd functions
                    val_preds_std = val_preds.std(axis=-1)
                    train_preds_std = train_preds.std(axis=-1)

                    val_preds_std = val_preds_std.reshape(val_preds_std.shape[0], 1).astype("float32")
                    train_preds_std = train_preds_std.reshape(train_preds_std.shape[0], 1).astype("float32")

                    # append a dict representing one row of 
                    # a df containing tuner results
                    df_rows.append({
                        'Cycle:': cycle,
                        'Iteration:': iteration,
                        'Trial Number:': int(trial_id),
                        # 'val_crps Score': crps(y_val,val_preds),
                        'layer_units_num': hparams['neurons_'],
                        'input act. func.': hparams['act_'],
                        '# of layers': hparams['layers'],
                        # 'crps': crps(y_train, train_preds),
                        # 'ssrat': ssrat_avg(y_train, train_preds_mean, train_preds_std),
                        # 'ssrel': get_spread_skill_points(y_train, train_preds_mean, train_preds_std),
                        # 'pitd': get_pit_points(y_train, train_preds, train_preds_std),
                        # 'mf': mf(y_train, train_preds),
                        # 'di': di(y_train, train_preds),
                        'mae': mae(y_train, train_preds),
                        'mae12': mae12(y_train, train_preds),
                        # 'val_crps': crps(y_val,val_preds),
                        # 'val_ssrat': ssrat_avg(y_val, val_preds_mean, val_preds_std),
                        # 'val_ssrel': get_spread_skill_points(y_val, val_preds_mean, val_preds_std),
                        # 'val_pitd': get_pit_points(y_val, val_preds, val_preds_std),
                        # 'val_mf': mf(y_val, val_preds),
                        # 'val_di': di(y_val, val_preds),
                        'val_mae': mae(y_val, val_preds),
                        'val_mae12': mae12(y_val, val_preds)              
                    })  

                else:
                    print(f"Skipping trial {trial_id} due to missing model or weights SMH !!!")

    # print(df_rows)
    df = pd.DataFrame(df_rows)
    df.to_csv(f"./{lead_time}h_hyperparametersExcel_val_crps_ALL_Trials.csv")
    del df