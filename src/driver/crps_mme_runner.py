"""
Author(s): Christian Duff


The purpose of this program is to run the cool turtle multi-model ensemble
utilizing Continuous Rank Probability Score (CRPS) loss function, as well as 
run the subsequent funcitons needed to produce the metrics of the model


- Data preparation process should remain the same as how we handle our mlp and mme models
- Creating the model should be simmilar to how the cool turtles have created previous models
- Metrics should see a change due to the now ensemble of predictions, where as before our models produces a single discrete value 
- Similar k-fold cross validation method

Model Architecture:
    - Ensemble of 25 models
        - Ensemble of 100 predictions 

Metric Evaluation:
    - Spread Skill Ratio
    - PITD
    - CRPS
    - IGN
    - Attribute Diagram
    - Discard Test

Include user input to allow for flexibility and freedom...?

- Tuning or Training ? 
    ... save models and weights that can be pulled to test/predict 

- Separate file to compute and visualize metrics
"""

# Import packages
import os

import sys
sys.path.append('./src') # need this to import functinos from other files 

# import keras 
import keras
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


from keras.callbacks import TensorBoard
import keras_tuner as kt 
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pickle 

#from properscoring import crps_ensemble ; used for evaluating the model after using the testing set to create predictions

from src.helper.utils_mse_crps import crps_loss, crps
from src.helper.utils_mse_crps import preparingData
import pandas as pd
from datetime import datetime
import glob

import tensorflow.keras.backend as K

def mae_metric(y_true, y_pred):
    """Wrapper for mae that returns a tensor for Keras metric compatibility."""
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1)
    else:
        mean_pred = y_pred
    differences = tf.abs(tf.subtract(y_true, mean_pred))
    return tf.reduce_mean(differences)

def mae12_metric(y_true, y_pred):
    """Wrapper for mae12 that returns a tensor for Keras metric compatibility."""
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        true_val = y_true[:, :1]
    else:
        mean_pred = y_pred
        true_val = y_true
    mask = true_val < 12
    filtered_y_true = tf.boolean_mask(true_val, mask)
    filtered_y_pred_mean = tf.boolean_mask(mean_pred, mask)
    differences = tf.abs(filtered_y_true - filtered_y_pred_mean)
    # Return 0 if no values match the condition
    return tf.cond(tf.size(differences) > 0, 
                   lambda: tf.reduce_mean(differences),
                   lambda: tf.constant(0.0))

def me_metric(y_true, y_pred):
    """Wrapper for me that returns a tensor for Keras metric compatibility."""
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1)
    else:
        mean_pred = y_pred
    mean_square = tf.reduce_mean((tf.subtract(y_true, mean_pred)), axis=-1)
    return tf.reduce_mean(mean_square)

def me12_metric(y_true, y_pred):
    """Wrapper for me12 that returns a tensor for Keras metric compatibility."""
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        true_val = y_true[:, :1]
    else:
        mean_pred = y_pred
        true_val = y_true
    mask = true_val < 12
    filtered_y_true = tf.boolean_mask(true_val, mask)
    filtered_y_pred_mean = tf.boolean_mask(mean_pred, mask)
    differences = filtered_y_true - filtered_y_pred_mean
    # Return 0 if no values match the condition
    return tf.cond(tf.size(differences) > 0,
                   lambda: tf.reduce_mean(differences),
                   lambda: tf.constant(0.0))

def convert_metrics_to_callables(metric_list):
    """
    Convert metric names (strings) to their corresponding callable functions.
    
    Args:
        metric_list: List of metric names as strings (e.g., ['mae', 'mae12', 'me12'])
    
    Returns:
        List of callable metric functions
    """
    
    # Dictionary mapping metric names to their callable functions
    metrics_dict = {
        'mae': mae_metric,
        'mae12': mae12_metric,
        'me': me_metric,
        'me12': me12_metric,
        'crps': crps,
        'mse': 'mse',  # keras built-in metric
        'mae_builtin': 'mae',  # keras built-in metric
        'mape': 'mape',  # keras built-in metric
    }
    
    converted_metrics = []
    for metric in metric_list:
        if metric in metrics_dict:
            converted_metrics.append(metrics_dict[metric])
        else:
            # Try to use it as a keras built-in metric
            print(f"Warning: Metric '{metric}' not found in custom metrics. Attempting to use as built-in Keras metric.")
            converted_metrics.append(metric)
    
    return converted_metrics

def temp(args):
    """
    RUN SCRIPT WITH COOLTURTLES DIRECTORY AS YOUR CWD
    """

    """GPU CHECK/USE EXPLICITLY"""

    # print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # print(f'{args.call_back_monitor} bleep')
    # print(f'{args.rotation} bloop')
    # print(f'{args.leadtime} blip')
    # return

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set TensorFlow to use only the first GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)

    """ MODEL ARCHITECTURE VARIABLES and HYPERPARAMETERS """

    # # MAIN - Location where models get saved to while training/tuning
    path_to_model_runs = f"{args.results_folder}/{args.model_type}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Convert metric strings to callable functions
    args.metrics = convert_metrics_to_callables(args.metrics)

    # dicitonary to hold the computation time per loop (cycle, leadtime, iteration)
    compute_times = {}

    # column names for the saving of the model predictions later within "train"
    prediction_column_names = []
    for k in range(args.num_output_neurons):
        prediction_column_names.append(f'pred_{k+1}')   

    print("\n\n----------------------------- TUNING ! -----------------------------\n\n")

    if not os.path.exists(path_to_model_runs):
        os.makedirs(path_to_model_runs)

    lead_time_compute_times = [] # to store the compute times for each leadtime
    
    for lead_time in args.leadtime_list:

        leadtime_start_time = datetime.now()

        rotation_compute_times = [] # to store the comput times for each rotation
        for rotation in args.rotation_list:
            # getting the time it takes to tune per rotation
            rotation_start_time = datetime.now()

            save_path = f"{path_to_model_runs}\Lead_Time_{lead_time}h_Rotation_{rotation}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            """ Model Input Variables """
            pred_atp_interval = 1 # hour intervals (3 hrs for operational team currently)

            """ Manipulating data for AI Model """
            x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData(args.data_set,
                                                                                                                                        input_structure=args.input_structure,
                                                                                                                                        independent_year="not used",
                                                                                                                                        input_hours_forecast=lead_time,
                                                                                                                                        atp_hours_back=args.atp_hours_back,
                                                                                                                                        wtp_hours_back=args.wtp_hours_back,
                                                                                                                                        pred_atp_interval=pred_atp_interval,
                                                                                                                                                                                                                                   cycle=rotation,
                                                                                                                                        model=args.model_type) # "model" variable only mattered for when we used lstm; lstm resuired a transofmration of dimensions of input shape
            inputShape = x_train[0].shape

            # batch size to be the full length (# rows) of dataset
            batch_size = x_train.shape[0]

            
            """ TUNING THE MODEL """
            # kerastuner hypermodel
            class MyHyperModel(kt.HyperModel):
                def build(self, hp):

                    model = Sequential()

                    # to tune for the number of units once, to then be appliead to all of the hidden layers
                    # to maintain consistency amongst the layers

                    # parameter search space
                    neurons = hp.Choice('neurons_', values=args.unit_list, default=128, ordered=False)
                    act_func = hp.Choice('act_',values=args.activation_function_list, default='leaky_relu',ordered=False)
                    # select a value from min_value to max_value
                    layers = hp.Int('layers', min_value=1, max_value=3, step=1, default=3)


                    # first layer = input layer
                    model.add(Input(shape=(inputShape)))

                    # hidden layer(s)
                    #
                    for i in range(layers):
                        model.add(Dense(units=neurons, 
                                        activation=act_func, 
                                        kernel_regularizer=args.kernel_regularizer))
                    
                    # last layer = output layer
                    model.add(Dense(args.num_output_neurons, activation=args.activation_function))

                    ''' This is legacy code that was used to tune the optimizer; settled for adam. '''
                    # Define the optimizer, learning rate as a hyperparameter to tune.

                    #chosen_optimizer=hp.Choice("optimizer", values=optimizer_list, ordered=False)  
                    chosen_optimizer = args.optimizer

                    if chosen_optimizer == "adam":
                        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lrate), 
                                    loss=args.loss_function, metrics=args.metrics)
                    
                    # elif chosen_optimizer == "adadelta":
                    #     model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=hp.Choice("learning_rate_",learning_rate_list, ordered=False)), 
                    #                 loss=loss_function, metrics=args.metrics)

                    # elif chosen_optimizer == "SGD":
                    #     model.compile(optimizer=keras.optimizers.SGD(learning_rate=hp.Choice("learning_rate_",learning_rate_list, ordered=False)), 
                    #                 loss=loss_function, metrics=args.metrics)
                    ''' End of legacy code used to tune the optimizer. '''

                    return model
                

                def fit(self, hp, model, *args, **kwargs):
                            #batch = hp.Choice("batch_size", values=batch_size_list, ordered=False)
                            batch = batch_size
                            history = model.fit(*args, batch_size=batch, **kwargs)
                            
                            return history
                

            # tuner is an instance of randomsearch
            # its attributes are:
            # hypermodel - instance of myhypermodel
            
            # the parameter search space is all of the parameters that we're gonna consider 
            # 54 combinations

            tuner = kt.RandomSearch(  # here change to either: BayesianOptimization, GridSearch, Random
                hypermodel=MyHyperModel(),
                objective=kt.Objective(args.tuner_objective, direction="min"), # min is descending
                overwrite=True, # if we rerun this tuner, overwrite anything that exists (in the directory)
                # trials = number of combinations if Grid Search
                max_trials=args.max_trials, 
                executions_per_trial=args.executions_per_trial, # run this particular combination this amount of times, take the avg of it, that's the metric
                directory=save_path,
                project_name="results" 
                )

            # Learning rate reducer
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=args.call_back_monitor, min_delta=0.001,
                                                            factor=0.1, patience=15, min_lr=0.00001)
            
            # Defining the early stopping
            early_stopping = EarlyStopping(monitor=args.call_back_monitor,
                                                min_delta=0.001,
                                                patience=25,
                                                verbose=2,
                                                mode='auto',
                                                restore_best_weights=True)
            
            log_dir = save_path + r"\logs"
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            

            tuner.search_space_summary()

            tuner.search(x_train, y_train, validation_data=(x_val, y_val),  epochs=args.epochs, callbacks=[early_stopping, reduce_lr, tensorboard_callback])

            tuner.results_summary()

            rotation_end_time = datetime.now()

            rotation_compute_times.append({f"Rotation {rotation}": rotation_end_time-rotation_start_time})
        
        with open((save_path + r"\total_cycle_compute_time.pkl"), 'wb') as f:
            pickle.dump(rotation_compute_times, f)

        lead_time_end_time = datetime.now()
        lead_time_compute_times.append({
            f"{lead_time}h Lead Time": lead_time_end_time-leadtime_start_time,
            "Rotations": rotation_compute_times  
        })

    with open(save_path + r"\total_lead_time_compute_time.pkl", 'wb') as f:
            pickle.dump(lead_time_compute_times, f)



    with open(save_path + r"\compute_times.pkl", 'wb') as f:
        pickle.dump(compute_times, f)


def main():
    print('I am in main')
    pass

def show_keys(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

from src.helper.my_parser import create_parser



if __name__ == "__main__":

    # Parse incoming command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    # checking what keys we have in our parser
    show_keys(args)

    main()

    temp(args) 