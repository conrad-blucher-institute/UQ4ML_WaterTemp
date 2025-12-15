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


def temp(args):
            

    """
    RUN SCRIPT WITH COOLTURTLES DIRECTORY AS YOUR CWD
    """

    """GPU CHECK/USE EXPLICITLY"""

    # print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set TensorFlow to use only the first GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)




    # path_to_data = "./June_May_Datasets"
    path_to_data = "./data/ESB_datasets"

    # path_to_saved_models = r"C:\Users\cduff4\OneDrive - Texas A&M University-Corpus Christi\CBI\AMS\AMS 2025\CROSS_VALIDATION_COMBO_RUN_RESULTS"

    # testing_datasets = [f"{path_to_data}\\simulated_cs_dataset_1.csv"] #, f"{path_to_data}\\simulated_cs_dataset_2.csv", f"{path_to_data}\\simulated_cs_dataset_3.csv"]

    """ TUNING ITERATIONS AND VARIABLES """
    tuner_iterations = [1]                     

    # trials = number of combinations if Grid Search
    max_trials = 30                               
    execution_per_trial = 2

    # units is synonymous with neurons
    unit_list = [16, 32, 64, 100, 128, 256]       
    activation_list = ['relu', 'selu', 'leaky_relu']
    obj = "val_mae"
    call_back_monitor = "val_loss"


    """TRAINING ITERATIONS - CROSS VALIDATION"""
    start_iteration = 21
    end_iteration = 23

    # step_direction is a step direction for moving through the loop 
    if start_iteration > end_iteration:
        step_direction = -1
    else:
        step_direction = 1
        end_iteration += 1

    """ MODEL ARCHITECTURE VARIABLES and HYPERPARAMETERS """
    # 1, 3, 6, 7, 9 are the cycles with a cold stunning event in the validation set (hyperparameter tuning)
    cycle_list = [0,1,2,3] 

    # 12, 48, 96 are our main;  leadtimes: 12, 24, 48, 72, 96, 108, 120
    lead_time_list = [120]#[12,48,96,120] 
    hours_back = 24  

    # MAIN - Location where models get saved to while training/tuning
    path_to_model_runs = "mape_tune_init_results" + f"/{args.model_type}_{lead_time_list[0]}_{end_iteration}_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # list of temperature perturbations, "0.0" --> perfect prognosis
    # this is for miranda's UQ things; it is set to 0 so it won't kick in
    # we don't worry about this 
    temperature_list = [0.0] #, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] 

    # number of ensemble predictions

    if args.model_type == "crps":

        loss_function = crps_loss
        metrics = [crps] # deal with later, create a code block that implement our custom functions including less than 12 functions

    elif args.model_type == "mse":

        loss_function = 'mse'


    elif args.model_type == "mape":
        loss_function = 'mape'
        

    input_structure = "descending"
    independent_year = "cycle"
    output_activation = 'linear'

    # starting with 0.01, the LEARNING RATE REDUCER reduces this value by 0.01 incrementally later within code # 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
    learning_rate = 0.01 

    optimizer = 'adam' # adam, adadelta, SGD
    kernel_regularizer = 'l2'

    # neurons = 200
    # act_func = 'leaky_relu'
    # num_layers = 1

    # batch size was determined to utilize the entire dataset... when left undeclared, the batch defaults to 32 
    #batch_size_list = [4096] # 4096, 2048, 1024, 512, 256, 128, 64

    # dicitonary to hold the computation time per loop (cycle, leadtime, iteration)
    compute_times = {}

    # column names for the saving of the model predictions later within "train"
    prediction_column_names = []
    for k in range(output_units):
        prediction_column_names.append(f'pred_{k+1}')   

    print("\n\n----------------------------- TUNING ! -----------------------------\n\n")

    if not os.path.exists(path_to_model_runs):
        os.makedirs(path_to_model_runs)

    # he prefers one for loop that iterates through the cartesian product instead of 3 nested for loops
    # so get rid of the outermost for loop one and do something with the 3 inner ones 
    # hector wants to refactor this later
    # actually delete this outermost for-loop later not refactor 
    # redundant notes ^
    for iteration in tuner_iterations: 
        # this tracks how long each takes to tune 
        iteration_time_start = datetime.now()

        lead_time_compute_times = [] # to store the compute times for each leadtime
        

        for lead_time in lead_time_list:

            leadtime_start_time = datetime.now()

            cycle_compute_times = [] # to store the comput times for each cycle
            for cycle in cycle_list:
                # getting the time it takes to tune per cycle
                cycle_start_time = datetime.now()

                save_path = f"{path_to_model_runs}\Iter_{iteration}_{lead_time}h_Cycle_{cycle}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

            

                """ Model Input Variables """
                input_hours_forecast = lead_time
                atp_hours_back = hours_back
                wtp_hours_back = hours_back
                pred_atp_interval = 1 # hour intervals (3 hrs for operational team currently)

                """ Manipulating data for AI Model """
                x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData(path_to_data,
                                                                                                                                            input_structure="descending",
                                                                                                                                            independent_year="not used",
                                                                                                                                            input_hours_forecast=lead_time,
                                                                                                                                            atp_hours_back=atp_hours_back,
                                                                                                                                            wtp_hours_back=wtp_hours_back,
                                                                                                                                            pred_atp_interval=pred_atp_interval,
                                                                                                                                            IPPOffset=temperature_list[0],
                                                                                                                                            cycle=cycle,
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
                        neurons = hp.Choice('neurons_', values=unit_list, default=128, ordered=False)
                        act_func = hp.Choice('act_',values=activation_list, default='leaky_relu',ordered=False)
                        # select a value from min_value to max_value
                        layers = hp.Int('layers', min_value=1, max_value=3, step=1, default=3)


                        # first layer = input layer
                        model.add(Input(shape=(inputShape)))

                        # hidden layer(s)
                        #
                        for i in range(layers):
                            model.add(Dense(units=neurons, 
                                            activation=act_func, 
                                            kernel_regularizer=kernel_regularizer))
                        

                        # last layer = output layer
                        model.add(Dense(output_units, activation=output_activation))


                        # Define the optimizer, learning rate as a hyperparameter to tune.
                        #chosen_optimizer=hp.Choice("optimizer", values=optimizer_list, ordered=False)  
                        chosen_optimizer = optimizer
                        
                        if chosen_optimizer == "adam":
                            model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate), 
                                        loss=loss_function, metrics=args.metrics)
                        
                        # elif chosen_optimizer == "adadelta":
                        #     model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=hp.Choice("learning_rate_",learning_rate_list, ordered=False)), 
                        #                 loss=loss_function, metrics=args.metrics)

                        # elif chosen_optimizer == "SGD":
                        #     model.compile(optimizer=keras.optimizers.SGD(learning_rate=hp.Choice("learning_rate_",learning_rate_list, ordered=False)), 
                        #                 loss=loss_function, metrics=args.metrics)
                        

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
                    objective=kt.Objective(obj, direction="min"), # min is descending
                    overwrite=True, # if we rerun this tuner, overwrite anything that exists (in the directory)
                    max_trials=max_trials, # 
                    executions_per_trial=execution_per_trial, # run this particular combination this amount of times, take the avg of it, that's the metric
                    directory=save_path,
                    project_name="results" 
                    )

                # Learning rate reducer
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=call_back_monitor, min_delta=0.001,
                                                                factor=0.1, patience=15, min_lr=0.00001)
                
                # Defining the early stopping
                early_stopping = EarlyStopping(monitor=call_back_monitor,
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

                cycle_end_time = datetime.now()

                cycle_compute_times.append({f"Cycle {cycle}": cycle_end_time-cycle_start_time})
            
            with open((save_path + r"\total_cycle_compute_time.pkl"), 'wb') as f:
                pickle.dump(cycle_compute_times, f)

            lead_time_end_time = datetime.now()
            lead_time_compute_times.append({
                f"{lead_time}h Lead Time": lead_time_end_time-leadtime_start_time,
                "Cycles": cycle_compute_times  
            })

        with open(save_path + r"\total_lead_time_compute_time.pkl", 'wb') as f:
                pickle.dump(lead_time_compute_times, f)

        iteration_time_end = datetime.now()
        compute_times[f"Iteration {iteration}"] = {
            f"Iteration {iteration}": iteration_time_end-iteration_time_start,
            r"Lead Time Times": lead_time_compute_times
        }


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