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

from utils import crps_loss, crps, pitd, ssrat
from utils import preparingData
import pandas as pd
from datetime import datetime
import glob

import tensorflow.keras.backend as K

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




# if tune:      tuning hyperparameters, prepping for training
# if train:     training a model using hyperparameters gained from tuning
# if test:      pull from location a .h5 trained model, and test
tune_train_test = "tune" # "train", "tune", "test"
model_name = "MSE" 

while True:
    user_input = input(f"You are about to start\n-------------------- {tune_train_test.upper()}ING --------------------\nAre you sure you want to continue {tune_train_test.upper()}ING ? (y/n)\n").strip().lower()
    if user_input in ['y', 'n']:
        break
    else:
        print("Invalid input. Try again. (input a 'y' or a 'n')")


if user_input == 'y':
    print(f"Continuing...")
elif user_input == 'n':
    print(f"Stopping...")
    sys.exit()

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

# MAIN - Location where models get saved to while training/tuning
path_to_model_runs = "mse_mape_tune_init_results" + f"/{model_name}_{start_iteration}_{end_iteration}_" + datetime.now().strftime("%Y%m%d-%H%M%S")

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
lead_time_list = [12,48,96, 120] 
hours_back = 24  

# list of temperature perturbations, "0.0" --> perfect prognosis
# this is for miranda's UQ things; it is set to 0 so it won't kick in
# we don't worry about this 
temperature_list = [0.0] #, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] 

# number of epochs
epochs = 20000

# number of ensemble predictions

if model_name == "CRPS":

    output_units = 100
    loss_function = crps_loss
    metrics = [crps]

elif model_name == "MSE":

    output_units = 1
    loss_function = 'mse'
    metrics = ['mae']


elif model_name == "MAPE":
    output_units = 1
    loss_function = 'mape'
    metrics = ['mape']


    
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



if tune_train_test == "tune":
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
                                                                                                                                            lead_time,
                                                                                                                                            atp_hours_back,
                                                                                                                                            wtp_hours_back,
                                                                                                                                            pred_atp_interval,
                                                                                                                                            IPPOffset = temperature_list[0],
                                                                                                                                            cycle=cycle,
                                                                                                                                            model=model_name) # "model" variable only mattered for when we used lstm; lstm resuired a transofmration of dimensions of input shape
                inputShape = x_train[0].shape

                # batch size to be the full length (# rows) of dataset
                batch_size = x_train.shape[0]

                
                """ TUNING THE MODEL """
                class MyHyperModel(kt.HyperModel):
                    def build(self, hp):

                        model = Sequential()

                        # to tune for the number of units once, to then be appliead to all of the hidden layers
                        # to maintain consistency amongst the layers
                        neurons = hp.Choice('neurons_', values=unit_list, default=128, ordered=False)
                        act_func = hp.Choice('act_',values=activation_list, default='leaky_relu',ordered=False)
                        layers = hp.Int('layers', min_value=1, max_value=3, step=1, default=3)

                        # first layer = input layer
                        model.add(Input(shape=(inputShape)))

                        # hidden layer(s)
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
                                        loss=loss_function, metrics=metrics)
                        
                        # elif chosen_optimizer == "adadelta":
                        #     model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=hp.Choice("learning_rate_",learning_rate_list, ordered=False)), 
                        #                 loss=loss_function, metrics=metrics)

                        # elif chosen_optimizer == "SGD":
                        #     model.compile(optimizer=keras.optimizers.SGD(learning_rate=hp.Choice("learning_rate_",learning_rate_list, ordered=False)), 
                        #                 loss=loss_function, metrics=metrics)
                        

                        return model
                    
                    def fit(self, hp, model, *args, **kwargs):
                                #batch = hp.Choice("batch_size", values=batch_size_list, ordered=False)
                                batch = batch_size
                                history = model.fit(*args, batch_size=batch, **kwargs)
                                
                                return history
                    

                tuner = kt.RandomSearch(  # here change to either: BayesianOptimization, GridSearch, Random
                    hypermodel=MyHyperModel(),
                    objective=kt.Objective(obj, direction="min"),
                    overwrite=True,
                    max_trials=max_trials,
                    executions_per_trial=execution_per_trial,
                    directory=save_path,
                    project_name=save_path + r"\results" 
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

                tuner.search(x_train, y_train, validation_data=(x_val, y_val),  epochs=epochs, callbacks=[early_stopping, reduce_lr, tensorboard_callback])

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

# da archives
elif tune_train_test == "train":
    print("\n\n----------------------------- TRAINING ! -----------------------------\n\n")


    with tf.device('/GPU:0'):

        if not os.path.exists(path_to_model_runs):
            os.makedirs(path_to_model_runs)

        for iteration in range(start_iteration, end_iteration, step_direction):  
                
            for lead_time in lead_time_list:

                if lead_time == 12:

                    cross_val_combinations = [1]

                elif lead_time == 48:
                    cross_val_combinations = [2]

                elif lead_time == 96:
                    
                    cross_val_combinations = [2]
                
                elif lead_time == 6 or lead_time == 24 or lead_time == 72 or lead_time == 120:
                    cross_val_combinations = [1, 2, 3, 4, 5]
                    
                for combination in cross_val_combinations:
                
                    if lead_time == 12:
                        
                        if model_name == "CRPS":
                            if combination == 1:
                                num_layers = 3
                                act_func = 'relu'
                                neurons = 32
            
                            elif combination == 2:
                                num_layers = 2
                                act_func = 'leaky_relu'
                                neurons = 256 
                        elif model_name == "MSE":
                            if combination == 1:
                                num_layers = 3
                                act_func = 'leaky_relu'
                                neurons = 32

                    elif lead_time == 48:

                        if model_name == "CRPS":
                            if combination == 1:
                                num_layers = 3
                                act_func = 'relu'
                                neurons = 32
            
                            elif combination == 2:
                                num_layers = 3
                                act_func = 'selu'
                                neurons = 64
                        elif model_name == "MSE":
                            if combination == 2:
                                num_layers = 2
                                act_func = 'leaky_relu'
                                neurons = 16

                    elif lead_time == 96:

                        if model_name == "CRPS":
                            if combination == 1:
                                num_layers = 3
                                act_func = 'selu'
                                neurons = 32 
            
                            elif combination == 2:
                                num_layers = 3
                                act_func = 'relu'
                                neurons = 100
                                
                        elif model_name == "MSE":
                            if combination == 2:
                                num_layers = 2
                                act_func = 'leaky_relu'
                                neurons = 16
                            
                    elif lead_time == 6:
                        if combination == 1:
                            num_layers = 1
                            act_func = 'relu'
                            neurons = 128
                        if combination == 2:
                            num_layers = 1
                            act_func = 'leaky_relu'
                            neurons = 256
                        if combination == 3:
                            num_layers = 3
                            act_func = 'relu'
                            neurons = 128
                        if combination == 4:
                            num_layers = 3
                            act_func = 'selu'
                            neurons = 64
                        if combination == 5:
                            num_layers = 1
                            act_func = 'leaky_relu'
                            neurons = 100
                        
                    elif lead_time == 120:
                        if combination == 1:
                            num_layers = 2
                            act_func = 'leaky_relu'
                            neurons = 256
                        if combination == 2:
                            num_layers = 2
                            act_func = 'selu'
                            neurons = 32
                        if combination == 3:
                            num_layers = 2
                            act_func = 'relu'
                            neurons = 100
                        if combination == 4:
                            num_layers = 3
                            act_func = 'selu'
                            neurons = 32
                        if combination == 5:
                            num_layers = 3
                            act_func = 'leaky_relu'
                            neurons = 64

                    elif lead_time == 24:
                        if combination == 1:
                            num_layers = 3
                            act_func = 'relu'
                            neurons = 100
                        if combination == 2:
                            num_layers = 2
                            act_func = 'selu'
                            neurons = 256
                        if combination == 3:
                            num_layers = 3
                            act_func = 'relu'
                            neurons = 128
                        if combination == 4:
                            num_layers = 1
                            act_func = 'leaky_relu'
                            neurons = 256
                        if combination == 5:
                            num_layers = 2
                            act_func = 'relu'
                            neurons = 256

                    elif lead_time == 72:
                        if combination == 1:
                            num_layers = 3
                            act_func = 'selu'
                            neurons = 64
                        if combination == 2:
                            num_layers = 3
                            act_func = 'leaky_relu'
                            neurons = 32
                        if combination == 3:
                            num_layers = 3
                            act_func = 'relu'
                            neurons = 128
                        if combination == 4:
                            num_layers = 3
                            act_func = 'selu'
                            neurons = 16
                        if combination == 5:
                            num_layers = 2
                            act_func = 'leaky_relu'
                            neurons = 256

                    combo_name = f"{model_name}-{num_layers}_layers-{act_func}-{neurons}_neurons"

                    for cycle in cycle_list:

                        save_path = path_to_model_runs + f"/{lead_time}h"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        save_path += f"/{combo_name}-cycle_{cycle}-iteration_{iteration}"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        print(f"RUNNING {lead_time}h, {combo_name}-cycle_{cycle}-iteration_{iteration} ...\n")
                        
                        cycle_time_start = datetime.now()
                        

                        """ Model Input Variables """
                        input_hours_forecast = lead_time
                        atp_hours_back = hours_back
                        wtp_hours_back = hours_back
                        pred_atp_interval = 1                   

                        data_prep_time_start = datetime.now()

                        # clearing stale nodes that might be persisting in the 
                        # computation graph, causing corruption; throwing an error
                        K.clear_session()

                        """ Manipulating data for AI Model """
                        x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData(path_to_data,
                                                                                                                                    input_structure,
                                                                                                                                    independent_year,
                                                                                                                                    input_hours_forecast,
                                                                                                                                    atp_hours_back,
                                                                                                                                    wtp_hours_back,
                                                                                                                                    pred_atp_interval,
                                                                                                                                    IPPOffset = temperature_list[0],
                                                                                                                                    cycle=cycle,
                                                                                                                                    model=model_name) # "model" variable only mattered for when we used lstm; lstm resuired a transofmration of dimensions of input shape

                        """PREPARINGDATA FUNCTION COMPUTE TIME"""
                        data_prep_time_end = datetime.now()

                        with open(save_path + r"/data_prep_compute_time.txt", 'w') as compute_time_file:
                            compute_time_file.write(f"preparingData() compute time: {data_prep_time_end-data_prep_time_start}")

                        train_time_start = datetime.now()

                        inputShape = x_train[0].shape
                        
                        if model_name == "MSE":
                            batch_size = x_train.shape[0]
                        
                        elif model_name == "CRPS":
                            batch_size = 512

                            # batch size to be the full length (# rows) of dataset
                            # batch_size = x_train.shape[0]

                        
                        """TRAINING THE MODEL"""

                        model = Sequential()                

                        # first layer = input layer
                        model.add(Input(shape=(inputShape)))

                        # hidden layer(s)
                        for i in range(num_layers):
                            model.add(Dense(units=neurons, 
                                            activation=act_func, 
                                            kernel_regularizer=kernel_regularizer))
                        
                        # DROPOUT LAYERS ?????

                        # last layer = output layer
                        model.add(Dense(output_units, activation=output_activation))


                        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate), 
                                    loss=loss_function, metrics=metrics)
                    
                    
                        # grabbing the epoch logs to save after training
                        from utils import TrainingLogger
                        logger = TrainingLogger(save_path + r"/std_output.txt")

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
                        
                        
                        log_dir = save_path + r"/tensorboard_logs"
                        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                        model_callbacks = [early_stopping, reduce_lr, tensorboard_callback, logger]

                        # Training the model
                        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, 
                                            batch_size=batch_size, callbacks=model_callbacks, verbose=2) 
                        
                        """TRAINING COMPUTE TIME"""
                        train_time_end = datetime.now()

                        

                        with open(save_path + r"/train_compute_time.txt", 'w') as compute_time_file:
                            compute_time_file.write(f"Model Train Time: {train_time_end-train_time_start}")
                        

                        """LOSS INFORMATION"""
                        loss = history.history['loss']
                        val_loss = history.history['val_loss']

                        losses = pd.DataFrame(columns=['Loss', 'Val_Loss']) 
                        losses['Loss'] = loss
                        losses['Val_Loss'] = val_loss
                        losses.to_csv(save_path + r"/losses.csv")


                        # saving the model to h5 file
                        hdf_file = save_path + '/model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_.h5'
                        keras_file = save_path + '/model_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_.keras'

                        model.save(keras_file) 
                        model.save(hdf_file)

                        # # full dataset (full season) metrics
                        # train_metrics = model.evaluate(x_train, y_train, batch_size=batch_size)
                        # print(train_metrics)
                        # val_metrics = model.evaluate(x_val, y_val, batch_size=batch_size)
                        # test_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)

                        
                        # # cold season metric dataframes initial
                        # x_train_cold = pd.DataFrame(data=x_train)
                        # x_val_cold = pd.DataFrame(data=x_val)
                        # x_test_cold = pd.DataFrame(data=x_test)

                        # y_train_cold = pd.DataFrame(data=y_train)
                        # y_val_cold = pd.DataFrame(data=y_val)
                        # y_test_cold = pd.DataFrame(data=y_test)

                        # # fill in training cold season evaluation metrics here ???

                        # """X TRAIN COLD"""
                        # x_train_cold.insert(loc=0, column='date_time', value=training_dates)
                        # x_train_cold['date_time'] = pd.to_datetime(x_train_cold['date_time'], format='%m/%d/%Y %H%M')
                        # x_train_cold = x_train_cold[(x_train_cold['date_time'].dt.month >= 11) | (x_train_cold['date_time'].dt.month <= 2)]
                        # x_train_cold.drop('date_time', axis=1, inplace=True)

                        # """Y TRAIN COLD"""
                        # y_train_cold.insert(loc=0, column='date_time', value=training_dates)
                        # y_train_cold['date_time'] = pd.to_datetime(y_train_cold['date_time'], format='%m/%d/%Y %H%M')
                        # y_train_cold = y_train_cold[(y_train_cold['date_time'].dt.month >= 11) | (y_train_cold['date_time'].dt.month <= 2)]
                        # y_train_cold.drop('date_time', axis=1, inplace=True)

                        # """X VAL COLD"""
                        # x_val_cold.insert(loc=0, column='date_time', value=validation_dates)
                        # x_val_cold['date_time'] = pd.to_datetime(x_val_cold['date_time'], format='%m/%d/%Y %H%M')
                        # x_val_cold = x_val_cold[(x_val_cold['date_time'].dt.month >= 11) | (x_val_cold['date_time'].dt.month <= 2)]
                        # x_val_cold.drop('date_time', axis=1, inplace=True)

                        # """Y VAL COLD"""
                        # y_val_cold.insert(loc=0, column='date_time', value=validation_dates)
                        # y_val_cold['date_time'] = pd.to_datetime(y_val_cold['date_time'], format='%m/%d/%Y %H%M')
                        # y_val_cold = y_val_cold[(y_val_cold['date_time'].dt.month >= 11) | (y_val_cold['date_time'].dt.month <= 2)]
                        # y_val_cold.drop('date_time', axis=1, inplace=True)

                        # """X TEST COLD"""
                        # x_test_cold.insert(loc=0, column='date_time', value=testingDates)
                        # x_test_cold['date_time'] = pd.to_datetime(x_test_cold['date_time'], format='%m/%d/%Y %H%M')
                        # x_test_cold = x_test_cold[(x_test_cold['date_time'].dt.month >= 11) | (x_test_cold['date_time'].dt.month <= 2)]
                        # x_test_cold.drop('date_time', axis=1, inplace=True)

                        # """Y TEST COLD"""
                        # y_test_cold.insert(loc=0, column='date_time', value=testingDates)
                        # y_test_cold['date_time'] = pd.to_datetime(y_test_cold['date_time'], format='%m/%d/%Y %H%M')
                        # y_test_cold = y_test_cold[(y_test_cold['date_time'].dt.month >= 11) | (y_test_cold['date_time'].dt.month <= 2)]
                        # y_test_cold.drop('date_time', axis=1, inplace=True)

                        # cold_train_metrics = model.evaluate(x_train_cold, y_train_cold, batch_size=batch_size)
                        # cold_val_metrics = model.evaluate(x_val_cold, y_val_cold, batch_size=batch_size)
                        # cold_test_metrics = model.evaluate(x_test_cold, y_test_cold, batch_size=batch_size)
                        

                        # train_predictions = model.predict(x_train)
                        # val_predictions = model.predict(x_val)
                        # test_predictions = model.predict(x_test)


                        # """SAVING PREDICTIONS AND OBSERVATIONS"""
                        # train_vs_preds = pd.DataFrame(columns=prediction_column_names, data=train_predictions)
                        # val_vs_preds = pd.DataFrame(columns=prediction_column_names, data=val_predictions)
                        # test_vs_preds = pd.DataFrame(columns=prediction_column_names, data=test_predictions)

                        # train_vs_preds.insert(loc=0, column='date_time', value=training_dates)
                        # val_vs_preds.insert(loc=0, column='date_time', value=validation_dates)
                        # test_vs_preds.insert(loc=0, column='date_time', value=testingDates)

                        # train_vs_preds.insert(loc=1, column='target', value=y_train)
                        # val_vs_preds.insert(loc=1, column='target', value=y_val)
                        # test_vs_preds.insert(loc=1, column='target', value=y_test)

                        # train_vs_preds.to_csv(save_path + r"\train_datetime_obsv_predictions.csv")
                        # val_vs_preds.to_csv(save_path + r"\val_datetime_obsv_predictions.csv")
                        # test_vs_preds.to_csv(save_path + r"\test_datetime_obsv_predictions.csv")


                        # """METRICS"""

                        # # [crps, pitd, ssrat, mae, ensemble_mae12, central_mae, central_ensemble_mae12, ssrel, ign, mf, di]
                        
                        # all_row_names = ['Train', 'Val', 'Test', 'COLD_Train', 'COLD_Val', 'COLD_Test']
                        # train_val_row_names = ['Train', 'Val','COLD_Train', 'COLD_Val']
                        

                        # saving_metrics = pd.DataFrame(columns=metric_names, index=all_row_names)
                        # saving_metrics.loc['Train'] = train_metrics
                        # saving_metrics.loc['Val'] = val_metrics
                        # saving_metrics.loc['Test'] = test_metrics
                        # saving_metrics.loc['COLD_Train'] = cold_train_metrics
                        # saving_metrics.loc['COLD_Val'] = cold_val_metrics
                        # saving_metrics.loc['COLD_Test'] = cold_test_metrics
                        # saving_metrics.to_csv(save_path + r"\train_val_test_metrics.csv")

                        # saving_train_val_metrics = pd.DataFrame(columns=metric_names, index=train_val_row_names)
                        # saving_train_val_metrics.loc['Train'] = train_metrics
                        # saving_train_val_metrics.loc['Val'] = val_metrics
                        # saving_train_val_metrics.loc['COLD_Train'] = cold_train_metrics
                        # saving_train_val_metrics.loc['COLD_Val'] = cold_val_metrics
                        # saving_train_val_metrics.to_csv(save_path + r"\ONLY_train_val_metrics.csv")

                        """TOTAL CYCLE MODEL COMPUTE TIME"""
                        cycle_time_end = datetime.now()

                        with open(save_path + r"/cycle_compute_time.txt", 'w') as compute_time_file:
                            compute_time_file.write(f"Total Cycle Time: {cycle_time_end-cycle_time_start}")                


elif tune_train_test == "test":
    print("\n\n----------------------------- TESTING ! -----------------------------\n\n")
    from mape_mse_utils import creatingAdditionalColumns, dateTimeRetriever
        
    cycle = 8
    iterations = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    lead_time_list = [12, 48, 96]


    simulated_cs = 1
    for dataset in testing_datasets:
        for iteration in iterations:  

            for lead_time in lead_time_list:

                if lead_time == 12:

                    cross_val_combinations = [1,2]

                elif lead_time == 48 or lead_time == 96:
                    
                    cross_val_combinations = [1,2]
                    
                for combination in cross_val_combinations:
                    
                    if lead_time == 12:

                        if combination == 1:
                            num_layers = 1
                            act_func = 'leaky_relu'
                            neurons = 100
        
                        elif combination == 2:
                            num_layers = 2
                            act_func = 'leaky_relu'
                            neurons = 100 

                    elif lead_time == 48:

                        if combination == 1:
                            num_layers = 1
                            act_func = 'selu'
                            neurons = 100
        
                        elif combination == 2:
                            num_layers = 3
                            act_func = 'selu'
                            neurons = 64 

                    elif lead_time == 96:

                        if combination == 1:
                            num_layers = 2
                            act_func = 'selu'
                            neurons = 128
        
                        elif combination == 2:
                            num_layers = 1
                            act_func = 'selu'
                            neurons = 64 


                    combo_name = f"CRPS-{num_layers}_layers-{act_func}-{neurons}_neurons"

                    print(f"RUNNING {dataset}\n {lead_time}h - {combo_name}_iteration_{iteration}")

                    input_hours_forecast = lead_time
                    atp_hours_back = hours_back
                    wtp_hours_back = hours_back 
                    pred_atp_interval = 1


                    save_path = path_to_saved_models + f"\SIMULATED_CS_EVENT_{simulated_cs}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    save_path += f"\{lead_time}h"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    save_path += f"\{combo_name}-cycle_{cycle}-iteration_{iteration}"
                    
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    testing = pd.read_csv(dataset)

                    creating_columns_start_time = datetime.now()
                    testing = creatingAdditionalColumns(testing, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset=temperature_list[0])
                    creating_columns_end_time = datetime.now()


                    x_test = testing.iloc[:,3:-1].values.astype(float) 
                    y_test = testing.iloc[:,-1].values.astype(float)

                    # panda = pd.DataFrame(data=x_test)
                    # panda.to_csv("x_test.csv")

                    # panda = pd.DataFrame(data=y_test)
                    # panda.to_csv("y_test.csv")

                    model_path = glob.glob(f"{path_to_saved_models}\\CRPS_RESULTS_Batch_512\\{lead_time}h\\{combo_name}-cycle_{cycle}-iteration_{iteration}\\*.h5")[0]
                    
                    model = tf.keras.models.load_model(model_path, compile=False)

                    # learning rate at 0.001 based off previous work and referencing Ryan github
                    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=loss_function, metrics=metrics)

                    test_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)

                    predictions_start_time = datetime.now()
                    predictions = model.predict(x_test)
                    predictions_end_time = datetime.now()
                    
                    # print("\nTime\n")
                    # print(predictions_end_time-predictions_start_time)
                    # print()

                    # """SAVING PREDICTIONS"
                    date_times = dateTimeRetriever(testing, lead_time)
                    preds = pd.DataFrame(columns=prediction_column_names, data=predictions)
                    preds.insert(0, "date_time", date_times)
                    preds.insert(1, "Target", y_test)
                    preds.to_csv(f"{save_path}\preds.csv")

                    # print("\nMetrics:\n")
                    # print(test_metrics)
                    
                    # """METRICS"""
                    metric_names = ['loss', 'crps', 'pitd', 'ssrat', 'mae', 'ensemble_mae12']
                    row_name = ['TEST']
                    saving_metrics = pd.DataFrame(columns=metric_names, index=row_name)
                    saving_metrics.loc['TEST'] = test_metrics

                    preds.drop(preds.columns[:1], axis=1, inplace=True)
                    test_true = preds.iloc[:, [0]].astype(float)
                    test_pred = preds.iloc[:, 1:].astype(float)

                    saving_metrics['central_mae'] = None
                    saving_metrics['central_ensemble_mae12'] = None
                    saving_metrics['ssrel'] = None
                    saving_metrics['ign'] = None
                    saving_metrics['mf'] = None
                    saving_metrics['di'] = None

                    # print(test_true)
                    # print(test_pred)

                    adding_metrics_start_time = datetime.now()
                    
                    central_mae_start_time = datetime.now()
                    test_central_mae = central_mae(test_true, test_pred)
                    saving_metrics.at['TEST', 'central_mae'] = test_central_mae.numpy().item()
                    central_mae_end_time = datetime.now()

                    central_mae12_start_time = datetime.now()
                    test_central_mae12 = central_ensemble_mae12(test_true, test_pred)
                    saving_metrics.at['TEST', 'central_ensemble_mae12'] = test_central_mae12.numpy().item()
                    central_mae12_end_time = datetime.now()

                    ssrel_start_time = datetime.now()
                    test_ssrel = ssrel(test_true, test_pred)
                    saving_metrics.at['TEST', 'ssrel'] = test_ssrel.numpy().item()
                    ssrel_end_time = datetime.now()

                    from mape_mse_utils import ign, mf, di

                    ign_start_time = datetime.now()
                    test_ign = ign(test_true, test_pred)
                    saving_metrics.at['TEST', 'ign'] = test_ign
                    ign_end_time = datetime.now()

                    mf_start_time = datetime.now()
                    test_mf = mf(test_true, test_pred)
                    saving_metrics.at['TEST', 'mf'] = test_mf.numpy().item()
                    mf_end_time = datetime.now()

                    di_start_time = datetime.now()
                    test_di = di(test_true, test_pred)
                    saving_metrics.at['TEST', 'di'] = test_di.numpy().item()
                    di_end_time = datetime.now()

                    adding_metrics_end_time = datetime.now()

                    with open(save_path + r"\compute_time.txt", 'w') as compute_time_file:
                        compute_time_file.write(f"creatingAdditionalColumns() time: {creating_columns_end_time-creating_columns_start_time}")
                        compute_time_file.write(f"\nmodel.predict() time: {predictions_end_time-predictions_start_time}")

                        compute_time_file.write(f"\ncentral_mae time: {central_mae_end_time-central_mae_start_time}")
                        compute_time_file.write(f"\ncentral_mae < 12 time: {central_mae12_end_time-central_mae12_start_time}")
                        compute_time_file.write(f"\nssrel time: {ssrel_end_time-ssrel_start_time}")
                        compute_time_file.write(f"\nign time: {ign_end_time-ign_start_time}")
                        compute_time_file.write(f"\nmf time: {mf_end_time-mf_start_time}")
                        compute_time_file.write(f"\ndi time: {di_end_time-di_start_time}")

                        
                        compute_time_file.write(f"\n\ntotal metrics add time: {adding_metrics_end_time-adding_metrics_start_time}")

                    saving_metrics.to_csv(save_path + r"\metrics.csv")
        
        simulated_cs -= 1