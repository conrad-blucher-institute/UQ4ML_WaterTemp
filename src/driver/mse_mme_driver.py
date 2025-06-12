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

Include user input to allow for flexibility and freedom...

- Tuning or Training ? 
    ... save models and weights that can be pulled to test/predict 

- Separate file to compute and visualize metrics
"""

# Import packages
#from src.utils import mme_crps_model
import keras 
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import keras_tuner as kt 
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pickle 

#from properscoring import crps_ensemble ; used for evaluating the model after using the testing set to create predictions

from utils import crps_loss, crps, ssrat, mae, mse, rmse, pitd, ssrel
from utils import preparingData
import pandas as pd
from datetime import datetime

"""
RUN SCRIPT WITH COOLTURTLES DIRECTORY AS YOUR CWD
"""


# path_to_data = r".\repos\coolTurtles\June_May_Dataset"
# path_to_tuner = r".\repos\coolTurtles\CRPS_MME_HYPERPARAMETER_TUNING"
path_to_data = r"C:\Users\cduff4\OneDrive - Texas A&M University-Corpus Christi\CBI\repos\coolTurtles\June_May_Dataset"
path_to_tuner = r"C:\Users\cduff4\OneDrive - Texas A&M University-Corpus Christi\CBI\repos\coolTurtles\CRPS_MME_TUNER_RESULTS"
tensorboard_log_dir = r"C:\Users\cduff4\OneDrive - Texas A&M University-Corpus Christi\CBI\CRPS_MME_TENSORBOARD" 

""" TUNING VARIABLES """
tuner_iterations = [1, 2]                      # number of runs/iterations 
max_trials = 18                               # 30 for 1 execution, 15 for 2  trials = # of combinations if Grid Search
execution_per_trial = 2                       # determined by marina previously; does this remain the same ?
epochs = 1000                                # determined by marina previously; does this remain the same ?

""" MODEL ARCHITECTURE VARIABLES, HYPERPARAMETERS """
cycle_list = [1,3,6,7,9] 
# 1, 3, 6, 7, 9 are the cycles with a cold stunning event in the validation set (hyperparameter tuning)

lead_time_list = [12,48,96] # 12, 48, 96 are our main leadtimes: 12, 24, 48, 72, 96, 108, 120
hours_back = 24  

# utilizing only perfect prog ?
temperature_list = [0.0] #, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] # list of temperature perturbations


output_units = 100 # number of ensemble predictions

unit_list = [16, 32, 64, 100, 128, 256] # number of units or "neurons" per layer
activation_list = ['relu', 'selu', 'leaky_relu']
output_activation = 'linear'
learning_rate_list = [0.01] # starting with 0.01, the LEARNING RATE REDUCER reduces this value by 0.01 incrementally later within code # 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
# batch size was determined to utilize the entire dataset... when left undeclared, the batch defaults to 32 
#batch_size_list = [4096] # 4096, 2048, 1024, 512, 256, 128, 64
optimizer_list = ['adam'] # # adam, adadelta, SGD
kernel_regularizer = 'l2'
loss_function = crps_loss
metrics = [crps, pitd, ssrat, mae, mse, rmse]
obj = "val_crps"
call_back_monitor = "val_loss"

# dicitonary to hold the computation time per loop (cycle, leadtime, iteration)
compute_times = {}

for iteration in tuner_iterations:
    iteration_time_start = datetime.now()

    lead_time_compute_times = [] # to store the comput times for each leadtime
    
    for lead_time in lead_time_list:

        leadtime_start_time = datetime.now()

        cycle_compute_times = [] # to store the comput times for each cycle
        for cycle in cycle_list:
            # getting the time it takes to tune per cycle
            cycle_start_time = datetime.now()

        

            """ Model Input Variables """
            input_hours_forecast = lead_time + 6
            atp_hours_back = hours_back
            wtp_hours_back = hours_back
            pred_atp_interval = 1 # hour intervals (3 hrs for operational team currently)

            """ Manipulating data for MLP Ensemble """
            x_train, y_train, x_val, y_val, x_test, y_test, testingDates, testingAir = preparingData(path_to_data,
                                                                                                    lead_time,
                                                                                                    atp_hours_back,
                                                                                                    wtp_hours_back,
                                                                                                    pred_atp_interval,
                                                                                                    IPPOffset = temperature_list[0],
                                                                                                    cycle=cycle,
                                                                                                    model="CRPS-MME")
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
                    chosen_optimizer = optimizer_list[0]
                    
                    if chosen_optimizer == "adam":
                        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate_",learning_rate_list, ordered=False)), 
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
                directory=path_to_tuner,
                project_name='Iteration_' + str(iteration) + '_' + str(lead_time) + 'h_Lead_Time_Cycle_' + str(cycle) 
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
            
            log_dir = tensorboard_log_dir + '\Iteration_' + str(iteration) + '_' + str(lead_time) + 'h_Lead_Time_Cycle_' + str(cycle) + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            

            tuner.search_space_summary()

            tuner.search(x_train, y_train, validation_data=(x_val, y_val),  epochs=epochs, callbacks=[early_stopping, reduce_lr, tensorboard_callback])

            tuner.results_summary()


            # loss = tuner.hypermodel.fit.history['loss']
            # val_loss = tuner.fit.history['val_loss']
            # file = open('Loss.txt', 'w')
            # file.write("Loss, Validation Loss \n")

            # for i in range(len(val_loss)):
            #     file.write(str(loss[i]) + ",  " + str(val_loss[i]) + "\n")
            # file.close()

            cycle_end_time = datetime.now()

            cycle_compute_times.append({f"Cycle {cycle}": cycle_end_time-cycle_start_time})
        
        with open((path_to_tuner + "\Iteration_" + str(iteration) + "_lead_time_" + str(lead_time) + "_total_cycle_compute_time.pkl"), 'wb') as f:
            pickle.dump(cycle_compute_times, f)

        lead_time_end_time = datetime.now()
        lead_time_compute_times.append({
            f"{lead_time}h Lead Time": lead_time_end_time-leadtime_start_time,
            "Cycles": cycle_compute_times  
        })

    with open(path_to_tuner + "\Iteration_" + str(iteration) + "_total_lead_time_compute_time.pkl", 'wb') as f:
            pickle.dump(lead_time_compute_times, f)

    iteration_time_end = datetime.now()
    compute_times[f"Iteration {iteration}"] = {
        f"Iteration {iteration}": iteration_time_end-iteration_time_start,
        "Lead Time Times": lead_time_compute_times
    }


with open(path_to_tuner + "\compute_times.pkl", 'wb') as f:
    pickle.dump(compute_times, f)
