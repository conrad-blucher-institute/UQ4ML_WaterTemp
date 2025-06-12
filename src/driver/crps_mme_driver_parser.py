"""
Author(s): Christian Duff
Co-Author(s): Hector Marrero-Colominas (adding in refactoring to work with parsers)


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
import pandas as pd
from datetime import datetime
import parser_crps
from src.helper.utils import preparingData
#from properscoring import crps_ensemble ; used for evaluating the model after using the testing set to create predictions

loss_function = None
metrics = None

# Grab all variables
all_vars = parser_crps.get_all_variables()

# Unpack all variables at once
(path_to_data, path_to_tuner, tensorboard_log_dir, 
    tuner_iterations, max_trials, execution_per_trial, epochs, 
    cycle_list, lead_time_list, hours_back, temperature_list, output_units, unit_list, activation_list, 
    output_activation, learning_rate_list, optimizer_list, kernel_regularizer, obj, call_back_monitor, 
    compute_times) = all_vars

# Print
parser_crps.print_configuration()

if loss_function == None or metrics == None:
    try:
        from src.helper.utils import crps_loss, crps, pitd, ssrat, mae, mse, rmse
        loss_function = crps_loss
        metrics = [crps, pitd, ssrat, mae, mse, rmse]
    except ImportError:
        # Option 2: Set them as None and define them in your main script after setup
        loss_function = None
        metrics = None
        print("Warning: Could not import loss_function and metrics. Define them after setup_global_configuration()")

"""
RUN SCRIPT WITH COOLTURTLES DIRECTORY AS YOUR CWD
"""

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
