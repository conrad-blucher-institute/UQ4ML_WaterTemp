
"""
Author(s): Christian Duff

Modified by: Jarett Woodall


The purpose of this script is to tune, train, and/or test cool turtle machine learning model(s) 
for predicting water temperature in the Laguna Madre, TX for Cold Stunning Events
"""


# Import packages
from src.helper.utils_mse_crps import crps_loss, crps
from src.helper.utils_mse_crps import preparingData

# for manipulating paths
from pathlib import Path

import keras

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

from keras.callbacks import TensorBoard

from keras.layers import Input, Dense

from keras.models import Sequential

from keras.callbacks import EarlyStopping

import pandas as pd

from datetime import datetime

import tensorflow.keras.backend as K

"""
RUN SCRIPT WITH UQ4ML_WaterTemperature AS YOUR CWD / CURRENT WORKING DIRECTORY
"""

# # if train:     training a model using hyperparameters gained from tuning
# tune_train_test = "train"

# model_name_list = ["MAPE"] 
# model_name = "MAPE" # turn this into a string list w/ "MSE", "MAPE", "NLL", "CRPS"

# This determines if the models train normally or if the users wishes to test on independent testing years
# Set this to be '2021' or '2024'
# For Regular testing on rolling origin rotation structure set to "cycle"
# independent_year = "cycle"

def temp(args):

    # independent_year = "2021"

    """ MODEL ARCHITECTURE VARIABLES and HYPERPARAMETERS """
    # 1, 3, 6, 7, 9 are the cycles with a cold stunning event in the validation set (hyperparameter tuning)
    # these will be re-named to rotations
    # cycle_list = [0] #[0, 1, 2, 3]

    """TRAINING ITERATIONS - CROSS VALIDATION"""
    start_iteration = args.start_iteration#1
    end_iteration = args.end_iteration  #15

    # 12, 48, 96 are our main;  leadtimes: 12, 24, 48, 72, 96, 108, 120
    lead_time_list = [12] #[12, 48, 96, 120]
    # hours_back = 24  

    # list of temperature perturbations, "0.0" --> perfect prognosis
    # temperature_list = [0.0] #, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] 

    # epochs = args.epochs


    # input_structure = "descending"
    # output_activation = 'linear'

    # starting with 0.01, the LEARNING RATE REDUCER reduces this value by 0.01 incrementally later within code # 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
    # learning_rate = 0.01 

    # optimizer = 'adam' 
    # kernel_regularizer = 'l2'

    # #change this here
    # path_to_data = "data\ESB_datasets"

    # DELETE COMMENTED-OUT DEAD VARIABLES ONCE YOU CONFIRM THEY'RE NO LONGER NEEDED

    """TRAINING ITERATIONS - CROSS VALIDATION"""
    if start_iteration > end_iteration:
        up_down = -1
    else:
        up_down = 1
        end_iteration += 1

    # If structure  specifies Model specific variables
    # if model_name == "CRPS":

    #     output_units = 100
    #     loss_function = crps_loss
    #     metrics = [crps]

    # elif model_name == "MSE":

    #     output_units = 1
    #     loss_function = 'mse'
    #     metrics = ['mae']

    # # mape is deterministic so it only predicts the temperature
    # # does not capture the uncertainty 
    # elif model_name == "MAPE":
    #     output_units = 1
    #     loss_function = 'mape'
    #     metrics = ['mae','mape']

    # this is for picking which metric is being checked during training
    # for callback options such as early stopping and learning rate reducer
    call_back_monitor = "val_loss"

    # batch size was determined to utilize the entire dataset... when left undeclared, the batch defaults to 32 
    #batch_size_list = [4096] # 4096, 2048, 1024, 512, 256, 128, 64

    #def runner_function():
    # dicitonary to hold the computation time per loop (cycle, leadtime, iteration)
    # compute_times = {}

    # column names for the saving of the model predictions later within "train"
    prediction_column_names = []
    for k in range(args.num_output_neurons):
        prediction_column_names.append(f'pred_{k+1}')   

    if args.tune_train_test == "train":
        print("\n\n----------------------------- TRAINING ! -----------------------------\n\n")

        # for model_name in model_name_list:
        for leadtime in lead_time_list:
            for iteration in range(start_iteration, end_iteration, up_down):  
            # for lead_time in lead_time_list:
                # if lead_time == 12:
                    # if model_name == "MSE":
                    #     num_layers = 2
                    #     act_func = 'leaky_relu'
                    #     neurons = 16
                #     if model_name == "MSE":
                #         num_layers = 1
                #         act_func = 'leaky_relu'
                #         neurons = 2
                # elif lead_time == 48:
                #     if model_name == "MSE":
                #         num_layers = 3
                #         act_func = 'leaky_relu'
                #         neurons = 16
                # elif lead_time == 96:
                #     if model_name == "MSE":
                #         num_layers = 2
                #         act_func = 'leaky_relu'
                #         neurons = 32
                # elif lead_time == 120:
                #     if model_name == "MSE":
                #         num_layers = 2
                #         act_func = 'leaky_relu'
                #         neurons = 16
                # # if lead_time == 12:
                # #     if model_name == "MAPE":
                # #         num_layers = 1
                # #         act_func = 'leaky_relu'
                # #         neurons = 100
                # if lead_time == 12:
                #     if model_name == "MAPE":
                #         num_layers = 1
                #         act_func = 'leaky_relu'
                #         neurons = 100
                # elif lead_time == 48:
                #     if model_name == "MAPE":
                #         num_layers = 3
                #         act_func = 'leaky_relu'
                #         neurons = 32
                # elif lead_time == 96:
                #     if model_name == "MAPE":
                #         num_layers = 1
                #         act_func = 'leaky_relu'
                #         neurons = 256
                # elif lead_time == 120:
                #     if model_name == "MAPE":
                #         num_layers = 3
                #         act_func = 'relu'
                #         neurons = 256
                
                combo_name = f"{args.model_type.lower()}-{args.num_layers}_layers-{args.activation_function}-{args.neurons}_neurons"

                for rotation in args.rotation_list:
                    print(f"RUNNING {leadtime}h, {combo_name}-cycle_{rotation}-iteration_{iteration} ...\n")
                    
                    cycle_time_start = datetime.now()
                    

                    """ Model Input Variables """
                    input_hours_forecast = lead_time
                    # atp_hours_back = hours_back
                    # wtp_hours_back = hours_back
                    # pred_atp_interval = 1                   

                    data_prep_time_start = datetime.now()

                    # clearing stale nodes that might be persisting in the 
                    # computation graph, causing corruption; throwing an error
                    K.clear_session()

                    """ Manipulating data for AI Model """
                    x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData(args.data_set,
                                                                                                                                args.input_structure,
                                                                                                                                args.independent_year,
                                                                                                                                input_hours_forecast,
                                                                                                                                args.atp_hours_back,
                                                                                                                                args.wtp_hours_back,
                                                                                                                                args.pred_atp_interval,
                                                                                                                                IPPOffset = args.temperature_list[0],
                                                                                                                                cycle=rotation,
                                                                                                                                model=args.model_type) # "model" variable only mattered for when we used lstm; lstm resuired a transofmration of dimensions of input shape
                    
                    """PREPARINGDATA FUNCTION COMPUTE TIME"""
                    data_prep_time_end = datetime.now()

                    # Path to folder for visualization results
                    save_path = Path("src") / "results" / f"{args.model_type.lower()}_results" / f"{args.leadtime}h" / f"{combo_name}-rotation_{rotation}-iteration_{iteration}"
                    save_path.mkdir(parents=True, exist_ok=True)

                    with open(save_path / "data_prep_compute_time.txt", 'w') as compute_time_file:
                        compute_time_file.write(f"preparingData() compute time: {data_prep_time_end - data_prep_time_start}")

                    train_time_start = datetime.now()

                    inputShape = x_train[0].shape
                    
                    if args.batch_size == -1:
                        batch_size = x_train.shape[0]

                    else:
                        batch_size = args.batch_size
                        
                    if args.model_type == "MSE":
                        batch_size = x_train.shape[0]

                    elif args.model_type == "MAPE":
                        batch_size = x_train.shape[0] 
                    
                    elif args.model_type == "CRPS":
                        batch_size = 512

                    """TRAINING THE MODEL"""

                    model = Sequential()                

                    # first layer = input layer
                    model.add(Input(shape=(inputShape)))

                    # hidden layer(s)
                    for _ in range(args.num_layers):
                        model.add(Dense(units=args.neurons, 
                                        activation=args.activation_function, 
                                        kernel_regularizer=args.kernel_regularizer))
                    
                    # last layer = output layer
                    model.add(Dense(args.num_output_neurons, activation=args.output_activation))

                    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lrate), 
                                loss=args.loss_function, metrics=args.metrics)
                
                
                    # grabbing the epoch logs to save after training
                    from src.helper.utils_mse_crps import TrainingLogger

                    logger = TrainingLogger(save_path / "std_output.txt")

                    
                    # Learning rate reducer
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=args.call_back_monitor, min_delta=args.min_delta,
                                                                    factor=args.factor, patience=args.lr_reducer_patience, min_lr=args.min_lr)
                    
                    # Defining the early stopping
                    early_stopping = EarlyStopping(monitor=args.call_back_monitor,
                                                        min_delta=args.min_delta,
                                                        patience=args.early_stop_patience,
                                                        verbose=args.verbose,
                                                        mode='auto',
                                                        restore_best_weights=True)
                    
                    
                    log_dir = save_path / "tensorboard_logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                    model_callbacks = [early_stopping, reduce_lr, tensorboard_callback, logger]

                    
                    # Training the model
                    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, 
                                        batch_size=batch_size, callbacks=model_callbacks, verbose=args.verbose) 
                    
                    """TRAINING COMPUTE TIME"""
                    train_time_end = datetime.now()

                    # 
                    # look into re-loading models instead of re-training them again 
                    # use pickle
                    # need to do this for re-producing the bug 

                    with open(save_path / "train_compute_time.txt", 'w') as compute_time_file:
                        compute_time_file.write(f"Model Train Time: {train_time_end-train_time_start}")
                    


                    """LOSS INFORMATION"""
                    loss = history.history['loss']
                    val_loss = history.history['val_loss']
                

                    losses = pd.DataFrame(columns=['Loss', 'Val_Loss']) 
                    losses['Loss'] = loss
                    losses['Val_Loss'] = val_loss
                    losses.to_csv(save_path / "losses.csv")

                    # saving the model to h5 file
                    model.save(save_path / f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}_.keras") 
                    
                    train_predictions = model.predict(x_train)
                    val_predictions = model.predict(x_val)
                    # put print here; focus on those 3 days during the cs event of 2021 to ease analysis
                    test_predictions = model.predict(x_test)

                    """SAVING PREDICTIONS AND OBSERVATIONS"""
                    train_vs_preds = pd.DataFrame(columns=prediction_column_names, data=train_predictions)
                    val_vs_preds = pd.DataFrame(columns=prediction_column_names, data=val_predictions)
                    test_vs_preds = pd.DataFrame(columns=prediction_column_names, data=test_predictions)
                    
                    train_vs_preds.insert(loc=0, column='date_time', value=training_dates)
                    val_vs_preds.insert(loc=0, column='date_time', value=validation_dates)
                    test_vs_preds.insert(loc=0, column='date_time', value=testingDates)

                    train_vs_preds.insert(loc=1, column='target', value=y_train)
                    val_vs_preds.insert(loc=1, column='target', value=y_val)
                    test_vs_preds.insert(loc=1, column='target', value=y_test)
                    
                    """TOTAL CYCLE MODEL COMPUTE TIME"""
                    cycle_time_end = datetime.now()
                    train_path = save_path / "train_datetime_obsv_predictions.csv"
                    val_path = save_path / "val_datetime_obsv_predictions.csv"

                    if args.independent_year != "cycle":
                        test_path = save_path / f"{args.independent_year}_datetime_obsv_predictions.csv"
                    else:

                        test_path = save_path / "test_datetime_obsv_predictions.csv"


                    train_vs_preds.to_csv(train_path)
                    val_vs_preds.to_csv(val_path)
                    test_vs_preds.to_csv(test_path)
                    
                    # """TOTAL CYCLE MODEL COMPUTE TIME"""
                    # cycle_time_end = datetime.now() why is this here twice? delete later if it's for sure not needed

                    with open(save_path / "cycle_compute_time.txt", 'w') as compute_time_file:
                        compute_time_file.write(f"Total Cycle Time: {cycle_time_end-cycle_time_start}")

from src.helper.my_parser import create_parser

def main():
    print("yippee!")
    pass

def show_keys(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")
        
# make sure you run
# python -m src.driver.operational_mse_crps_driver @configs/mape_12h.txt 
if __name__ == "__main__":
    # parse incoming command-line arguments
    parser = create_parser()
    args = parser.parse_args()

    #checking what keys we have in our parser
    show_keys(args)
    main()
    temp(args)