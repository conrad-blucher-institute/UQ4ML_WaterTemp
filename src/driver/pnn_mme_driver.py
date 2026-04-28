'''
Author: Hector Marrero-Colominas
Original Author & Inspo: Dr. Fagg
'''
# example execution:    $ python pnn_mme_driver.py @models/config.txt --enviroment=local (add optional command line args here)

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization, Dropout, SpatialDropout2D, InputLayer
import re
import multiprocessing as mp

from pathlib import Path
import sys
import tensorflow_probability as tfp


from src.helper.utils_pnn import preparingData
# logging experiment to mimic schooners std.out & std.error files
from src.helper.Logger import Logging 
from src.helper.job_iterator import JobIterator
# import result_visualizer

from src.helper.my_parser import create_parser
from src.helper.utils_mse_crps import ryan_ssrel, ssrat_avg, pitd, mae, mse, mae12, me12, me, errorBelow12c, max10PercentError


import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning) # ignoring the performance warning that pandas throws because the method we are using to combine dataframes is causing de-fragmentation

################## Configure figure parameters
FONTSIZE = 18
FIGURE_SIZE = (10,4)
FIGURE_SIZE2 = (10,10)

plt.rcParams.update({'font.size': FONTSIZE})
plt.rcParams['figure.figsize'] = FIGURE_SIZE
# Default tick label size
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
##################

#

def create_classifier_network_generic_probability(  input_shape=None,
                                
                                nchannels=None, # for CNN ; conv layers
                                num_output_neurons=None, 
                                learning_rate=None, 
                                lambda_l2=None, # None or a float
                                loss_function=None,

                                activation_function=None,

                                p_dropout=None,
                                p_spatial_dropout=None,
                                n_hidden=None, #[5]
                                metrics=None,
                                
                                path=None): 

    # create our custom loss function
    def mdn_cost(mu, sigma, y):
        dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        return tf.reduce_mean(-dist.log_prob(y))

    regularizer = tf.keras.regularizers.l2(lambda_l2) if lambda_l2 is not None else None
    genericInputLayer = keras.layers.Input(input_shape, name="generic_input_layer_w_shape")

    x = genericInputLayer
    if len(input_shape) == 1:
        print("inside len == 1 mlp")
        # how we did it in the lab before 

        for i, v in enumerate(n_hidden):
            x = keras.layers.Dense(units=v, activation=activation_function, kernel_regularizer=regularizer)(x)
        
        if p_dropout is not None:
            layerDropout = Dropout(p_dropout)(x)
    
    layer1 = x

    @keras.saving.register_keras_serializable(package="hector_pnn", name="sigma_activation")
    def SigmaActivation(x):
        return tf.nn.elu(x)+1.1 # added a .1 so that it will never be negative or zero, minimum will be 0.1

    # Output Nodes
    mu = Dense(1, name="mu", activation="linear")(layer1)
    sigma = Dense(1, name="sigma", activation=SigmaActivation)(layer1) # added a .1 so that it will never be negative or zero, minimum will be 0.1

    
    # Loss Function
    y_real = keras.layers.Input(shape=(1,), name="y_real_input") 
    lossF = mdn_cost(mu, sigma, y_real)

    # Build Model
    if path != None:
        model = tf.keras.models.load_model(path, compile=False)
    else:
        model = keras.models.Model(inputs=[genericInputLayer, y_real], outputs=[mu, sigma]) # btw keras.Model & keras.models.Model are equivalent
    
    model.add_loss(lossF)

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                    amsgrad = False)
    
    model.summary()  # Print the summary of the neural network

    # Bind the model to the optimizer
    model.compile(  
                    optimizer=opt,
                    metrics=metrics)

    return model


def load_MLP_dataset(args):
    x_train, y_train, x_val, y_val, x_test, y_test, date_time, _, val_date_time = preparingData(input_hours_forecast=args.c_leadtime, 
                                                                                atp_hours_back=args.atp_hours_back, 
                                                                                wtp_hours_back=args.wtp_hours_back,
                                                                                cycle=args.c_cycle,
                                                                                path_to_data=args.data_set,
                                                                                date_time=True,
                                                                                val_date_time=True)
    return x_train, y_train, x_val, y_val, x_test, y_test, date_time, val_date_time

#fname functions needs to be updated
def generate_fname_folder(args):

    strng = 'UQ4ML_WaterTemp/src/results/pnn_results/' 

    strng += 'results/' + args.results_folder + '/' 

    strng += '_LT_' + str(args.c_leadtime) + '_/'

    # strng += '_cycle_' + str(args.c_cycle) + '_/' # i took out cycle and replaced with fbase to match the structure the rest of the cool turtle team uses
    fbase = generate_fname(args)  
    strng += fbase + '_/'

    # strng += '_rep_num_' + str(args.c_repetitions) + '_/'
    
    return strng
def generate_fname(args):
    strng = 'results'
    
    strng += f'_LT_{args.c_leadtime:03d}_'

    strng += '_cycle_' + str(args.c_cycle) + '_'

    strng += f'_rep_num_{args.c_repetitions:03d}_'
    
    # strng = strng + '_EX_NUM_' + str(args.experiment_number) + '_'

    # strng = strng + '_LR_%f'%(args.lrate)
    
    if args.dropout_rate is not None:
        strng = strng + '_DR_%.1f'%(args.dropout_rate)
        
    if args.spatial_dropout is not None:
        strng = strng + '_SDR_%.1f'%(args.spatial_dropout)
    
    # if args.l2 is not None:
    #     strng = strng + '_L2_%f'%(args.l2)
    
    if args.n_filters is not None:
        strng = strng + '_filters_' + '_'.join(str(n) for n in args.n_filters)
    
    if args.kernel_sizes is not None:
        strng = strng + '_kernels_' + '_'.join(str(n) for n in args.kernel_sizes)

    if args.pooling is not None:
        strng = strng + '_pooling_' + '_'.join(str(n) for n in args.pooling)
    
    # if args.n_hidden is not None:
    #     strng = strng + '_hidden_' + '_'.join(str(n) for n in args.n_hidden)
    
    if args.jobid is not None:
        strng += '_jobid_' + str(args.jobid)

    return strng
def generate_short_fname(args):
    strng = 'r'
    
    strng += f'_LT_{args.c_leadtime:03d}_'

    strng += '_c_' + str(args.c_cycle) + '_'

    strng += f'_rn_{args.c_repetitions:03d}_'

    return strng

def execute_experiment(args):

    our_dictionary = {"leadtime":args.leadtime , "cycle":args.cycle , "repetitions":(list(range(args.repetitions)))}
    if args.verbose > 1: 
        print(type(our_dictionary))

    # uses Dr. Faggs JobIterator class to create a cartesian product list of our experiment variations
    breakdown = JobIterator(our_dictionary).get_index(args.experiment_number)

    if args.verbose > 0:
        print("leadtime ",args.leadtime , "_cycle ", args.cycle , "_repetitions ", args.repetitions)
        print(our_dictionary)
        print(breakdown)

    # saving the current leadtime, cycle, and repetition number for this experiment run, for future referencing 
    args.c_leadtime = breakdown['leadtime']
    args.c_cycle = breakdown['cycle']
    args.c_repetitions = breakdown['repetitions']


    folder_fbase = generate_fname_folder(args)              # generate path for our folder
    fbase = generate_fname(args)                            # generate path for saving results
    Path(folder_fbase).mkdir(parents=True, exist_ok=True)   # create path for results; so directory wont get flooded



    # Check if the .pkl file and model folder already exists
    model_result_file = Path(folder_fbase + "%s_model" % fbase)
    pickle_result_file = Path(folder_fbase + "%s_results.pkl" % fbase)
    if model_result_file.exists() and pickle_result_file.exists():
        print(f"Skip experiment {args.experiment_number:03d}. Name: {fbase} is completed. Stored: {folder_fbase}")
        return  # Skip running this experiment if the results file exists

    print(f"Starting Experiment number {args.experiment_number}. Name: {fbase}.")

    if args.enviroment == 'local': # checks whether enviroment is on local machine or schooner
        sys.stdout = Logging(folder_fbase + "stdout.log", sys.stdout) # create & save file w/ stdout
        sys.stderr = Logging(folder_fbase + "stderr.log", sys.stderr) # create & save file w/ error log for debugging

    # Function call to the dataPreparation file to generate the inputs for the AI model
    ins, outs, ins_validation, outs_validation, x_test, y_test, testing_date_time, validation_date_time = load_MLP_dataset(args)

    ## hector, this is really inelegant, fix it later future_hector.        
    if args.model_type == 'mlp_prob':
        model = create_classifier_network_generic_probability(
                                            input_shape=ins[0].shape, 

                                            num_output_neurons=args.num_output_neurons, 
                                            learning_rate=args.lrate,
                                            loss_function=args.loss_function,
                                            activation_function=args.activation_function,
                                            p_spatial_dropout=args.spatial_dropout,
                                            p_dropout=args.dropout_rate,
                                            lambda_l2=args.l2,

                                            
                                            n_hidden=args.n_hidden,
                                            metrics=args.metrics,

                                            )


    if args.verbose > 0:
        model.summary()

    if args.nogo:
        # Stop execution
        print("No execution")
        return

    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                        restore_best_weights=True,
                                                        min_delta=args.min_delta)
    
    # add learning rate reducer here

    # can add tensorboard here

    # Training (trains the model, either mlp or )
    if args.model_type == 'mlp':
        history = model.fit(x=ins, y=outs, 
                            epochs=args.epochs, 
                            # verbose=(args.verbose > 1),
                            verbose=args.verbose,
                            validation_data=(ins_validation, outs_validation), 
                            callbacks=[early_stopping_cb])

    #this mlp_prob is a Parametric Distributional Prediction (PDP)
    elif args.model_type == 'mlp_prob':
        
        if args.verbose == 2:
            print("I am before model.fit.")

        # Assuming you want to use zeros as a dummy target for sigma
        dummy_sigma_train, dummy_sigma_val = np.zeros_like(outs), np.zeros_like(outs_validation)
        
        if args.batch_size == 1:
            args.batch_size = ins.shape[0]
        history = model.fit(x=[ins, outs],
                            epochs=args.epochs, 
                            verbose=args.verbose,
                            validation_data=([ins_validation, outs_validation],), 
                            callbacks=[early_stopping_cb],
                            batch_size=args.batch_size)
        
        
        if args.verbose == 2:
            print("I am after model.fit.")



    # # Generate results data
    # args.c_year = str(testing_date_time[120])[:4] # was going to save c_year, but since i am saving args i am saving c_year in there so it would be redundent -hector (feb 8)(delete next time u come across this comment)

    results = {}
    results['fname_base'] = fbase
    results['args'] = args
    results['history'] = history.history
    results['fname_folder_base'] = folder_fbase

    results['modify_loss'] = ['MU:', str(args.modify_mu_loss), args.modify_mu_loss, 'SIGMA:', str(args.modify_sigma_loss),  args.modify_sigma_loss]

    results['x_test'] = x_test
    results['y_test'] = y_test
    
    results['x_train'] = ins
    results['y_train'] = outs

    results['x_val'] = ins_validation
    results['y_val'] = outs_validation


    results['model_path'] = folder_fbase + "%s_model"%(fbase)
    if args.verbose > 0: 
        print("Model path saved.")

    results['testing_data_dateAndTime'] = testing_date_time # + dateoffset needs double checking that its working properly # when was this comment made tho? -hector feb 6th 2025
    results['validation_data_dateAndTime'] = validation_date_time


    if args.model_type == 'mlp':
        results['predict_y_test'] = model.predict(x_test) 
        # results['model_eval_metrics'] = modelEvaluation(results['predict_y_test'], y_test)  

        results['model.evaluate'] = model.evaluate(x_test,y_test) # this line (as is) is impossible to run for the PNN due to the 2 output nuerons

    elif args.model_type == 'mlp_prob':
        
        if args.verbose > 1:
            print("before test predict.")
        
        mu_pred, sigma_pred = model.predict(list((x_test, x_test))) 
        results['predict_y_test_mu'] = mu_pred
        results['predict_y_test_sigma'] = sigma_pred
        
        if args.verbose > 1:
            print("after test predict.")

        

        if args.verbose > 1:
            print("before val predict.")
        
        mu_pred, sigma_pred = model.predict(list((ins_validation, ins_validation))) 
        results['predict_y_val_mu'] = mu_pred
        results['predict_y_val_sigma'] = sigma_pred        

        if args.verbose > 1:
            print("after val predict.")

        #
        #


        
        year = '2021'
        # year__data, year__target, date_time = load_for_testing(args.c_leadtime, args.atp_hours_back, args.wtp_hours_back, year=year, dataset='Full')
        ins, outs, ins_validation, outs_validation, x_test, y_test, date_time, _, val_date_time = preparingData(input_hours_forecast=args.c_leadtime, 
                                                                                atp_hours_back=args.atp_hours_back, 
                                                                                wtp_hours_back=args.wtp_hours_back,
                                                                                cycle=args.c_cycle,
                                                                                path_to_data=args.data_set,
                                                                                date_time=True,
                                                                                val_date_time=True,
                                                                                ind=year)

        results[f"{year}_data_dateAndTime"] = date_time
        results[f"x_{year}"] = x_test
        results[f"y_{year}"] = y_test

        year__predictions_mu, year__predictions_sigma = model.predict(list((x_test, x_test)))

        results[f"predict_y_{year}_mu"] = year__predictions_mu
        results[f"predict_y_{year}_sigma"] = year__predictions_sigma
        #--------------------------
        year = '2024'
        # year__data, year__target, date_time = load_for_testing(args.c_leadtime, args.atp_hours_back, args.wtp_hours_back, year=year, dataset='Full')
        ins, outs, ins_validation, outs_validation, x_test, y_test, date_time, _, val_date_time = preparingData(input_hours_forecast=args.c_leadtime, 
                                                                                atp_hours_back=args.atp_hours_back, 
                                                                                wtp_hours_back=args.wtp_hours_back,
                                                                                cycle=args.c_cycle,
                                                                                path_to_data=args.data_set,
                                                                                date_time=True,
                                                                                val_date_time=True,
                                                                                ind=year)

        results[f"{year}_data_dateAndTime"] = date_time
        results[f"x_{year}"] = x_test
        results[f"y_{year}"] = y_test

        year__predictions_mu, year__predictions_sigma = model.predict(list((x_test, x_test)))

        results[f"predict_y_{year}_mu"] = year__predictions_mu
        results[f"predict_y_{year}_sigma"] = year__predictions_sigma



        if args.verbose > 1:
            print("after val predict.")

    '''Save model'''
    #if args.save_model:
    # model.save(args.results_folder + "%s_model"%(fbase)) # not using this method anymore 
    model.save(folder_fbase + "%s_model.keras"%(fbase))
    if args.verbose > 0: 
        print("Model .keras saved successfully.")

    model.save(folder_fbase + "%s_model.h5"%(fbase)) # added since the rest of the Cool Turtles team (and the Operational team) use a .h5 file to save the model
    if args.verbose > 0: 
        print("Model .h5 saved successfully.")

    '''Save results dictionary'''
    with open(folder_fbase + "%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    if args.verbose > 0: 
        print("results dictionary saved successfully.")

    if args.verbose > 0:
        print(fbase)

    print(f"Finished Experiment num {args.experiment_number}. Name: {fbase}.")


    if args.enviroment == 'local':
        sys.stdout.flush()
        sys.stderr.flush()

def execute_experiment_wrapper(i, args, function):
    """
    This function serves as a wrapper around the execute_experiment function.
    It sets the experiment number and prints information (optional) about the 
    experiment before executing the actual experiment.

    Args:
    - i (int): The index of the current experiment.
    - args (Namespace): Arguments containing the parameters for the experiment.
    """
    args.experiment_number = i # Set the experiment number in the args object
    if args.verbose >= 1:
        print(f'exp num: {i}, current experiment number: {args.experiment_number}, total number of experiment: {len(args.cycle) * len(args.leadtime) * args.repetitions}') 
    function(args) # Execute the actual experiment using the provided args

def configue_gpus(verbose=0, use_gpu=False):
    if ("CUDA_VISIBLE_DEVICES" in os.environ.keys()) or (use_gpu):
        # Fetch list of logical GPUs that have been allocated
        #  Will always be numbered 0, 1, …
        physical_devices = tf.config.get_visible_devices('GPU')
        n_physical_devices = len(physical_devices)

        # Set memory growth for each
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
        if verbose>=1:
            print()
            print("Using any available gpu's.")
    else:
        # No allocated GPUs: do not delete this case!                                                                	 
        # if you dont need a gpu, 

        # this line tells tf to ignore any available gpu's
        tf.config.set_visible_devices([], 'GPU')

        if verbose>=1:
            print("Ignoring any available gpu's.")

def run_experiments(args, function):
    """
    Runs all experiments using multiprocessing (CPU parrallel execution approach). 
    The experiments are distributed across multiple processes to run concurrently 
    with a limit of 5 concurrent processes.

    Args:
    - args (Namespace): Arguments containing the parameters for the experiment.
    """
    num_of_total_experiments = len(args.cycle) * len(args.leadtime) * args.repetitions # Calculate the total number of experiments based on input parameters
    
    # Create a pool of 5 processes to run experiments concurrently
    with mp.Pool(processes=args.pool) as pool:
        # Submit tasks asynchronously
        for i in range(num_of_total_experiments):
            pool.apply_async(execute_experiment_wrapper, args=(i, args, function))

        pool.close() # Close the pool to prevent any new tasks from being submitted
        pool.join() # Wait for all tasks to complete before moving forward

if __name__ == "__main__":

    # Parse incoming command-line arguments
    parser = create_parser()
    args = parser.parse_args()


    # Main code block to check environment and run experiments
    if args.enviroment == 'local':
        configue_gpus(verbose=args.verbose, use_gpu=True)
        # catch_exceptions(run_experiments, args)

        if args.parallel_computing == True:
            run_experiments(args, execute_experiment) # If running locally, use multiprocessing to run experiments

        else:
            num_of_total_experiments = len(args.cycle)*len(args.leadtime)*args.repetitions
            for i in range(num_of_total_experiments): #do the work
                args.experiment_number = i
                print('exp num:', i, args.experiment_number)
                print('total num:', num_of_total_experiments)
                execute_experiment(args) # Execute the actual experiment using the provided args

    else:    
        configue_gpus(verbose=args.verbose)
        execute_experiment(args) # If not running locally (e.g., on schooner), run the experiment directly

    # os.system("python -m src.helper.pnn_to_csv @configs/pnn_12h.txt") # unsure if i want this here -hector 7-10-2025

