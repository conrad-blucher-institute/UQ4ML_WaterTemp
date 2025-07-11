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
                                n_filters=None,  #[10], # for CNN ; conv stack
                                kernel_size=None, #[3], # for CNN ; conv stack
                                pooling=None, #[1], # for CNN ; conv stack
                                n_hidden=None, #[5]
                                metrics=None,
                                modify_sigma_loss=None,
                                sigma_threshold=None,
                                sigma_regularization_parameter=None,

                                modify_mu_loss=None,
                                mu_threshold=None,
                                mu_regularization_parameter=None,
                                
                                path=None): 

    # create our custom loss function
    def mdn_cost(mu, sigma, y):
        '''
        
        '''
        # print('shape before converting a kerastensor to a tensor', mu.shape, sigma.shape)

        # # mu = tf.convert_to_tensor(mu)
        # # sigma = tf.convert_to_tensor(sigma)
        
        # # mu = tf.keras.backend.identity(mu)
        # # sigma = tf.keras.backend.identity(sigma)

        # print('shape after converting a kerastensor to a tensor', mu.shape, sigma.shape)

        dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        return tf.reduce_mean(-dist.log_prob(y))

    def loss_uq_normal(y_true, y_pred):
        """
        This is the log-probability loss to calculate uncertainty
        with a normal distribution.
        This form predicts the uncertainty mean and standard deviation
        for a normal distribution.
        From Barnes, Barnes, & Gordillo (2021).
        """

        y_pred64 = tf.cast(y_pred, tf.float64)
        y_true64 = tf.cast(y_true, tf.float64)

        # network prediction of the value
        mu = y_pred64[..., 0]

        # network prediction of uncertainty
        print(f"y_pred64.shape: {y_pred64.shape}")
        print(f"whats inside[0]: {y_pred64[0]}")
        print(f"whats inside[1]: {y_pred64[1]}")
        std = tf.math.exp(y_pred64[..., 1])

        # normal distribution defined by N(mu,sigma)
        norm_dist = tfp.distributions.Normal(mu, std)

        # compute the log as -log(p)
        loss = -norm_dist.log_prob(y_true64)

        return tf.reduce_mean(loss)

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


    # Output Nodes
    mu = Dense(1, name="mu", activation="linear")(layer1)
    sigma = Dense(1, name="sigma", activation=lambda x: tf.nn.elu(x) + 1.1)(layer1) # added a .1 so that it will never be negative or zero, minimum will be 0.1

    
    # Loss Function
    y_real = keras.layers.Input(shape=(1,), name="y_real_input") 
    lossF = mdn_cost(mu, sigma, y_real)

    # Build Model
    if path != None:
        model = tf.keras.models.load_model(path, compile=False)
    else:
        model = keras.models.Model(inputs=[genericInputLayer, y_real], outputs=[mu, sigma]) # btw keras.Model & keras.models.Model are equivalent
    
    model.add_loss(lossF)

    '''new experiment'''
    if modify_sigma_loss:
        print("INSIDE MODIFY SIGMA")
        # threshold = 2
        # regularization_parameter = 0.1
        threshold = sigma_threshold
        regularization_parameter = sigma_regularization_parameter

        error = tf.reduce_mean(tf.math.maximum(threshold-sigma, 0)) # try 2, 1, 0.5, 0.25, 0.1
        model.add_loss(regularization_parameter * error) #0.1 = regularization parameter

    if modify_mu_loss:
        print("INSIDE MODIFY MU")
        # threshold = 0.5
        # regularization_parameter = 0.2
        threshold = mu_threshold
        regularization_parameter = mu_regularization_parameter

        penalty = tf.where(mu < threshold, tf.square(threshold - mu), 0.0) # Penalize predictions below 0.5
        model.add_loss(tf.reduce_mean(penalty) * regularization_parameter)



    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                    amsgrad = False)
    
    model.summary()  # Print the summary of the neural network

    # Bind the model to the optimizer

    metrics_func = [ryan_ssrel, ssrat_avg, pitd, mae, mse, mae12, me12, me, errorBelow12c, max10PercentError]

    model.compile(  
                    optimizer=opt,
                    metrics=metrics)
                    #metrics=['categorical_accuracy'])

    return model
def create_classifier_network_generic(  input_shape=None,
                                
                                nchannels=None, # for CNN ; conv layers
                                num_output_neurons=None, 
                                learning_rate=None, 
                                lambda_l2=None, # None or a float
                                loss_function=None,

                                activation_function=None,

                                p_dropout=None,
                                p_spatial_dropout=None,
                                n_filters=None,  #[10], # for CNN ; conv stack
                                kernel_size=None, #[3], # for CNN ; conv stack
                                pooling=None, #[1], # for CNN ; conv stack
                                n_hidden=None, #[5]
                                metrics=None): 



    # if lambda_l2 is not None: regularizer = tf.keras.regularizers.l2(lambda_l2) else: regularizer = None # assume a float
    regularizer = tf.keras.regularizers.l2(lambda_l2) if lambda_l2 is not None else None

    # model = Sequential()
    genericInputLayer = keras.layers.Input(input_shape)

    '''####################'''
    # in progress block of code to replace block below
    if len(input_shape) == 1:
        if args.verbose > 0: 
            print("inside len == 1 mlp")
        # how we did it in the lab before 
        # layer1 = keras.layers.Dense(units=input_neurons, activation=activation_function, kernel_regularizer=regularizer, input_shape=input_shape)(genericInputLayer)
        # model.add(Dense(units=input_neurons, activation=activation_function, kernel_regularizer=regularizer, input_shape=input_shape))
        
        
        for i, v in enumerate(n_hidden):
            x = keras.layers.Dense(units=v, activation=activation_function, kernel_regularizer=regularizer)(x)
        

        if p_dropout is not None:
            layer1 = Dropout(p_dropout)(layer1)
            # model.add(Dropout(p_dropout))
    layer1 = x
        
    '''@@@@@@@@@@@@@@@@@@@'''
    # Output
    y_output = Dense(units=num_output_neurons, activation=activation_function, input_shape=input_shape)(layer1)
    # y_output = Dense(units=num_output_neurons, activation=activation_function, input_shape=input_shape)(layerDropout) if p_dropout is not None else (Dense(units=num_output_neurons, activation=activation_function, input_shape=input_shape))(layer1)
    # model.add(Dense(units=num_output_neurons, activation=activation_function, input_shape=input_shape))
    '''@@@@@@@@@@@@@@@@@@@'''

    model = keras.models.Model(inputs=[genericInputLayer], outputs=y_output)

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                    amsgrad = False)
    
    if args.verbose >= 1:
        model.summary()  # Print the summary of the neural network

    # Bind the model to the optimizer
    model.compile(loss=loss_function,
                    optimizer=opt,
                    metrics=metrics)
                    #metrics=['categorical_accuracy'])

    
    # plot_model(
    #                                 model,
    #                                 to_file='model.png',
    #                                 show_shapes=True,
    #                                 show_dtype=False,
    #                                 show_layer_names=True,
    #                                 rankdir='TB',
    #                                 expand_nested=False,
    #                                 dpi=96,
    #                                 layer_range=None,
    #                                 show_layer_activations=False,
    #                                 show_trainable=False
    #                             )

    return model
def load_MLP_dataset(args):
    ins, outs, ins_validation, outs_validation, x_test, y_test, date_time, _, val_date_time = preparingData(input_hours_forecast=args.c_leadtime, 
                                                                                atp_hours_back=args.atp_hours_back, 
                                                                                wtp_hours_back=args.wtp_hours_back,
                                                                                cycle=args.c_cycle,
                                                                                path_to_data=args.data_set,
                                                                                date_time=True,
                                                                                val_date_time=True)
    return ins, outs, ins_validation, outs_validation, x_test, y_test, date_time, val_date_time
def create_classifier_network(  input_shape=None,

                                
                                nchannels=None, # for CNN ; conv layers
                                num_output_neurons=None, 
                                learning_rate=None, 
                                lambda_l2=None, # None or a float
                                loss_function=None,

                                activation_function=None,


                                p_dropout=None,
                                p_spatial_dropout=None,
                                n_filters=None,  #[10], # for CNN ; conv stack
                                kernel_size=None, #[3], # for CNN ; conv stack
                                pooling=None, #[1], # for CNN ; conv stack
                                n_hidden=None, #[5]
                                metrics=None): 



    if lambda_l2 is not None:
        # assume a float
        regularizer = tf.keras.regularizers.l2(lambda_l2)
    else:
        regularizer = None


    model = Sequential()

    '''####################'''
    # in progress block of code to replace block below
    if len(input_shape) == 1:
        if args.verbose > 0:
            print("inside len == 1 mlp")
        # how we did it in the lab before 
        # model.add(Dense(units=input_neurons, activation=activation_function, kernel_regularizer=regularizer, input_shape=input_shape))
        
        for i, v in enumerate(n_hidden):
            model.add(Dense(units=v, activation=activation_function, kernel_regularizer=regularizer, input_shape=input_shape))
        

        if p_dropout is not None:
            model.add(Dropout(p_dropout))
        # following Dr. Faggs example but changed to try and accomadate my mlp dataset
        # model.add(InputLayer(input_shape=input_shape))
        pass#mlp
    elif len(input_shape) == 2:
        if args.verbose > 0:
            print("inside len == 2 cnn")
        ## image_size and nchannels are needed for convolution (when the data are images)
        model.add(InputLayer(input_shape=(input_shape[0], input_shape[1], nchannels)))


    # convolutional layers
    ## n_filters and kernel_size and pooling
    if n_filters is not None:
        for i, (n, s, p) in enumerate(zip(n_filters, kernel_size, pooling)):
            model.add(Convolution2D(filters=n,
                                kernel_size=s,
                                padding='same',
                                use_bias=True,
                                kernel_regularizer=regularizer,
                                name='C%d'%(i),
                                activation='elu'))
            
            if p_spatial_dropout is not None:
                model.add(SpatialDropout2D(p_spatial_dropout))
                
            if p > 1:
                model.add(MaxPooling2D(pool_size=p,
                                        strides=p,
                                        name='MP%d'%(i)))
            
        
        # Flatten
        model.add(GlobalMaxPooling2D())
    

    if n_hidden is not None:
        # Dense layers
        for i,n in enumerate(n_hidden):
            model.add(Dense(units=n,
                        activation=activation_function,
                        use_bias='True',
                        kernel_regularizer=regularizer,
                        name='D%d'%i))
            
            if p_dropout is not None:
                model.add(Dropout(p_dropout))
        
    '''@@@@@@@@@@@@@@@@@@@'''
    # Output
    # model.add(Dense(units=num_output_neurons,
    #                 activation='softmax',
    #                 use_bias='True',
    #                 kernel_regularizer=regularizer,
    #                 name='output'))
    
    model.add(Dense(units=num_output_neurons, activation=activation_function, input_shape=input_shape))
    '''@@@@@@@@@@@@@@@@@@@'''

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                    amsgrad = False)
    
    # Bind the model to the optimizer
    model.compile(loss=loss_function,
                    optimizer=opt,
                    metrics=metrics)
                    #metrics=['categorical_accuracy'])

    return model


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


    if False: # this code was used only once to remove the ex num from the file and folder name of the existing results
        # Loop through all files in the folder
        for item in os.listdir(folder_fbase):
            # Search for _EX_NUM_<number> pattern (e.g., _EX_NUM_20 or _EX_NUM_100)
            match = re.search(r'_EX_NUM_(\d+)', item)
            if match:
                # Get the number part
                ex_num = match.group(1)
                # Remove the _EX_NUM_<number> part
                new_item_name = item.replace(f'_EX_NUM_{ex_num}_', '')

                # Rename the file (same name but without _EX_NUM_<number>)
                os.rename(os.path.join(folder_fbase, item), os.path.join(folder_fbase, new_item_name))
                print(f'Renamed: {item} -> {new_item_name}')


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

    # for debugging
    if args.verbose >= 2: 
        print("")
        print("checking ins.shape", ins.shape)
        print("checking ins[0].shape", ins[0].shape)
        print("checking outs.shape", outs.shape)
        print("checking outs[0].shape", outs[0].shape)
        print("")

    ## hector, this is really inelegant, fix it later future_hector.
    if args.model_type == 'mlp':
        model = create_classifier_network_generic(
                                            input_shape=ins[0].shape, 
                                            
                                            num_output_neurons=args.num_output_neurons, 
                                            learning_rate=args.lrate,
                                            loss_function=args.loss_function,
                                            activation_function=args.activation_function,
                                            p_spatial_dropout=args.spatial_dropout,
                                            p_dropout=args.dropout_rate,
                                            lambda_l2=args.l2,
                                            
                                            n_hidden=args.n_hidden,
                                            metrics=args.metrics
                                            )
        
    elif args.model_type == 'mlp_prob':
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

                                            modify_sigma_loss=args.modify_sigma_loss,
                                            sigma_threshold=args.sigma_threshold,
                                            sigma_regularization_parameter=args.sigma_regularization_parameter,

                                            modify_mu_loss=args.modify_mu_loss,
                                            mu_threshold=args.mu_threshold,
                                            mu_regularization_parameter=args.mu_regularization_parameter
                                            )

    elif args.model_type == 'cnn':
        model = create_classifier_network(
                                            input_shape=(ins.shape[1], ins.shape[2]), 
                                            nchannels=ins.shape[3], 
                                            
                                            num_output_neurons=args.num_output_neurons, 
                                            learning_rate=args.lrate,
                                            p_spatial_dropout=args.spatial_dropout,
                                            p_dropout=args.dropout_rate,
                                            lambda_l2=args.l2,
                                            n_filters=args.n_filters,
                                            kernel_size=args.kernel_sizes,
                                            pooling=args.pooling,
                                            n_hidden=args.n_hidden,
                                            metrics=args.metrics
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
        
        # history = model.fit(x=ins, 
        #                     y=outs,
        #                     epochs=args.epochs, 
        #                     verbose=args.verbose,
        #                     validation_data=(ins_validation,
        #                                      outs_validation), 
        #                     callbacks=[early_stopping_cb])
        # history = model.fit(x=[ins, ins], 
        #                     y=[outs, dummy_sigma_train],
        #                     epochs=args.epochs, 
        #                     verbose=args.verbose,
        #                     validation_data=([ins_validation, ins_validation],
        #                                      [outs_validation, dummy_sigma_val]), 
        #                     callbacks=[early_stopping_cb])
        if args.batch_size == 1:
            args.batch_size = ins.shape[0]
        history = model.fit(x=[ins, outs],
                            epochs=args.epochs, 
                            verbose=args.verbose,
                            validation_data=([ins_validation, outs_validation],), 
                            callbacks=[early_stopping_cb],
                            batch_size=args.batch_size)
        # history = model.fit([(ins, outs)],
        #                     epochs=args.epochs, 
        #                     verbose=args.verbose,
        #                     validation_data=([ins_validation, outs_validation],), 
        #                     callbacks=[early_stopping_cb])
        
        if args.verbose == 2:
            print("I am after model.fit.")

    elif args.model_type == 'cnn':
        pass 
    '''will fill in if ever need to run a cnn'''




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
        mu_pred, sigma_pred = model.predict(list((x_test, x_test))) 
        results['predict_y_test_mu'] = mu_pred
        results['predict_y_test_sigma'] = sigma_pred
        # results['model_eval_metrics'] = modelEvaluation(results['predict_y_test_mu'], y_test)

        mu_pred, sigma_pred = model.predict(list((ins_validation, ins_validation))) 
        results['predict_y_val_mu'] = mu_pred
        results['predict_y_val_sigma'] = sigma_pred


    # Generate predictions for the year 2021
    if args.verbose > 0: 
        print('before predicted the year 2021')
    # result_visualizer.loadmodel_test_on_year(results, model, year='2021', probability=True, filePath='cmd_ai_builder')
    if args.verbose > 0: 
        print('successfully predicted the year 2021')





    '''Save model'''
    #if args.save_model:
    # model.save(args.results_folder + "%s_model"%(fbase))
    # model.save(folder_fbase + "%s_model"%(fbase)) # this method to save the models has been deprecated
    model.save(folder_fbase + "%s_model.keras"%(fbase))
    model.save(folder_fbase + "%s_model.h5"%(fbase)) # added since the rest of the Cool Turtles team (and the Operational team) use a .h5 file to save the model
    if args.verbose > 0: 
        print("Model saved successfully.")

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
    '''

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
    '''

    def catch_exceptions(func, arg_list):
        try:
            func(arg_list) # If running locally, use multiprocessing to run experiments

        except KeyboardInterrupt:
            sys.exit(0) # 0 means succesful exit
        except Exception as e:
            print(f"Unhandled Exception {e}")
            sys.exit(1) # non 0 number means error/failure


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

