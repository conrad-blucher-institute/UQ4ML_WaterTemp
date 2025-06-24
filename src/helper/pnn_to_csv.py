'''
Example usage of this script:
$ python result_visualizer.py @models/PNN_customLoss-Mu.txt -e -v
$ python result_visualizer.py @models/PNN_customLoss-Mu.txt -e --repetitions=5 --test_only
'''
# r = results[] (dictionary with results)

# Visualizercolor_list12
#### Imports
import tensorflow as tf
import pickle
import re
import os
import pandas as pd

from pathlib import Path

from src.helper.utils import readingData
from src.helper.my_parser import create_parser
from src.helper.utils import load_for_testing

## THE COLOR

# THECOLOR = '#E7C65D'

# helper function
def changeArrayDepthTo1(mu=None, sigma=None):
    if mu is not None:
        list_mu = []
        for i in range(len(mu)):
            list_mu.append(mu[i][0])


    if sigma is not None:
        list_sigma = []
        for i in range(len(sigma)):
            list_sigma.append(sigma[i][0])


    if mu is not None and sigma is not None:
        return list_mu, list_sigma
    elif mu is not None:
        return list_mu
    elif sigma is not None:
        return list_sigma

def loadmodel_test_on_year(results, modelPath, year=None, probability=False, filePath=None, verbose=0):
    if filePath == None: raise NameError("filePath inside loadmodel_test2021 is none")
    if year == None: raise NameError("year inside loadmodel_test2021 is none")

    args = results['args']

    year__data, year__target, date_time = load_for_testing(args.c_leadtime, args.atp_hours_back, args.wtp_hours_back, year=year, dataset='Full')


    if verbose > 0:
        '''for testing purposes'''
        print("Input shape:", model.layers[0].input_shape)
        print("Input shape year__data:", year__data[0].shape)
        print('\n\nChecking params in loadmodel test on year: LT', args.c_leadtime, '_atp_hb_', args.atp_hours_back, '_wtp_hb_', args.wtp_hours_back, 'year:',year, 'filePath:',filePath, '\n\n')



    from src.driver.pnn_mme_driver import create_classifier_network_generic_probability
    model = create_classifier_network_generic_probability(
                        input_shape=year__data[0].shape, 

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
                        mu_regularization_parameter=args.mu_regularization_parameter,

                        path=modelPath)


    results[year + '_testing_data_dateAndTime'] = date_time
    results[year + '_x_test'] = year__data
    results[year + '_y_test'] = year__target

    if probability == True:
        year__predictions_mu, year__predictions_sigma = model.predict(list((year__data, year__data)))

        results[year + '_predict_y_test_mu'] = year__predictions_mu
        results[year + '_predict_y_test_sigma'] = year__predictions_sigma
        
    else:
        year__predictions = model.predict(year__data)

        results[year + '_predict_y_test'] = year__predictions
        

    if filePath == 'cmd_ai_builder': 
        return
    with open(filePath, "wb") as fp:
        pickle.dump(results, fp)
            # return year_2021_predictions
        # new_model_prob = tf.keras.models.load_model("old_results/results_beforeJuly11/amsSM_prob___/_LT_120_/_cycle_6_/results_LT_120__cycle_6__rep_num_0__EX_NUM_30__LR_0.000100_L2_0.010000_model/")

def looper(leadtime, cycle, directory, numTrials, verbose=0, independent=False):#, myFunction, myFunction_args):
    pickles = []
    if verbose>0:
        print('inside looper, heres pickles list:',pickles)

    for lt in leadtime:
        if verbose==1:
            print('lt:',lt)

            
        folder = 'UQ4ML_WaterTemp/src/results/pnn_results/' 

        folder += 'results/' + directory + '/_LT_' + str(lt) + '_/'
        
        # folder = f"E:/pnn_tuning/Phase_3_7zip/{directory}/_LT_{lt}_/" # ssd path 

        for c in cycle:
            if verbose==1:
                print('cycle:',c)


            for i in range(numTrials):

            
                
                if verbose==1:
                    print('numTrial:',i)

                # filePath = folder + pickle_files[i]
                # modelPath = folder + model_folders[i]
                
                filePath = f"results_LT_{lt:03d}__cycle_{c}__rep_num_{i:03d}__\\results_LT_{lt:03d}__cycle_{c}__rep_num_{i:03d}__results.pkl"
                modelPath = f"results_LT_{lt:03d}__cycle_{c}__rep_num_{i:03d}__\\results_LT_{lt:03d}__cycle_{c}__rep_num_{i:03d}__model.h5"

                if verbose == 2:
                    print("\nin looper: leadtime:", lt, "cycle:", c, "index/pickle file number:", i)
                    # print("filePath:", filePath, "\nmodelPath:", modelPath)

                with open(folder + filePath, "rb") as fp:
                    r = pickle.load(fp)

                    if independent:
                        # loadmodel_test_on_year(r, model, year='2021', probability=True, filePath='cmd_ai_builder', verbose=0)
                        # loadmodel_test_on_year(r, model, year='2024', probability=True, filePath='cmd_ai_builder', verbose=0)
                        loadmodel_test_on_year(r, modelPath=folder+modelPath, year='2021', probability=True, filePath=folder + filePath, verbose=0)
                        loadmodel_test_on_year(r, modelPath=folder+modelPath, year='2024', probability=True, filePath=folder + filePath, verbose=0)

                    pickles.append(r)
                    # myFunction(myFunction_args)
    if verbose == 2:
        print("\ndone loading pickles\n")
    return pickles


# postprocessing
if __name__ == "__main__":
    # Parse incoming command-line arguments (same one as in cmd_ai_builder.py)
    parser = create_parser()
    
    parser.add_argument('--histogram',              action='store_true',                help="should results_visualizer draw histograms--default=false")
    parser.add_argument('--everything',     '-e',   action='store_true', default=False, help="should results_visualizer draw everything--default=true")
    parser.add_argument('--sigma_sum',              action='store_true',                help="should results_visualizer draw sigma_sumation--default=false")
    parser.add_argument('--test_only',      '-t',   action='store_true',                help="should results_visualizer execute a test--default=false")
    parser.add_argument('--csv_for_jarett', '-c',   action='store_true',                help="run a loop that creates csvs to send to jarett, that have metric and obsVsPreds for the Validation years")
    parser.add_argument('--calc_percent_in_range',  action='store_true',                help="should results_visualizer execute calc_percent_in_range function--default=false")
    parser.add_argument('--predict2021',    '-p',   action='store_true', default=False, help="should results_visualizer load and run predictions for independent testing year 2021--default=false")
    parser.add_argument('--skip_reg_j',             action='store_true',                help="should results_visualizer skip regular testing years--default=false")
    parser.add_argument('--skip_2021_j',            action='store_true',                help="should results_visualizer skip independent testing year 2021--default=false")
    parser.add_argument(                    '-I',   action='store_true',                help="Should we run the independent testing years? (2021 & 2024 as of April 9th 2025)")

    args = parser.parse_args()
    
    ''' looper function'''
    pickles = looper(leadtime=args.leadtime, cycle=args.cycle, directory=args.results_folder, numTrials=args.repetitions, verbose=args.verbose, independent=args.I)
    
    '''move below 2 lines to a notebook "testing" enviroment, should not be in this file, this file should be the "finished" enviroment'''
    # graph_a_season("Cold")
    # graph_a_season("Full")

    # a,b,c,d,e,f,g,h,i,j = readingData(args.data_set) # loads all 10 years of our dataset
    # allYears = [a,b,c,d,e,f,g,h,i,j] # creates a list for ease of use

    
    # need to load in the data_set, save the x and y val to pickle, run predict on the x and y val and save it, then create the cvs's jarett needs.
    print('len of pickles:',len(pickles))
    '''this test_only was used to create csv files for jarett to visualize, task was originally due by the cool turtle meeting on jan 31 2025'''
    for j in range(1):
        for index_of_pickles, r in enumerate(pickles): # go through all the models in the experiment
            model_args = r['args'] # sets this models args to a variable for easier access

            # Name of model saved 
            model_name = f"results_{model_args.results_folder}_LT_{model_args.c_leadtime}__cycle_{model_args.c_cycle}__rep_num_{model_args.c_repetitions}"

            # Path to folder
            model_folder = f"results/{model_args.results_folder}_csv/_LT_{model_args.c_leadtime}_/{model_name}_/"
            
            Path(model_folder).mkdir(parents=True, exist_ok=True)   # create path for results; so directory wont get flooded

            if model_args.c_leadtime == 12:
                combo = 'combo2'
            if model_args.c_leadtime == 48:
                combo = 'combo1'
            if model_args.c_leadtime == 96:
                combo = 'combo1'


            path_to_csv = f"UQ4ML_WaterTemp/src/results/pnn_results/{model_args.c_leadtime}h/pnn-{combo}-cycle_{model_args.c_cycle}-iteration_{model_args.c_repetitions+1}/"
            Path(path_to_csv).mkdir(parents=True, exist_ok=True)

            # Path to where model is saved 
            # modelPath = model_folder + model_name

            print('\n\nHere is the model path:',r['model_path'])

            r['test_data_dateAndTime'] = r['testing_data_dateAndTime']
            r['val_data_dateAndTime'] = r['validation_data_dateAndTime']

            for item in ['val', 'test']:

                mu_pred = r[f"predict_y_{item}_mu"]
                sigma_pred = r[f"predict_y_{item}_sigma"]
                
                mu_pred, sigma_pred = changeArrayDepthTo1(mu=mu_pred, sigma=sigma_pred)

                from src.helper.utils import mae # need mae
                mae = tf.reduce_mean(tf.abs(tf.subtract(r[f"y_{item}"], mu_pred))).numpy()
                print('mae:',mae)                
                r[f"metric_{item}_mae"] = mae

                from src.helper.utils import mae12 # need mae<12
                mae12 = mae12(r[f"y_{item}"], mu_pred)
                print('mae12:',mae12)
                r[f"metric_{item}_mae12"] = mae12

                #       Create DataFrame  
                df = pd.DataFrame({"": [f"{item}_metrics"], 
                                    "mae": mae, 
                                    "mae12": mae12})  
                #       Save to CSV  
                df.to_csv(model_folder + f"metrics_{item}.csv", index=False)
                df.to_csv(path_to_csv + f"metrics_{item}.csv", index=False)

                #       Create DataFrame  
                df = pd.DataFrame({"date_time": r[f"{item}_data_dateAndTime"], 
                                    "target": r[f"y_{item}"], 
                                    "pred_mu_1": mu_pred, 
                                    "sigma_1": sigma_pred})  
                #       Save to CSV  
                df.to_csv(model_folder + f"{item}_datetime_obsv_predictions.csv", index=False)
                df.to_csv(path_to_csv + f"{item}_datetime_obsv_predictions.csv", index=False)
