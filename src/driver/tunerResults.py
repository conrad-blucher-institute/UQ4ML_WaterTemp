'''
File to read in and save the val_mae results from the hyperparameter tuner

The idea will be to save everything, in order following an excel template, 
to a csv such that we can then transfer everything with a single copy and paste action.
'''


import json


def temp(args):
    # cycle_list = [1,3,6,7,9] #[0, 2, 5, 8]
    # cycle_list = [0, 1, 2, 3] #[0, 2, 5, 8]
    runs = 2 # number of iterations per hours back 
    hours_back = [24] #[6, 12, 24]
    max_trials = 30 # number of trials within each iteration
    results = [] # variable to store val_mae scores
    bestObjective = {} # dictionary to hold the val_mae ('MAE') values 
    num_mae_values = len(args.rotation_list) * runs * len(hours_back) * args.max_trials # variable to store range of total number of 'val_mae' values

    # directory name
    # dataset = r'C:\Users\cduff4\MAPE_72h_CD_TUNER\\'
    dataset = args.results_folder
    leadtime = args.leadtime_list
    # metrics = ['loss', 'mae', 'mae12', 'mape', 'val_mae', 'val_mae12', 'val_mape']
    # metrics = args.metrics
    objective = args.tuner_objective
    allTrials = True

    # Loop to create string for metrics added
    string = ""
    for item in args.metrics:
        string += ',' + str(item)
        
    #Path Name identifier to differentiate bewteen best and all trials
    if allTrials == True:
        identifier = "ALL_Trials"
    else:
        identifier = "best_trials"

    file = open(dataset + '_' + str(leadtime) + 'h_hyperparameters_' + objective + "_" + identifier+ '.txt', 'w')
    file2 = open(dataset + '_' + str(leadtime) + 'h_hyperparametersExcel_' + objective + "_" + identifier+'.csv', 'w')
    file2.write('Rotation:' + ',' + 'Iteration:' + ',' + 'Trial Number:' + ',' + objective +' Score:' + ',' + 'layer_units_' + 'num' + ','

    # Original line
    #file2.write('Cycle: ' + ',' + 'num' + ',' + 'Hours back: ' + ',' + 'num' + ',' + 'Iteration: ' + ',' + 'num' + ',' + 'Trial Number: ' + ',' + 'num' + ',' + 'Val_Mae Score: ' + ',' + 'num' + ',' + 'input_layer_units_' + 'num' + ','            

    + 'input act. func.'         + ','
    + '# of layers'              
    + string +'\n')
    """
    ','
    + 'optimizer'                + 'num' + ','
    + 'learning rate'            + 'num' + ','
    + 'batch size'               + 'num' + ','
    + '\n\n')
    """
    # loop to reach within each specific file
    for rotation in args.rotation_list:
        print('rotation', rotation)
        for h in hours_back:
            print('hours_back: ', h)
            for r in range(runs):
                print('Iteration: ', r+1)
                for m in range(args.max_trials):
                    print('Trial: ', m)
                    
                    # location = '../Cycle_' + str(c+1) + '_MLP Tuner/' + str(h) + '_hours_back_ iteration_' + str(r) + '/trial_' + str(m) + '/trial.json'
                    
                    if m < 10:
                        m = str("0" + str(m))

                    # storing location of specific file
                    location = f"{dataset}/Iter_{str(r+1)}_{leadtime}h_Rotation_{rotation}/results/trial_{str(m)}/trial.json"
                    
                    # reading dictionary from JSON file
                    with open(location) as access_json:
                        read_contents = json.load(access_json)
                        
                    # Debugging code to see if parameters are different across cycles
                    print(read_contents['hyperparameters']['values'])
                    print(read_contents['metrics']['metrics'][objective]['observations'][0]['value'][0])
        
                    # Checking the status of the trial files 
                    status = read_contents['status']
                
                    # if status == 'COMPLETED', then the hyperparameters chosen worked with the model
                    if status == 'COMPLETED':

                        # grabs MAE
                        bestObjective[m] = read_contents['metrics']['metrics'][objective]['observations'][0]['value'][0]

                    # If status == 'FAILED', then the hyperparameters chosen broke the model and are not usable
                    elif status == 'FAILED':

                        # setting bestMAE to a number that is large enough that it will be discarded once the minimum value of the dictionary is chosen later on
                        bestObjective[m] = 999


                    print(bestObjective)
                    
                # Prints the trial containing lowest MAE
                print(min(bestObjective, key = bestObjective.get))
                
                if allTrials == True:
                    for trial in bestObjective:
                        # Grabs the location of the min trial
                        newLocation = location = f"{dataset}/Iter_{str(r+1)}_{leadtime}h_Rotation_{rotation}/results/trial_{str(trial)}/trial.json"

                        with open(newLocation) as newAccess_json:
                            contents_read = json.load(newAccess_json)
                        
                        # Grabs the hyperparameter values from the best trial
                        hyperparameters = contents_read['hyperparameters']['values']
                        best_valObjective = contents_read['metrics']['metrics'][objective]['observations'][0]['value'][0]

                        # Writing the trials with the best val_mae score for each number of hours back per iteration for each cycle
                        file.write('Rotation: ' + str(rotation) + '\n' + 'Iteration: ' + str(r + 1) + '\n' + 'Trial Number: ' + str(min(bestObjective, key = bestObjective.get)) + '\n' +  str(objective) + 'score: ' + str(best_valObjective) + '\n' + str(hyperparameters) + '\n\n')

                        # order is
                        # val_mae, #of input units, 
                        # input act. func., #of layers, 
                        # #units per layer, act. func. per layer, /
                        # optimizer, learning rate, batch size
                        
                        input_layer_units = contents_read['hyperparameters']['values']['neurons_']
                        input_act_func = contents_read['hyperparameters']['values']['act_']
                        numLayers = contents_read['hyperparameters']['values']['layers']

                        
                        #List for storing retrieved metrics
                        metricString =""
                        
                        # This loop will be for grabbing and writing the evaluation metrics
                        for item in args.metrics:
                        
                            value = contents_read['metrics']['metrics'][item]['observations'][0]['value'][0]

                            value = "," + str(value)
                        
                            # Adds value to end of list
                            metricString += value
                            
                        
                        #optimizerChosen = contents_read['hyperparameters']['values']['optimizer']
                        #learningRateChosen = contents_read['hyperparameters']['values']['learning_rate_']
                        #batch_size_chosen = contents_read['hyperparameters']['values']['batch_size']

                        file2.write(str(rotation) + ',' +  str(r + 1) + ',' + str(trial) + ',' + str(best_valObjective) + ',' 
                                    
                        # Oriignal line for dealing with hours back
                        #file2.write('Cycle: ' + ',' + str(c + 1) + ',' + 'Hours back: ' + ',' + str(h) + ',' + 'Iteration: ' + ',' + str(r + 1) + ',' + 'Trial Number: ' + ',' + str(min(bestMAE, key = bestMAE.get)) + ',' + 'Val_Mae Score: ' + ',' + str(best_valmae) + ',' 
                        + str(input_layer_units) + ','
                        + str(input_act_func) + ','
                        + str(numLayers) 
                        + metricString +'\n') 
                        
                        # Removed extraneous things
                        """ + ','
                        + str(optimizerChosen) + ','
                        + str(learningRateChosen) + ','
                        + str(batch_size_chosen)
                        + '\n')"""
                        
                else:
                        
                    # Grabs the location of the min trial
                    newLocation = dataset +"/" + str(leadtime) +'h_LeadTime_Rotation_'  + str(rotation+1) + '_MLPTuner/' + str(h) + '_hours_back_ iteration_' + str(r+1) + '/trial_' + str(min(bestObjective, key = bestObjective.get)) + '/trial.json'
        
                    with open(newLocation) as newAccess_json:
                        contents_read = json.load(newAccess_json)
                    
                    # Grabs the hyperparameter values from the best trial
                    hyperparameters = contents_read['hyperparameters']['values']
                    best_valObjective = contents_read['metrics']['metrics'][objective]['observations'][0]['value'][0]
        
                    # Writing the trials with the best val_mae score for each number of hours back per iteration for each cycle
                    file.write('Rotation: ' + str(rotation) + '\n' + 'Iteration: ' + str(r + 1) + '\n' + 'Trial Number: ' + str(min(bestObjective, key = bestObjective.get)) + '\n' + str(objective) +' Score: ' + str(best_valObjective) + '\n' + str(hyperparameters) + '\n\n')
        
                    # order is
                    # val_mae, #of input units, 
                    # input act. func., #of layers, 
                    # #units per layer, act. func. per layer, /
                    # optimizer, learning rate, batch size
                    
                    input_layer_units = contents_read['hyperparameters']['values']['neurons_']
                    input_act_func = contents_read['hyperparameters']['values']['act_']
                    numLayers = contents_read['hyperparameters']['values']['layers']
        
                    
                    #List for storing retrieved metrics
                    metricString =""
                    
                    # This loop will be for grabbing and writing the evaluation metrics
                    for item in args.metrics:
                    
                        value = contents_read['metrics']['metrics'][item]['observations'][0]['value'][0]
        
                        value = "," + str(value)
                    
                        # Adds value to end of list
                        metricString += value
                        
                    
                    #optimizerChosen = contents_read['hyperparameters']['values']['optimizer']
                    #learningRateChosen = contents_read['hyperparameters']['values']['learning_rate_']
                    #batch_size_chosen = contents_read['hyperparameters']['values']['batch_size']
        
                    file2.write(str(rotation) + ',' +  str(r + 1) + ',' + str(min(bestObjective, key = bestObjective.get)) + ',' + str(best_valObjective) + ',' 
                                
                    # Oriignal line for dealing with hours back
                    #file2.write('Cycle: ' + ',' + str(c + 1) + ',' + 'Hours back: ' + ',' + str(h) + ',' + 'Iteration: ' + ',' + str(r + 1) + ',' + 'Trial Number: ' + ',' + str(min(bestMAE, key = bestMAE.get)) + ',' + 'Val_Mae Score: ' + ',' + str(best_valmae) + ',' 
                    + str(input_layer_units) + ','
                    + str(input_act_func) + ','
                    + str(numLayers)
                    + metricString +'\n') 
                    
                    # Removed extraneous things
                    """ + ','
                    + str(optimizerChosen) + ','
                    + str(learningRateChosen) + ','
                    + str(batch_size_chosen)
                    + '\n')"""

    file.close() 
    file2.close()
    print(num_mae_values)

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
