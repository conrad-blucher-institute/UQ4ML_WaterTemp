#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:04:05 2025

@author: Jarett Woodall

"""

import numpy as np

import pandas as pd

from pathlib import Path

from evaluation_functions import *

def aggregateTable(leadTimes, cycles, architectures, threshold, byCycle):

    # Initialize list to store results
    resultsCsvWhole = []
    resultsCsvCold = []
    resultsCsvTemp = []
    
    # Conditional to run for byCombo
    if byCycle == False:
        
        # Runs helper function to combine files for byCombo metrics
        pre_aggreagete_combo_file(leadTimes, architectures)
        
        # This line is here so that the code executes only once, instead of through all the cycles
        cycles = [0]
        
    for cycle in cycles:
    
        for leadTime in leadTimes:
            
            for architecture in architectures:
            
                if leadTime == 12 and architecture == "mse":
                    
                    hyper_combos = ['3_layers-leaky_relu-32_neurons']
                    
                    
                elif leadTime == 48 and architecture == "mse":
                    
                    hyper_combos = ['2_layers-leaky_relu-16_neurons']
                    
                elif leadTime == 96 and architecture == "mse":
                    
                    hyper_combos = ['2_layers-leaky_relu-16_neurons']
                    
                elif leadTime == 12 and architecture == "CRPS":
                    
                    hyper_combos = ['3_layers-relu-32_neurons']
                    
                elif leadTime == 48 and architecture == "CRPS":
                    
                    hyper_combos = ['3_layers-selu-64_neurons']
                    
                elif leadTime == 96 and architecture == "CRPS":
                    
                    hyper_combos = ['3_layers-relu-100_neurons']
    
                elif leadTime == 12 and architecture == "PNN":
    
                    hyper_combos = ['combo2']    
    
                elif leadTime == 48 and architecture == "PNN":
    
                    hyper_combos = ['combo1']    
    
                elif leadTime == 96 and architecture == "PNN":
                    
                    hyper_combos = ['combo1']   
    
                else:
                    hyper_combos = []
                        
                for combo in hyper_combos:
                    
                    # Initializes dictionary for storage of 3 dataframes
                    dfDict = {}
            
                    # IF structure to retrieve relevant code
                    if byCycle == True:
                        
                        df = pd.read_csv(str(leadTime) + 'h_' + architecture + '_Cycle_' + str(cycle) + '_Combo_' + combo + ".csv")
                        
                        saveIdentifier = "byCycle"
                        
                    else:
                        
                        df = pd.read_csv("_".join(architectures) + "_Combo_UQ_Files/" + str(leadTime) + 'h_' + architecture + '_Combo_' + combo + ".csv")
                        
                        saveIdentifier = "byCombo"
                        
                    # Adds whole Df to the dictionary 
                    dfDict['Whole'] = df
    
                    ################ Data Filtering ###################
                    #Cold Season
                    modDf1 = df.copy()
                    modDf1.index = pd.to_datetime(df['date_time'])
                    
                    # Mask for dates where the month is November (11) through February
                    mask = (modDf1.index.month >= 11) | (modDf1.index.month <= 2)
                    # Filter the dataframe using the mask
                    coldDf = modDf1[mask]
                    
                    # Adds Cold dataframe to dictionary
                    dfDict['COLD'] = coldDf
                    
                    #Temperature Filter
                    modDf2 =  df.copy()
                    
                    filtered_df = modDf2[modDf2['target'] < threshold]
                    
                    filtered_df.index = pd.to_datetime(filtered_df['date_time'])
                    
                    # Adds filtered temp dataframe to dictionary
                    dfDict['Temp<'+str(threshold)] = filtered_df
                    
                    #######################################################
                    
                    for key in dfDict:
                        
                        modDf = dfDict[key]
                        
                        # Print statements to give the user an idea of where the calcilations are
                        print(cycle)
                        print(leadTime)
                        print(architecture)
                        print(key)
                    
                        # Data Preparation for Evaluation
                        # Numpy
                        actualShaped = modDf['target'].values.astype(float)
                        averageReshaped = modDf["Mean"].values.astype(float)
                        stdReshaped = modDf["Stdev"].values.astype(float)
                        
                        # Reshaped to Tensors
                        actualReshaped = modDf['target'].values.astype(float).reshape(-1, 1)
                        averageShapedTens =  averageReshaped.reshape(-1, 1)
    
                        # Reshapes for CRPS Calculations
                        actualReshapedTens = modDf['target'].values.astype(np.float32).reshape(-1, 1)
                        averageReshapedTens = modDf["Mean"].values.astype(np.float32).reshape(-1, 1)
                        stdReshapedTens = modDf["Stdev"].values.astype(np.float32).reshape(-1, 1)
                        
                        # Evaluation Calculations
                        pitAverageCalc = get_pit_points(actualShaped, averageReshaped, stdReshaped)
                        print("Pit Calculated")
                        
                        ssrelCalcAVG = get_spread_skill_points(actualShaped, averageReshaped, stdReshaped)
                        print("SSREL Calculated")
                        
                        crpsCalc_gauss = crps_gaussian_tf(averageReshapedTens, stdReshapedTens, actualReshapedTens).numpy()
                        print("CRPS Calculated")
                        
                        ssratAverage = ssrat_avg(actualShaped, averageReshaped, stdReshaped)
                        print("SSRAT Calculated")
                        
                        meCalc = me(actualReshaped, averageShapedTens)
                        print("ME Calculated")
                        
                        mae12Calc = mae12(actualReshaped, averageShapedTens)
                        print('MAE12 Calculated')
                        
                        me12Calc = me12(actualReshaped, averageShapedTens)
                        print("ME12 Calculated")
                        
                        maeCalc = mae(actualReshaped, averageShapedTens)
                        print('MAE calculated')
        
                        rmseCalc = rmse_avg(actualReshaped, averageShapedTens)
                        print("RMSE Calculated")
        
                        mseCalc = mse(actualReshaped, averageShapedTens)
                        print("MSE Calculated")
                        
                        row = {
                        'architecture': architecture,
                        'dataset': key,
                        'leadTime': leadTime,
                        'cycle': cycle,
                        'combination': combo,
                        'pit' : pitAverageCalc,
                        'ssrel': ssrelCalcAVG,
                        'crps': crpsCalc_gauss,
                        'ssrat' : ssratAverage,
                        'me': meCalc,
                        'mae12': mae12Calc,
                        'me12' : me12Calc,
                        'mae': maeCalc,
                        'rmse': rmseCalc,
                        'mse': mseCalc
                        }
                        
                        # Structure to append the rows to their corresponding containers
                        if key == "Whole":
                                                    
                            resultsCsvWhole.append(row)
                            
                        elif key == "COLD":
                            
                            resultsCsvCold.append(row)
                    
                        elif key == 'Temp<'+str(threshold):
                            
                            resultsCsvTemp.append(row)
                            
                    # Line of code to help ensure memory is being freed up
                    del df, coldDf, filtered_df, dfDict

    # Convert the results list into a DataFrame
    whole_results_df = pd.DataFrame(resultsCsvWhole)
    cold_results_df = pd.DataFrame(resultsCsvCold)
    temp_results_df = pd.DataFrame(resultsCsvTemp)
    
    
    # Save the DataFrame to a CSV file
    whole_results_df.to_csv('whole_metrics_results_performance_' + saveIdentifier + 'AggregateTable.csv', index=False)
    cold_results_df.to_csv('cold_metrics_results_performance_' + saveIdentifier + 'AggregateTable.csv', index=False)
    temp_results_df.to_csv('temp<' + str(threshold) + '_metrics_results_performance_' + saveIdentifier + 'AggregateTable.csv', index=False)
    
# END: def aggregateTable()


def pre_aggreagete_combo_file(leadTimes, architectures):
    
    """
    Function to combine files so that they are no longer separated by Cycle.
    """
    
    # Use the current working directory as the base path
    folder_path = Path.cwd() / ("_".join(architectures) + "_Combo_UQ_Files")
    
    # Create the folder if it does not exist
    folder_path.mkdir(parents=True, exist_ok=True)
    
    for leadTime in leadTimes:
        
        for architecture in architectures:
    
            if leadTime == 12 and architecture == "mse":
                
                hyper_combos = ['3_layers-leaky_relu-32_neurons']
                
                
            elif leadTime == 48 and architecture == "mse":
                
                hyper_combos = ['2_layers-leaky_relu-16_neurons']
                
            elif leadTime == 96 and architecture == "mse":
                
                hyper_combos = ['2_layers-leaky_relu-16_neurons']
                
            elif leadTime == 12 and architecture == "CRPS":
                
                hyper_combos = ['3_layers-relu-32_neurons']
                
            elif leadTime == 48 and architecture == "CRPS":
                
                hyper_combos = ['3_layers-selu-64_neurons']
                
            elif leadTime == 96 and architecture == "CRPS":
                
                hyper_combos = ['3_layers-relu-100_neurons']

            elif leadTime == 12 and architecture == "PNN":

                hyper_combos = ['combo2']    

            elif leadTime == 48 and architecture == "PNN":

                hyper_combos = ['combo1']    

            elif leadTime == 96 and architecture == "PNN":
                
                hyper_combos = ['combo1']   

            else:
                hyper_combos = []
    
            for combo in hyper_combos:
                
                dataframeList = []
                
                for cycle in cycles:
                    
                    # Retrieves Files
                    file_name = f"{leadTime}h_{architecture}_Cycle_{cycle}_Combo_{combo}.csv"
            
                    # Read the CSV file
                    df = pd.read_csv(file_name)
                    dataframeList.append(df)

                    print("Appeneded")

                mainDf = pd.concat(dataframeList, axis=0, ignore_index=True)
                
                
                mainDf.to_csv(folder_path / f"{leadTime}h_{architecture}_Combo_{combo}.csv")
                
                print("Converted to CSV")
##END: def pre_aggreagte_combo_file()

if __name__ == "__main__":
    
    # Variables that are changeable
    leadTimes = [12, 48, 96] # You can change this 
    cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    architectures = ['PNN', 'mse', 'CRPS']
    threshold = 12 # Changeable
    byCycle = True # Set to false if you wish to run byCombo code
    
    # Runs the Function helper
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle)
    
else:
    
    print("AggregateTables is running")