#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:04:05 2025

@author: Jarett Woodall

This file process and evaluates model performance and places them into Aggregated Tables.
This file creates byCycle tables and byUQMethod tables, 
for the byUQMethod tables, this file handles data processing as well as evaluation.
"""

####### Imports #######
import numpy as np

import pandas as pd

from pathlib import Path

from evaluations.evaluation_functions import mae12, mae, rmse_avg, crps_gaussian_tf, get_pit_points, get_spread_skill_points, mse, me, me12, ssrat_avg

from evaluations.cross_validation_visuals_paper import model_selection_conditional

######## Table Code and Data Retrieval Function ########
def aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred, URI=False, padding = 24):
    
    """
    This function creates byCycle and byUQMethod tables displaying 
    performance and evaludation metrics.
    
    Inputs:
        
        leadTimes: list of ints,
        cycles: list of ints,
        architectures: list of strings,
        threshold: int or float representing temperature
        byCycle: boolean to swap between byCycle and byUQMethod,
        obsVsPred: string denoting whether or not val, testing, 2021, or 2024 data is being used,
        URI: boolean to run additional calculations if 2021 independent year is present
        padding: int (set to 24, user can determine how many hours before and after Winter Storm URI the user can include in calculations)
        
    Output: 
        Aggregate tables in csv format. 
    """

    # Initialize list to store results
    resultsCsvWhole = []
    resultsCsvCold = []
    resultsCsvTemp = []
    resultsCsvURI = []
    resultsCsvURIPadded = []
    
    # Conditional to run for byCombo
    if byCycle == False:
        
        #Change to byUQMethod
        # Runs helper function to combine files for byCombo metrics (wraps all cycles into one file for a model)
        pre_aggregate_byUQMethod_file(leadTimes, cycles, architectures, obsVsPred)
        
        # This line is here so that the code executes only once, instead of through all the cycles
        cycles = [0]
        
    for cycle in cycles:
    
        for leadTime in leadTimes:
            
            for architecture in architectures:
            
                model_list = model_selection_conditional(leadTime, architecture)
                        
                for model in model_list:
                    
                    # Initializes dictionary for storage of 3 dataframes
                    dfDict = {}
            
                    # IF structure to retrieve relevant file
                    if byCycle == True:
                        
                        # Path creation
                        input_path = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "UQ_Files"/ f"{obsVsPred}_{leadTime}h_{architecture}_Cycle_{cycle}_Model_{model}.csv"
                        df = pd.read_csv(input_path)
                        
                        saveIdentifier = "byCycle"
                        
                    else:
                        # Path creation
                        uq_folder = "_".join(architectures) + "_byUQMethod_UQ_Files"
                        input_path = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / uq_folder / f"{obsVsPred}_{leadTime}h_{architecture}_Model_{model}.csv"
                        
                        df = pd.read_csv(input_path)

                        saveIdentifier = "byUQMethod"
                        
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
                    
                    if URI == True:
                        
                        # Define start and end for Winter Storm Uri
                        start = pd.Timestamp("2021-02-14 05:00")
                        end = pd.Timestamp("2021-02-21 15:00")
                        
                        # Mask for Winter Storm URI
                        mask = (modDf1.index >= start) & (modDf1.index <= end)
                        
                        # Filter the dataframe using the mask
                        uriDf = modDf1[mask]
                        
                        # Adds URI dataframe to dictionary
                        dfDict['URI'] = uriDf
                        
                        # Apply padding for hours before and after URI
                        start_padded = start - pd.Timedelta(hours=padding)
                        end_padded = end + pd.Timedelta(hours=padding)
                        
                        # Filter using padded time range
                        mask = (modDf1.index >= start_padded) & (modDf1.index <= end_padded)
                        uriDfpadded = modDf1[mask]
                        
                        # Add to dictionary
                        dfDict['URI' + str(padding)] = uriDfpadded
                                                
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
                        'dataset': obsVsPred,
                        'selection': key,
                        'leadTime': leadTime,
                        'cycle': cycle,
                        'model': model,
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
                            
                        elif key == 'URI':
                            
                            resultsCsvURI.append(row)
                            
                        elif key == 'URI' + str(padding):
                            
                            resultsCsvURIPadded.append(row)
                            
                    # Line of code to help ensure memory is being freed up
                    del df, coldDf, filtered_df, dfDict

    # Convert the results list into a DataFrame
    whole_results_df = pd.DataFrame(resultsCsvWhole)
    cold_results_df = pd.DataFrame(resultsCsvCold)
    temp_results_df = pd.DataFrame(resultsCsvTemp)

    # This section of code takes care of creating a folder called AggregateTables that will store all of the csvs
    # Define the folder path
    aggregate_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "Aggregate_Tables"

    # Create it if it doesn't exist
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    
    if URI == True:
        
        # Places dictionaries into dataframes
        results_URI_df = pd.DataFrame(resultsCsvURI)
        results_URI_padded_df = pd.DataFrame(resultsCsvURIPadded)
        
        # Places into csvs
        results_URI_df.to_csv(aggregate_dir /f'{obsVsPred}_URI_metrics_results_performance_{saveIdentifier}AggregateTable.csv')
        results_URI_padded_df.to_csv(aggregate_dir /f'{obsVsPred}_URI_padded_metrics_results_performance_{saveIdentifier}AggregateTable.csv')
    
    
    # Save the DataFrame to a CSV file
    whole_results_df.to_csv(aggregate_dir /f'{obsVsPred}_whole_metrics_results_performance_{saveIdentifier}AggregateTable.csv', index=False)
    cold_results_df.to_csv(aggregate_dir /f'{obsVsPred}_cold_metrics_results_performance_{saveIdentifier}AggregateTable.csv', index=False)
    temp_results_df.to_csv(aggregate_dir /f'{obsVsPred}_temp<{threshold}_metrics_results_performance_{saveIdentifier}AggregateTable.csv', index=False)
    
# END: def aggregateTable()

def pre_aggregate_byUQMethod_file(leadTimes, cycles, architectures, obsVsPred):
    
    """
    Function to combine files so that they are no longer separated by Cycle.
    """
    for leadTime in leadTimes:
        
        for architecture in architectures:
            
            # Helper function for model selection
            model_list = model_selection_conditional(leadTime, architecture)
    
            for model in model_list:
                
                dataframeList = []
                
                for cycle in cycles:
                    
                    # Path to ensure compatability
                    input_path = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "UQ_Files"/ f"{obsVsPred}_{leadTime}h_{architecture}_Cycle_{cycle}_Model_{model}.csv"
                    df = pd.read_csv(input_path)

                    # Appends dataframe to list for concatenation
                    dataframeList.append(df)

                    print("Appeneded")

                # Concats all of the dataframes together vertically
                mainDf = pd.concat(dataframeList, axis=0, ignore_index=True)

                # Define the subfolder name based on the architectures list
                uq_folder_name = "_".join(architectures) + "_byUQMethod_UQ_Files"

                # Define the full path under UQ_Visuals_Tables_Files
                output_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / uq_folder_name

                # Create the directory (and any parents) if it doesn't exist
                output_dir.mkdir(parents=True, exist_ok=True)
                
                mainDf.to_csv(output_dir/ f"{obsVsPred}_{leadTime}h_{architecture}_Model_{model}.csv")
                
                print("Converted to CSV")
                
##END: def pre_aggregate_byUQMethod_file()

######## DEBUGGING CODE ###########
if __name__ == "__main__":
    
    # Variables that are changeable
    leadTimes = [12, 48, 96] # You can change this 
    cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    architectures = ['PNN', 'mse', 'CRPS']
    threshold = 12 # Changeable
    byCycle = True # Set to false if you wish to run byCombo code
    obsVsPred = "2021"
    
    # Runs the Function helper
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)
    
    # Run this line if you are working with data using 2021, if you dont the code will break and not run correctly.
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred, True ,24)
    
else:
    
    print("AggregateTables is running")