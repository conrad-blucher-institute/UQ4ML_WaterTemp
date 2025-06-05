#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:04:05 2025

@author: Jarett Woodall

NAVIGATE AT YOUR OWN PERIL!
"""

import numpy as np

import pandas as pd

import sys

sys.path.append(r'./src') # need this to import functinos from other files

from evaluation_functions import *

# Variables that are changeable
leadTimes = [12, 48, 96] # You can change this 
cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
architectures = ['PNN', 'mse', 'CRPS']
threshold = 12

path_to_files = ""


# Initialize list to store results
resultsCsvNew = []
resultsCsvCold = []
resultsCsvTemp = []

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
        
                df = pd.read_csv(path_to_files + str(leadTime) + 'h_' + architecture + '_Cycle_' + str(cycle) + '_Combo_' + combo + ".csv")

                #Cold Season
                modDf1 = df.copy()
                modDf1.index = pd.to_datetime(df['date_time'])
                
                # Mask for dates where the month is November (11) through February
                mask = (modDf1.index.month >= 11) | (modDf1.index.month <= 2)
                # Filter the dataframe using the mask
                coldDf = modDf1[mask]
                
                
                #Temperatuer Filter
                modDf2 =  df.copy()
                
                filtered_df = modDf2[modDf2['target'] < threshold]
                
                filtered_df.index = pd.to_datetime(filtered_df['date_time'])
                
                calcDfWhole = pd.DataFrame(index=df.index)
                calcDfCold = pd.DataFrame(index = coldDf.index)
                calcDfTemp = pd.DataFrame(index = filtered_df.index)
                
                # Iterate through prediction columns and compute metrics using averaged predictions
                
                cols_used = [col for col in df.columns if "iteration_" in col and int(col.split('_')[-1]) <= trials]

                if architecture == "PNN":

                    calcDfWhole["avg_pred_" + str(trials)] = df['Mean'] # Compute dynamic average column
                    calcDfWhole["std_pred_" + str(trials)] =  df['Stdev'] # Compute dynamic standard deviation column


                else:
                    # Get the first 'i' prediction columns
                    calcDfWhole["avg_pred_" + str(trials)] = round(df[cols_used].apply(np.mean, axis=1), 2)  # Compute dynamic average column
                    calcDfWhole["std_pred_" + str(trials)] =  round(df[cols_used].apply(np.std, axis=1), 2)  # Compute dynamic standard deviation column
                
                
                #Whole
                actualReshaped = df['target'].values.astype(float).reshape(-1, 1)
                averageReshaped = calcDfWhole["avg_pred_" + str(trials)].values.astype(float)
                stdReshaped = calcDfWhole["std_pred_" + str(trials)].values.astype(float)

                
                actualReshapedTens = df['target'].values.astype(np.float32).reshape(-1, 1)
                averageReshapedTens = calcDfWhole["avg_pred_" + str(trials)].values.astype(np.float32).reshape(-1, 1)
                stdReshapedTens = calcDfWhole["std_pred_" + str(trials)].values.astype(np.float32).reshape(-1, 1)
                

                #New
                pitAverageCalc = get_pit_points(df['target'].values.astype(float), averageReshaped, stdReshaped)
                print("Pit average ran")
                
                #NEWER
                ssrelCalcAVG = get_spread_skill_points(df['target'].values.astype(float), averageReshaped, stdReshaped)

                
                crpsCalc_gauss = crps_gaussian_tf(averageReshapedTens, stdReshapedTens, actualReshapedTens).numpy()
                
                #NEWER
                ssratAverage = ssrat_avg(df['target'].values.astype(float), averageReshaped, stdReshaped)

                
                meCalc = me(actualReshaped, averageReshaped.reshape(-1, 1))
                
                print("ME Calculated")
                
                mae12Calc = mae12(actualReshaped, averageReshaped.reshape(-1, 1))
                print('MAE12 Calculated')
                
                me12Calc = me12(actualReshaped, averageReshaped.reshape(-1, 1))
                
                print("ME12 Calculated")
                maeCalc = mae(actualReshaped, averageReshaped.reshape(-1, 1))
                
                print(actualReshaped.shape)
                print(averageReshaped.reshape(-1, 1).shape)
                rmseCalc = rmse_avg(actualReshaped, averageReshaped.reshape(-1, 1))
                print("RMSE Calculated")

                mseCalc = mse(actualReshaped, averageReshaped.reshape(-1, 1))
                
                row = {
                'architecture': architecture,
                'leadTime': leadTime,
                'cycle': cycle,
                'combination': combo,
                'pit' : pitAverageCalc,
                'ssrel': ssrelCalcAVG,
                'crps_gauss': crpsCalc_gauss,
                'ssrat' : ssratAverage,
                'me': meCalc,
                'mae12': mae12Calc,
                'me12' : me12Calc,
                'mae': maeCalc,
                'rmse': rmseCalc,
                'mse': mseCalc
                }
                
                # Append the row to the results list
                resultsCsvNew.append(row)
                
                ##########COLD SEASON AGGREGATE TABLE######################
                
                #COLD 
                # Convert the index to datetime if it's not already

                if architecture == "PNN":

                    calcDfCold["avg_pred_" + str(trials)] = coldDf['Mean']#round(coldDf[pred_cols].apply(np.mean, axis=1), 2)  # Compute dynamic average column
                    calcDfCold["std_pred_" + str(trials)] =  coldDf['Stdev']#round(coldDf[sigma_cols].apply(np.mean, axis=1), 2)  # Compute dynamic standard deviation column


                else:
                
                    calcDfCold["avg_pred_" + str(trials)] =  round(coldDf[cols_used].apply(np.mean, axis=1), 2)   # Compute dynamic average column
                    calcDfCold["std_pred_" + str(trials)] =  round(coldDf[cols_used].apply(np.std, axis=1), 2) 
                
                coldActual = coldDf['target'].values.astype(float).reshape(-1, 1)
                averageColdReshaped = calcDfCold["avg_pred_" + str(trials)].values.astype(float)
                stdColdReshaped = calcDfCold["std_pred_" + str(trials)].values.astype(float)
                
                actualReshapedTensCold = coldDf['target'].values.astype(np.float32).reshape(-1, 1)
                averageReshapedTensCold = calcDfCold["avg_pred_" + str(trials)].values.astype(np.float32).reshape(-1, 1)
                stdReshapedTensCold = calcDfCold["std_pred_" + str(trials)].values.astype(np.float32).reshape(-1, 1)
                
                # Pass dataframe for CRPS
                overalldfCold = coldDf[cols_used].values.astype(float)
                #New
                pitAverageCalcCold = get_pit_points(coldDf['target'].values.astype(float), averageColdReshaped, stdColdReshaped)
                print("Pit average ran")
                
                #NEWER
                ssrelCalcAVGCold = get_spread_skill_points(coldDf['target'].values.astype(float), averageColdReshaped, stdColdReshaped)
                print("SSREL_ran")
                
                crpsCalcCold_gauss = crps_gaussian_tf(averageReshapedTensCold, stdReshapedTensCold, actualReshapedTensCold).numpy()
                
                #NEWER
                ssratAverageCold = ssrat_avg(coldDf['target'].values.astype(float), averageColdReshaped, stdColdReshaped)
                print("SSRAT Finished")
                
                meCalcCold = me(coldActual, averageColdReshaped.reshape(-1, 1))
                print("ME Calculated")
                
                mae12CalcCold = mae12(coldActual, averageColdReshaped.reshape(-1, 1))
                print('MAE12 Calculated')
                
                me12CalcCold = me12(coldActual, averageColdReshaped.reshape(-1, 1)) 
                
                maeCalcCold = mae(coldActual, averageColdReshaped.reshape(-1, 1)) 
                
                rmseCalcCold = rmse_avg(coldActual, averageColdReshaped.reshape(-1, 1)) 

                mseCalcCold = mse(coldActual, averageColdReshaped.reshape(-1, 1))
                
                
                coldRow = {
                'architecture': architecture,
                'leadTime': leadTime,
                'cycle': cycle,
                'combination': combo,
                'pit' : pitAverageCalcCold,
                'ssrel': ssrelCalcAVGCold,
                'crps_gauss': crpsCalcCold_gauss,
                'ssrat' : ssratAverageCold,
                'me': meCalcCold,
                'mae12': mae12CalcCold,
                'me12' : me12CalcCold,
                'mae': maeCalcCold,
                'rmse': rmseCalcCold,
                'mse': mseCalcCold
                }
                
                resultsCsvCold.append(coldRow)
                print(architecture)
                print(combo)
                print(leadTime)
                

                
                calcDfTemp["avg_pred_" + str(trials)] =  filtered_df['Mean']#round(filtered_df[pred_cols].apply(np.mean, axis=1), 2)   # Compute dynamic average column
                calcDfTemp["std_pred_" + str(trials)] =  filtered_df['Stdev']#round(filtered_df[sigma_cols].apply(np.mean, axis=1), 2)


                tempActual = filtered_df['target'].values.astype(float).reshape(-1, 1)
                averageTempReshaped = calcDfTemp["avg_pred_" + str(trials)].values.astype(float)
                stdTempReshaped = calcDfTemp["std_pred_" + str(trials)].values.astype(float)
                
                actualReshapedTensTemp = filtered_df['target'].values.astype(np.float32).reshape(-1, 1)
                averageReshapedTensTemp = calcDfTemp["avg_pred_" + str(trials)].values.astype(np.float32).reshape(-1, 1)
                stdReshapedTensTemp = calcDfTemp["std_pred_" + str(trials)].values.astype(np.float32).reshape(-1, 1)
                
                # Pass dataframe for CRPS
                overalldfTemp = filtered_df[cols_used].values.astype(float)
                #New
                pitAverageCalcTemp = get_pit_points(filtered_df['target'].values.astype(float), averageTempReshaped, stdTempReshaped)
                print("Pit average ran")
                
                #NEWER
                ssrelCalcAVGTemp = get_spread_skill_points(filtered_df['target'].values.astype(float), averageTempReshaped, stdTempReshaped)
                print("SSREL_ran")
                
                #crpsCalcCold = crps(Actual, overalldfCold)
                print("CRPS RAN")
                
                crpsCalcTemp_gauss = crps_gaussian_tf(averageReshapedTensTemp, stdReshapedTensTemp, actualReshapedTensTemp).numpy()
                
                #NEWER
                ssratAverageTemp = ssrat_avg(filtered_df['target'].values.astype(float), averageTempReshaped, stdTempReshaped)
                print("SSRAT Finished")
                
                meCalcTemp = me(tempActual, averageTempReshaped.reshape(-1, 1))
                print("ME Calculated")
                
                mae12CalcTemp = mae12(tempActual, averageTempReshaped.reshape(-1, 1))
                print('MAE12 Calculated')
                
                me12CalcTemp = me12(tempActual, averageTempReshaped.reshape(-1, 1))
                
                maeCalcTemp = mae(tempActual, averageTempReshaped.reshape(-1, 1))
                
                rmseCalcTemp = rmse_avg(tempActual, averageTempReshaped.reshape(-1, 1))

                mseCalcTemp = mse(tempActual, averageTempReshaped.reshape(-1, 1))
                
                
                tempRow = {
                'architecture': architecture,
                'leadTime': leadTime,
                'cycle': cycle,
                'combination': combo,
                'pit' : pitAverageCalcTemp,
                'ssrel': ssrelCalcAVGTemp,
                'crps_gauss': crpsCalcTemp_gauss,
                'ssrat' : ssratAverageTemp,
                'me': meCalcTemp,
                'mae12': mae12CalcTemp,
                'me12' : me12CalcTemp,
                'mae': maeCalcTemp,
                'rmse': rmseCalcTemp,
                'mse': mseCalcTemp
                }
                
                resultsCsvTemp.append(tempRow)
                print(architecture)
                print(combo)
                print(leadTime)
# Convert the results list into a DataFrame
results_df = pd.DataFrame(resultsCsvNew)

cold_results_df = pd.DataFrame(resultsCsvCold)

temp_results_df = pd.DataFrame(resultsCsvTemp)

# Save the DataFrame to a CSV file
results_df.to_csv(path_to_files + 'whole_metrics_results_performance_byCycleAggregateTable.csv', index=False)
cold_results_df.to_csv(path_to_files + 'cold_metrics_results_performance_byCycleAggregateTable.csv', index=False)
temp_results_df.to_csv(path_to_files + 'temp<' + str(threshold) + '_metrics_results_performance_byCycleAggregateTable.csv', index=False)