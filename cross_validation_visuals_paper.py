#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:37:16 2025

@author: Jarett Woodall
"""

####### IMPORTS ########

import numpy as np


import pandas as pd

import tensorflow as tf

from evaluation_functions import *


######## Code To Run ########
save = True
cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
leadTimes = [12, 48, 96]
architectures = ['mse','PNN', "CRPS"]
iterations = 100
obsVsPred = 'val' # This should be set to either "val" or "testing"


########### Data Retrieval Code ############

"""
function to grab corressponding combination information

# combination - string representation of a specific combo, obsVsPred - string to determine to look at training or val, iterations - integer of the number of times a specific combo was ran, cycle - integer representation of the cycle

return dataframe containing all the predictions
"""
def hyper_combo_parser(MAIN_DIRECTORY, combination, architecture, obsVsPred, iterations, cycle, leadTime):
    
    # Creates empty dataframe
    mainDf = pd.DataFrame()
    
    # Loop for parsing data and grabbing data
    for i in range(iterations):

        if architecture == "PNN":
            df = pd.read_csv(MAIN_DIRECTORY + "/_LT_" +str(leadTime) + "_/results_" + str(architecture) + "_" + str(leadTime)+"_"+combination + "_LT_" + str(leadTime) +"__cycle_" + str(cycle) + "__rep_num_" + str(i) + "_/" + str(obsVsPred) + "_datetime_obsv_predictions.csv")

        else:
            df = pd.read_csv(MAIN_DIRECTORY +"/" +str(leadTime) + "h/"+ str(architecture) + "-" + combination + "-cycle_" + str(cycle) + "-iteration_" + str(i) + "/" + str(obsVsPred) + "_datetime_obsv_predictions.csv")

        if i == 0:
            
            mainDf["target"] = df['target']
            mainDf['date_time'] = df['date_time']
            mainDf.set_index('date_time', inplace=True)
            
        # Sets index
        df.set_index('date_time', inplace=True)
            
        # Drops target and date_time
        df.drop(['target'], axis=1, inplace=True)
        
        #Adds string identifiers to the end
        df = df.add_suffix('_iteration_' + str(i))
        
        # Combine data
        mainDf = pd.concat([mainDf, df], axis=1)
        
        #mainDf['pred_average_iteration_' + i] = round(df.apply(np.mean, axis=1), 2) 
     
    # Drops unnamed columns
    mainDf = mainDf.loc[:, ~mainDf.columns.str.contains('^Unnamed')]
        
    return mainDf
        
# END: def hyper_combo_parser()


########## Driver Code ############


def mme_mse_crps_PNN_lead_times_singlePlot(architectures, iterations, cycles, leadTimes, obsVsPred):
    
    """
        Process the hyperparameter combinations and metrics, and then for each cycle produce one
        boxplot that shows results for all lead times and for all architectures.
        
        Parameters:
          hyper_combos : list
              List of hyperparameter combinations.
          architectures : list
              List of model architectures.
          iterations : int
              Number of iterations to run.
          cycles : list
              List of cycle numbers.
          leadTimes : list
              List of lead times.
          obsVsPred : type
              Flag or data indicating observed versus predicted values.
    
    """
    # Loop over each cycle so that each cycle gets its own combined boxplot.
    for cycle in cycles:
        
        # Loop over each architecture.
        for architecture in architectures:
            
            if architecture == "mse":
                
                MAIN_DIRECTORY = 'VOID'
                
            elif architecture == "CRPS": 
                
                MAIN_DIRECTORY = 'VOID'

            elif architecture == "PNN":
                MAIN_DIRECTORY = 'Void'

            # For every lead time, grab data for each hyperparameter combo.
            for leadTime in leadTimes:
                
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
                
                # Loop over each hyperparameter combo.
                for combo in hyper_combos:
                    # Grab and process the data for this hyperparameter combo, cycle, leadTime, and architecture.
                    print("Before ingestion")

                    df = hyper_combo_parser(MAIN_DIRECTORY, combo, architecture, obsVsPred, iterations, cycle, leadTime)
                    
                    # Remove any duplicate indices.
                    df = df[~df.index.duplicated(keep='first')]
                    
                    # Calculate basic metrics and additional metrics.
                    modDf1 = visualization_metric_calcs(df, architecture)
                    
                    print("Before Metrics")
                    
                    if architecture == 'CRPS':

                        modDf2 = crps_metrics(modDf1)
                        
                    elif architecture == 'PNN':
                        # Runs calc on PNN MME
                        modDf2 = pnn_metrics(modDf1)    
                    else:
                        modDf2 = mse_metrics(modDf1)

                    # Save the processed CSV (you can modify the filename as needed).
                    modDf2.to_csv(f"{leadTime}h_{architecture}_Cycle_{cycle}_Combo_{combo}.csv")

# END: def mme_mse_crps_PNN_lead_times_singlePlot()

"""
This function serves as a driver that will retrieve the created files and 
plot standard deviation plots.

inputs: architectures : list of strings, leadTime : integer, 
cycles: list of integers, save: boolean (True or False)

outputs: plots made using plotly
"""
def decentralized_graphing_driver(architectures, leadTime, cycles, save):

    # Loop over each cycle so that each cycle gets its own combined boxplot.
    for cycle in cycles:
        # Dictionary to accumulate data across all lead times and architectures for the given cycle.
        combosDict_cycle = {}
        
        # Loop over each architecture.
        for architecture in architectures:
                            
            if leadTime == 12 and architecture == "mse":
                hyper_combos = ['3_layers-leaky_relu-32_neurons']#['3_layers-leaky_relu-32_neurons']
                
            elif leadTime == 12 and architecture == "mape":
                hyper_combos = ['3_layers-leaky_relu-8_neurons', '2_layers-leaky_relu-128_neurons']
                
            elif leadTime == 48 and architecture == "mse":
                hyper_combos = ['2_layers-leaky_relu-16_neurons']#['2_layers-leaky_relu-16_neurons']
                
            elif leadTime == 48 and architecture == "mape":
                hyper_combos = ['2_layers-relu-32_neurons', '3_layers-selu-32_neurons']
                
            elif leadTime == 96 and architecture == "mse":
                hyper_combos = ['2_layers-leaky_relu-16_neurons']#['2_layers-leaky_relu-16_neurons']
                
            elif leadTime == 96 and architecture == "mape":
                
                hyper_combos = ['2_layers-relu-8_neurons', '3_layers-leaky_relu-8_neurons']
                
            elif leadTime == 12 and architecture == "CRPS":
                
                hyper_combos = ['3_layers-relu-32_neurons']#['2_layers-leaky_relu-100_neurons', '1_layers-selu-128_neurons', '1_layers-leaky_relu-100_neurons', '1_layers-leaky_relu-128_neurons', '3_layers-relu-32_neurons', '1_layers-relu-128_neurons', '3_layers-leaky_relu-128_neurons', '2_layers-leaky_relu-256_neurons', '3_layers-leaky_relu-100_neurons']
                
            elif leadTime == 48 and architecture == "CRPS":
                hyper_combos = ['3_layers-selu-64_neurons']#['3_layers-selu-64_neurons', '2_layers-relu-128_neurons', '2_layers-relu-100_neurons', '1_layers-selu-128_neurons', '1_layers-selu-100_neurons', '3_layers-leaky_relu-64_neurons', '2_layers-selu-100_neurons', '3_layers-relu-32_neurons']
                
            elif leadTime == 96 and architecture == "CRPS":
                hyper_combos = ['3_layers-relu-100_neurons']#['1_layers-selu-128_neurons', '1_layers-selu-64_neurons', '2_layers-selu-128_neurons', '2_layers-selu-256_neurons', '3_layers-selu-32_neurons', '3_layers-selu-256_neurons', '2_layers-leaky_relu-128_neurons', '3_layers-relu-100_neurons']

            elif leadTime == 12 and architecture == "PNN":

                hyper_combos = ['combo2']    

            elif leadTime == 48 and architecture == "PNN":

                hyper_combos = ['combo1']    

            elif leadTime == 96 and architecture == "PNN":
                hyper_combos = ['combo1']   

            else:
                hyper_combos = []
            # For this leadTime, initialize a temporary dictionary.
            combosDict = {}
            
            # Loop over each hyperparameter combo.
            for combo in hyper_combos:
                
                df = pd.read_csv(f"{leadTime}h_{architecture}_Cycle_{cycle}_Combo_{combo}.csv")

                df['date_time'] = pd.to_datetime(df["date_time"])

                df = df.set_index('date_time')

                # Create a key that encodes the combo, architecture, and leadTime.
                key = f"{architecture}-{leadTime}h"
                combosDict[key] = df
                
            # Update the cycle-level dictionary with the lead time–specific data.
            combosDict_cycle.update(combosDict)

    arch_title = ", ".join(architectures)
    # Call the boxplot function with the aggregated data.
    standardDeviationFan_leadTime_plot(combosDict_cycle, leadTime, arch_title, cycle, save)


########### Graphing Function ############
"""
"""
def standardDeviationFan_leadTime_plot(dfDictBox, leadTime, architecture, cycle, save):
    import plotly.graph_objects as go
    import pathlib
    
    # Creation of Figure 
    fig = go.Figure()

    # Storage for traces for effective layering
    fan_traces = []
    mean_traces = []

    # Sorts the Keys in an Effective Manner
    sorted_keys = sorted(dfDictBox.keys(), key=lambda x: ('mse' not in x, 'CRPS' not in x, 'PNN' not in x))

    # Loops through the dictionary of information for plotting
    for key in sorted_keys:
        
        # Retrieves Dataframe
        df = dfDictBox[key]
        
        # Model Name Changes
        model_name = key.split("-")[0] + "-MME"
        model_name = model_name.upper()
        
        # Color Logic
        if 'CRPS' in key:
            color = "#D8B7DD"
        elif 'PNN' in key:
            color = "#ADD8E6"
        elif 'mse' in key:
            color = "#A8E6A1"
        else:
            color = ""


        if 'CRPS' in key:
            customda = df[['target', 'mae', 'mae<12', 'central_mae', 'central_mae<12', 'crps_gauss']]
            hovertemp = "<br>".join([
                "date_time: %{x}",
                f"Combination: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                "Actual temperature (°C): %{customdata[0]}",
                "Indiv_MAE (°C): %{customdata[1]}",
                "Indiv_MAE<12 (°C): %{customdata[2]}",
                "CRPS (°C): %{customdata[5]}",
                "Central_MAE (°C): %{customdata[3]}",
                "Central_MAE<12 (°C): %{customdata[4]}"
            ])
            
        elif 'PNN' in key:
            customda = df[['target', 'crps', 'central_mae', 'central_mae<12']]
            hovertemp = "<br>".join([
                "date_time: %{x}",
                f"Combination: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                "Actual temperature (°C): %{customdata[0]}",
                "CRPS (°C): %{customdata[1]}",
                "Central_MAE (°C): %{customdata[2]}",
                "Central_MAE<12 (°C): %{customdata[3]}"
            ])
            
        else:
            customda = df[['target', 'rmse', 'mae', 'mae<12', 'central_mae', 'central_mae<12', 'rmse_avg']]
            hovertemp = "<br>".join([
                "date_time: %{x}",
                f"Combination: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                "Actual temperature (°C): %{customdata[0]}",
                "Indiv_RMSE (°C): %{customdata[1]}",
                "Indiv_MAE (°C): %{customdata[2]}",
                "Indiv_MAE<12 (°C): %{customdata[3]}",
                "RMSE_Average_Func (°C): %{customdata[6]}",
                "Central_MAE (°C): %{customdata[4]}",
                "Central_MAE<12 (°C): %{customdata[5]}"
            ])

        # Store fan traces
        fan_traces.append(go.Scatter(
            x=df.index,
            y=df['Below2Std'],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            legendgroup=model_name,
            opacity=1,
            connectgaps=False
        ))
        fan_traces.append(go.Scatter(
            x=df.index,
            y=df['Above2Std'],
            mode='lines',
            line=dict(color=color, width=2),
            fill='tonexty',
            #fillcolor=fill_rgba,
            name=f"{model_name} ±2SD",
            showlegend=True,
            legendgroup=model_name,
            legendrank=1,
            opacity=1,
            connectgaps=False
        ))

        # Store mean traces
        mean_traces.append(go.Scatter(
            x=df.index,
            y=df['Mean'].round(3),
            name=f"{model_name} Mean Prediction",
            customdata=customda,
            hovertemplate=hovertemp,
            line=dict(color=color, width=5),
            line_dash="dot",
            mode='lines',
            showlegend=True,
            legendgroup=model_name,
            legendrank=2,
            opacity=1,
            connectgaps=False
        ))


    # Code to add Traces to the Plot in an order to clearly see
    for trace in fan_traces:
        fig.add_trace(trace)
    for trace in mean_traces:
        fig.add_trace(trace)

    rep_df = next(iter(dfDictBox.values()))
    fig.add_trace(go.Scatter(x=rep_df.index, y=rep_df['target'].round(3), name="Observed Water Temperature", marker=dict(color='black'), mode='lines', showlegend=True,line=dict( width=3)))

    # Threshold Line
    fig.add_hline(y=8, line_dash="dot", line_color="red", annotation_text="Turtle Threshold", annotation_position="top left", annotation_font_size=26, annotation_font_color="red")

    
    # Laebeling 
    title_text = f"Stdev_plot_{leadTime}h_Cycle_{cycle}"
    save_path = f"{architecture}_{leadTime}h_Cycle_{cycle}"

    # Plot adjustme
    fig.update_layout(
        title=title_text,
        font=dict(size=26),
        margin=dict(b=180),
        xaxis_title='DateTime',
        yaxis_title='Temperature (°C)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.62,
            xanchor='center',
            x=0.5,
            font=dict(size=26),
            title=None,
        ),
        xaxis=dict(
            showline=True,
            linecolor='black',
            ticks='outside',
            tickwidth=2,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            showline=True,
            linecolor='black',
            ticks='outside',
            tickwidth=2,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1
        ),
    )

    if save:
        p = pathlib.Path(architecture + "_StdevPlot/")
        p.mkdir(parents=True, exist_ok=True)
        fig.write_html(p / f"{save_path}.html")
    else:
        fig.show()

#END: def standardDeviationFan_leadTime_plot()


############# Hourly Metric Code ##########

"""
# Example Usage:
mu_list = [15.2, 14.8, 15.0, 14.9]  # Means from different PNN models
sigma_list = [0.5, 0.6, 0.55, 0.52]  # Standard deviations from different PNN models

mu_final, sigma_final = combine_gaussian_predictions(mu_list, sigma_list)

print(f"Final Mean (μ): {mu_final:.3f}")
print(f"Final Std Dev (σ): {sigma_final:.3f}")
"""
def combine_gaussian_predictions(mu_list, sigma_list):
    """
    Combines multiple Gaussian predictions from an ensemble of models
    into a single Gaussian approximation.

    Parameters:
    - mu_list: List or numpy array of mean predictions from each model.
    - sigma_list: List or numpy array of standard deviations from each model.

    Returns:
    - mu_final: Combined mean prediction.
    - sigma_final: Combined standard deviation prediction.
    """

    # Convert to numpy arrays if needed
    mu_list = np.array(mu_list)
    sigma_list = np.array(sigma_list)
   
    # Step 1: Compute the mean of the mixture
    mu_final = np.mean(mu_list)
   
    # Step 2: Compute the variance of the mixture
    variance_mean = np.var(mu_list, ddof=0)  # Variance of the means (epistemic uncertainty)
    mean_variance = np.mean(sigma_list**2)   # Average of individual variances (aleatoric uncertainty)
   
    sigma_final = np.sqrt(mean_variance + variance_mean)  # Final standard deviation

    return mu_final, sigma_final

# END: def combine_gaussian_predictions()

"""
Calculates necessary information for plotting box plots, fan plots and spaghetti plots
"""
def visualization_metric_calcs(df, architecture):
    
    # This structure goes into and combines the PNN mean and sigma values
    if architecture == "PNN":

        if 'Mean' not in df.columns:
            # Filters out columns that do not contain pred
            desired_columns = [x for x in df.columns if x.startswith("pred")]
            
            sigma_columns = [x for x in df.columns if x.startswith("sigma")]
            
            muCombined = []
            sigmaCombined = []

            for index, row in df.iterrows():

                predictions = [row[col] for col in desired_columns]  # Convert to list
                sigmas = [row[col] for col in sigma_columns] 

                mu_final, sigma_final = combine_gaussian_predictions(predictions, sigmas)

                muCombined.append(mu_final)
                sigmaCombined.append(sigma_final)

            df['Mean'] = muCombined

            df['Stdev'] = sigmaCombined

        # 2 * standard error
        df['2Stdev'] = round(df['Stdev'] * 2, 2)
        
        # For FanPlot    
        df['BelowStd'] = round(df['Mean'] - df['Stdev'], 2)
        df['AboveStd'] = round(df['Mean'] + df['Stdev'], 2)
        
        df['Below2Std'] = round(df['Mean'] - df['2Stdev'], 2)
        df['Above2Std'] = round(df['Mean'] + df['2Stdev'], 2)

    else:
        # Filters out columns that do not contain pred
        desired_columns = [x for x in df.columns if x.startswith("pred")]

        # Creates new dataframe
        newDf = df[desired_columns]
        
        # Mean Calculation to create hourly mean values for predictions
        df['Mean'] = round(newDf.apply(np.mean, axis=1), 2) 
        
        # Standard Deviation Calculation to create hourly stdev values for the predictions
        df['Stdev'] = round(newDf.apply(np.std, axis=1), 2) 
            
        
        # 2 * standard error
        df['2Stdev'] = round(df['Stdev'] * 2, 2)
        
        # 1 Standard Deviation 
        df['BelowStd'] = round(df['Mean'] - df['Stdev'], 2)
        df['AboveStd'] = round(df['Mean'] + df['Stdev'], 2)
        
        df['Below2Std'] = round(df['Mean'] - df['2Stdev'], 2)
        df['Above2Std'] = round(df['Mean'] + df['2Stdev'], 2)
    
    return df
    
#END: def visualization_metric_calcs   

"""
Custom function for calculating relevant metrics for CRPS
"""
def crps_metrics(df):
    """
    Processes the dataframe `df` by computing CRPS, SSrat, MAE, and MAE<12 for both the ensemble and a singular 'Mean' value.
    Assumes:
      - The ensemble predictions are in columns whose names start with "pred".
      - The true target value is in the column 'target'.
      - The singular prediction is in the column 'Mean'.
    """
    # Filter out columns that start with "pred" to get ensemble predictions.
    desired_columns = [x for x in df.columns if x.startswith("pred")]
    newDf = df[desired_columns]    
    
    # Value Storage Lists
    crpsGauss = []
    maeList = []
    maeSingular = []
    mae12List = []
    mae12Singular = []
    
    for index, row in newDf.iterrows():
        # Retrieve ensemble predictions as a list (e.g., [p1, p2, p3, ...])
        preds = row.tolist()
        # Retrieve the singular prediction (e.g., the ensemble mean already computed elsewhere)
        predSingular = [df.loc[index, 'Mean']]
        
        # Create a list of target values that matches the ensemble length.
        actual = [df.loc[index, 'target']] * len(preds)
        actualSingular = [df.loc[index, 'target']]

        sigma_avg = [df.loc[index, 'Stdev']]
    
        #Convert types
        predsTensor = tf.cast(tf.convert_to_tensor(preds), tf.float64)
        predsTensor = tf.expand_dims(predsTensor, axis=0)  # Now shape: (1, n)
        
        actualTensor = tf.cast(tf.convert_to_tensor(actual), tf.float64)
        actualTensor = tf.expand_dims(actualTensor, axis=0)  # Now shape: (1, n)
        
        # For singular values, create tensors with shape (1, 1).
        predsSingleTensor = tf.cast(tf.convert_to_tensor(predSingular), tf.float32)
        predsSingleTensor = tf.expand_dims(predsSingleTensor, axis=-1)
        
        actualSingleTensor = tf.cast(tf.convert_to_tensor(actualSingular), tf.float32)
        actualSingleTensor = tf.expand_dims(actualSingleTensor, axis=-1)

        # Tensors for singular values
        sigmaAvgSingleTensor = tf.cast(tf.convert_to_tensor(sigma_avg), tf.float32)
        sigmaAvgSingleTensor = tf.expand_dims(sigmaAvgSingleTensor, axis=-1)

        mae_val = mae(actualTensor, predsTensor)
        maeList.append(round(mae_val, 3))
        #print(mae_val)
        mae12_val = mae12(actualTensor, predsTensor)
        mae12List.append(round(mae12_val, 3))

        crps_gaussian = crps_gaussian_tf(predsSingleTensor, sigmaAvgSingleTensor, actualSingleTensor).numpy()
        crpsGauss.append(round(crps_gaussian, 3))

        # Singular metrics.
        maeSingular_val = mae(actualSingleTensor, predsSingleTensor)
        maeSingular.append(round(maeSingular_val, 3))
        
        mae12Singular_val = mae12(actualSingleTensor, predsSingleTensor)
        mae12Singular.append(round(mae12Singular_val, 3))
        
    # Append computed metrics as new columns to the dataframe.
    df['crps_gauss'] = crpsGauss
    df['mae'] = maeList
    df['mae<12'] = mae12List
    df['central_mae'] = maeSingular
    df['central_mae<12'] = mae12Singular
        
    return df

#END: def crps_metrics()

"""
Custom function for calculating relevant metrics for MME MAPE and MSE
"""    
def mse_metrics(df):
    
    # Filters out columns that do not contain pred
    desired_columns = [x for x in df.columns if x.startswith("pred")]
    
    # Creates new dataframe
    newDf = df[desired_columns]

    # Value Storage Lists
    maeList = []
    mae12List = []
    rmseList = []
    maeSingular = []
    mae12Singular = []
    rmseSingular = []
    
    for index, row in newDf.iterrows():
        
        # Retrieves values
        preds = row.tolist()
        predSingular = [df.loc[index, 'Mean']]
        
        # Finds target value and creates a list of the same length
        actual = [df.loc[index, 'target']] * len(preds)
        actualSingular = [df.loc[index, 'target']]
    
        # Tesnors for multiple values
        predsTensor = tf.cast(tf.convert_to_tensor(preds), tf.float64)
        predsTensor = tf.expand_dims(predsTensor, axis=-1)

        actualTensor = tf.cast(tf.convert_to_tensor(actual), tf.float64)
        actualTensor = tf.expand_dims(actualTensor, axis=-1)

        # Tensors for singular values
        predsSingleTensor = tf.cast(tf.convert_to_tensor(predSingular), tf.float32)
        predsSingleTensor =  tf.expand_dims(predsSingleTensor, axis=-1)

        actualSingleTensor = tf.cast(tf.convert_to_tensor(actualSingular), tf.float32)
        actualSingleTensor =  tf.expand_dims(actualSingleTensor, axis=-1)

        #Grabs values and appends them to a list to be placed into dataframe
        rmseList.append(round(rmse(actualTensor, predsTensor), 3))
        maeList.append(round(mae(actualTensor, predsTensor), 3))
        mae12List.append(round(mae12(actualTensor, predsTensor), 3))
        
        rmseSingular.append(round(rmse_avg(actualSingleTensor, predsSingleTensor), 3))
        maeSingular.append(round(mae(actualSingleTensor, predsSingleTensor), 3))
        mae12Singular.append(round(mae12(actualSingleTensor, predsSingleTensor), 3))
    
    df['rmse'] = rmseList
    df['mae'] = maeList
    df['mae<12'] = mae12List
    
    df['rmse_avg'] = rmseSingular
    df['central_mae'] = maeSingular
    df['central_mae<12'] = mae12Singular
    
    return df

#END: def mme_metrics()

"""
Custom function for caluclating relevant metrics for PNN
"""
def pnn_metrics(df):
    
    crpsList = []
    ssratList_avg = []
    maeSingular = []
    mae12Singular = []
    
    for index, row in df.iterrows():
        
        # Retrieves Mean Values (The predictions have already been combined)
        predSingular = [df.loc[index, 'Mean']]
        
        # Finds target value and creates a list of the same length
        actualSingular = [df.loc[index, 'target']]
        
        sigma_avg = [df.loc[index, 'Stdev']]

        # Tensors for singular values
        predsSingleTensor = tf.cast(tf.convert_to_tensor(predSingular), tf.float32)
        predsSingleTensor =  tf.expand_dims(predsSingleTensor, axis=-1)
        actualSingleTensor = tf.cast(tf.convert_to_tensor(actualSingular), tf.float32)
        actualSingleTensor =  tf.expand_dims(actualSingleTensor, axis=-1)
        
        # Tensors for singular values
        sigmaAvgSingleTensor = tf.cast(tf.convert_to_tensor(sigma_avg), tf.float32)
        sigmaAvgSingleTensor = tf.expand_dims(sigmaAvgSingleTensor, axis=-1)

        # Forcibly converts crps value into a numpy float
        crps_value = crps_gaussian_tf(predsSingleTensor, sigmaAvgSingleTensor, actualSingleTensor)
        if isinstance(crps_value, tf.Tensor):
            crps_value = crps_value.numpy().item()
        
        #Grabs values and appends them to a list to be placed into dataframe
        crpsList.append(round(crps_value, 3))

        #Grabs values and appends them to a list to be placed into dataframe
        maeSingular.append(round(mae(actualSingleTensor, predsSingleTensor), 3))
        mae12Singular.append(round(mae12(actualSingleTensor, predsSingleTensor), 3))
                
    df['crps'] = crpsList
    df['central_mae'] = maeSingular
    df['central_mae<12'] = mae12Singular
        
    return df

# END: def pnn_metrics()

if __name__ == "__main__":
    
    ##### TO Run the Code#######
    # This line is ran so that the files needed to create the pltos are created and retrieved
    mme_mse_crps_PNN_lead_times_singlePlot(architectures, iterations, cycles, leadTimes, obsVsPred)
    
    # For loop to loop through leadtimes and create plots for each cycle (rotation)
    for leadTime in leadTimes:
        
        # This Line Will need to be ran to plot the graphs
        decentralized_graphing_driver(architectures, leadTime, cycles)
        
else:
    
    print ("Cross_validation_visuals.py is running")
