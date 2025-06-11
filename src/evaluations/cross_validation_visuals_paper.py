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

import plotly.graph_objects as go

from pathlib import Path

from evaluations.evaluation_functions import mae12, mae, rmse_avg, crps_gaussian_tf

########### Data Retrieval Code ############

def model_parser(MAIN_DIRECTORY, combination, architecture, obsVsPred, iterations, cycle, leadTime):
    """
    function to grab corressponding combination information

    # combination - string representation of a specific combo, obsVsPred - string to determine to look at training or val, iterations - integer of the number of times a specific combo was ran, cycle - integer representation of the cycle

    return dataframe containing all the predictions
    """ 
    # Creates empty dataframe
    mainDf = pd.DataFrame()
    
    # Loop for parsing data and grabbing data
    for i in range(iterations):
        
        # Increment to start at 1
        i+=1
        
        # Reads dataframes in regardless of macOS or Windows
        file_path = Path("UQ4ML_WaterTemp") / "src" / MAIN_DIRECTORY / f"{leadTime}h" / f"{architecture}-{combination}-cycle_{cycle}-iteration_{i}" / f"{obsVsPred}_datetime_obsv_predictions.csv"
        df = pd.read_csv(file_path)
        if i == 1:
            
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
     
    # Drops unnamed columns
    mainDf = mainDf.loc[:, ~mainDf.columns.str.contains('^Unnamed')]
        
    return mainDf
        
# END: def model_parser()

######### Helper Function Code ########

def model_selection_conditional(leadTime, architecture):
    
    """
    Helper function for grabbing the model names for file reference.
    
    Inputs:
        
        leadTime: integer
        architecture: string
        
    returns:
        
        list of models
    """
    
    if leadTime == 12 and architecture == "mse":
        
        model_names = ['3_layers-leaky_relu-32_neurons']
        
    elif leadTime == 48 and architecture == "mse":
        
        model_names = ['2_layers-leaky_relu-16_neurons']

    elif leadTime == 96 and architecture == "mse":
        
        model_names = ['2_layers-leaky_relu-16_neurons']
        
    elif leadTime == 12 and architecture == "CRPS":
        
        model_names = ['3_layers-relu-32_neurons']
        
    elif leadTime == 48 and architecture == "CRPS":
        
        model_names = ['3_layers-selu-64_neurons']
        
    elif leadTime == 96 and architecture == "CRPS":
        
        model_names = ['3_layers-relu-100_neurons']
    
    elif leadTime == 12 and architecture == "PNN":

        model_names = ['combo2']  

    elif leadTime == 48 and architecture == "PNN":
        
        model_names = ['combo1']  

    elif leadTime == 96 and architecture == "PNN":
        
        model_names = ['combo1']  
        
    else:
        model_names = []
        
    return model_names

# END: def model_selection_conditional()

########## Driver Code ############

def mme_mse_crps_PNN_lead_times_singlePlot(architectures, iterations, cycles, leadTimes, obsVsPred, expanded):
    
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
              
          expanded: boolean
              Flag to indicate if the user wishes to have all prediction data
              or if they wish to only have the summary metrics. 
    
    """
    # Loop over each cycle so that each cycle gets its own combined boxplot.
    for cycle in cycles:
        
        # Loop over each architecture.
        for architecture in architectures:
            
            # Grabs the corresponding directory where training information was stored
            MAIN_DIRECTORY = 'results/' + str(architecture.lower()) + "_results"

            # For every lead time, grab data for each hyperparameter combo.
            for leadTime in leadTimes:
                
                # Calls helper function for conditional tree for models
                model_list = model_selection_conditional(leadTime, architecture)
                
                # Loop over each hyperparameter combo.
                for model in model_list:
                    # Grab and process the data for this model, cycle, leadTime, and architecture.
                    print("Before ingestion")

                    df = model_parser(MAIN_DIRECTORY, model, architecture, obsVsPred, iterations, cycle, leadTime)
                    
                    # Remove any duplicate indices. # Unnecessary
                    df = df[~df.index.duplicated(keep='first')]
                    
                    # Calculate basic metrics and additional metrics.
                    modDf1 = visualization_metric_calcs(df, architecture, expanded)
                    
                    print("Before Metrics")
                    
                    if architecture == 'CRPS':

                        modDf2 = crps_metrics(modDf1)
                        
                    elif architecture == 'PNN':
                        
                        # Runs calc on PNN MME
                        modDf2 = pnn_metrics(modDf1)  
                        
                    elif architecture == "mse":
                        modDf2 = mse_metrics(modDf1)

                    # To ensure cross compatability
                    base_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "UQ_Files"

                    # Create the directories if they do not exist
                    base_dir.mkdir(parents=True, exist_ok=True)

                    # Now define the output path
                    output_path = base_dir / f"{obsVsPred}_{leadTime}h_{architecture}_Cycle_{cycle}_Model_{model}.csv"
                    modDf2.to_csv(output_path)

# END: def mme_mse_crps_PNN_lead_times_singlePlot()

def decentralized_graphing_driver(architectures, leadTime, cycles, obsVsPred, save):
    """
    This function serves as a driver that will retrieve the created files and 
    plot standard deviation plots.

    inputs: architectures : list of strings, leadTime : integer, 
    cycles: list of integers, obsVsPred: string to differentiate between data, 
    save: boolean (True or False)

    outputs: plots made using plotly
    """
    # Loop over each cycle so that each cycle gets its own combined boxplot.
    for cycle in cycles:
        
        # Dictionary to accumulate data across all lead times and architectures for the given cycle.
        modelsDict_cycle = {}
        
        # Loop over each architecture.
        for architecture in architectures:
                            
            model_list = model_selection_conditional(leadTime, architecture)
                
            # For this leadTime, initialize a temporary dictionary.
            modelsDict = {}
            
            # Loop over each hyperparameter combo.
            for model in model_list:

                # Utilizes Path for cross compatability regardless of macOs or Windows
                input_path = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "UQ_Files"/ f"{obsVsPred}_{leadTime}h_{architecture}_Cycle_{cycle}_Model_{model}.csv"
                df = pd.read_csv(input_path)

                df['date_time'] = pd.to_datetime(df["date_time"])

                df = df.set_index('date_time')

                # Create a key that encodes the combo, architecture, and leadTime.
                key = f"{architecture}-{leadTime}h"
                modelsDict[key] = df
                
            # Update the cycle-level dictionary with the lead time–specific data.
            modelsDict_cycle.update(modelsDict)
    
    # Combines architectures to make an effective title
    if len(architectures) > 1:
        
        arch_title = ", ".join(architectures)
        
    else:
        
        arch_title = architectures[0]
    
    # Call the boxplot function with the aggregated data.
    standardDeviationFan_leadTime_plot(modelsDict_cycle, leadTime, arch_title, cycle, obsVsPred, save)
    
# END: def decentralized_graphing_driver()

########### Graphing Function ############

def standardDeviationFan_leadTime_plot(dfDict, leadTime, arch_title, cycle, obsVsPred, save):
    
    """
    This function serves as the plotting function for a standard deviation fan.
    
    Inputs:
        dfDict: dictionary holding dataframes for plotting,
        leadTime: integer representing leadtime,
        arch_title: string for the save file naming convention,
        obsVsPred: string to denote data being used,
        cycle: integer denoting cycle being plotted,
        save: boolean to save or display
    Output:
        
        A fan plot
    """
    
    # Creation of Figure 
    fig = go.Figure()

    # Storage for traces for effective layering
    fan_traces = []
    mean_traces = []

    # Sorts the Keys in an Effective Manner
    sorted_keys = sorted(dfDict.keys(), key=lambda x: ('mse' not in x, 'CRPS' not in x, 'PNN' not in x))

    # Loops through the dictionary of information for plotting
    for key in sorted_keys:
        
        # Retrieves Dataframe
        df = dfDict[key]
        
        # Model Name Changes
        model_name = key.split("-")[0] + "-MME"
        model_name = model_name.upper()
        
        # Hover Text Templates and Colors
        if 'CRPS' in key:
            color = "#D8B7DD"
            customda = df[['target', 'central_mae', 'central_mae<12', 'crps_gauss']]
            hovertemp = "<br>".join([
                "date_time: %{x}",
                f"Model: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                "Actual temperature (°C): %{customdata[0]}",
                "CRPS (°C): %{customdata[3]}",
                "Central_MAE (°C): %{customdata[1]}",
                "Central_MAE<12 (°C): %{customdata[2]}"
            ])
            
        elif 'PNN' in key:
            color = "#ADD8E6"
            customda = df[['target', 'crps', 'central_mae', 'central_mae<12']]
            hovertemp = "<br>".join([
                "date_time: %{x}",
                f"Model: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                "Actual temperature (°C): %{customdata[0]}",
                "CRPS (°C): %{customdata[1]}",
                "Central_MAE (°C): %{customdata[2]}",
                "Central_MAE<12 (°C): %{customdata[3]}"
            ])
            
        elif 'mse' in key:
            color = "#A8E6A1"
            customda = df[['target','central_mae', 'central_mae<12', 'rmse_avg']]
            hovertemp = "<br>".join([
                "date_time: %{x}",
                f"Model: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                "Actual temperature (°C): %{customdata[0]}",
                "RMSE_Average_Func (°C): %{customdata[3]}",
                "Central_MAE (°C): %{customdata[1]}",
                "Central_MAE<12 (°C): %{customdata[2]}"
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

    rep_df = next(iter(dfDict.values()))
    fig.add_trace(go.Scatter(x=rep_df.index, y=rep_df['target'].round(3), name="Observed Water Temperature", marker=dict(color='black'), mode='lines', showlegend=True,line=dict( width=3)))

    # Threshold Line
    fig.add_hline(y=8, line_dash="dot", line_color="red", annotation_text="Turtle Threshold", annotation_position="top left", annotation_font_size=26, annotation_font_color="red")

    # Laebeling 
    title_text = f"Stdev_plot_{leadTime}h_Cycle_{cycle}"
    save_path = f"{obsVsPred}_{arch_title}_{leadTime}h_Cycle_{cycle}"

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
        # Define output directory path
        output_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / f"{arch_title}_StdevPlot"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the figure to HTML
        fig.write_html(output_dir / f"{save_path}.html")
    else:
        fig.show()

#END: def standardDeviationFan_leadTime_plot()

############# Hourly Metric Code ##########

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

def visualization_metric_calcs(df, architecture, expanded):
    """
    Calculates necessary information for plotting box plots, fan plots and spaghetti plots
    
    Inputs:
        df: dataframe,
        architecture: string denoting architecture,
        expanded: boolean to determine if all predictions should be saved
        
    Output:
        returns a dataframe with calculations
    """
    
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
    
    # This conditional will determine if the user wishes to keep all prediction
    # Or if they wish to only use the summary metrics for plotting and calculations
    if expanded == True:
        
        
        return df
    else:
        summary_df = df[['Mean', 'Stdev', '2Stdev', 'BelowStd', 'AboveStd', 'Below2Std', 'Above2Std', 'target']].copy()
        
        return summary_df
    
#END: def visualization_metric_calcs   

def crps_metrics(df):
    """
    Custom function for caluclating relevant metrics for CRPS
    
    Input: 
        df: dataframe
        
    Return:
        
        dataframe with calculations
    """
    
    # Value Storage Lists
    crpsGauss = []
    maeSingular = []
    mae12Singular = []
    
    for index, row in df.iterrows():

        
        # Retrieve the singular prediction (e.g., the ensemble mean already computed elsewhere)
        predSingular = [df.loc[index, 'Mean']]
        
        # Retrieve actual target
        actualSingular = [df.loc[index, 'target']]

        sigma_avg = [df.loc[index, 'Stdev']]
        
        # For singular values, create tensors with shape (1, 1).
        predsSingleTensor = tf.cast(tf.convert_to_tensor(predSingular), tf.float32)
        predsSingleTensor = tf.expand_dims(predsSingleTensor, axis=-1)
        
        actualSingleTensor = tf.cast(tf.convert_to_tensor(actualSingular), tf.float32)
        actualSingleTensor = tf.expand_dims(actualSingleTensor, axis=-1)

        # Tensors for singular values
        sigmaAvgSingleTensor = tf.cast(tf.convert_to_tensor(sigma_avg), tf.float32)
        sigmaAvgSingleTensor = tf.expand_dims(sigmaAvgSingleTensor, axis=-1)

        crps_gaussian = crps_gaussian_tf(predsSingleTensor, sigmaAvgSingleTensor, actualSingleTensor).numpy()
        crpsGauss.append(round(crps_gaussian, 3))

        # Singular metrics.
        maeSingular_val = mae(actualSingleTensor, predsSingleTensor)
        maeSingular.append(round(maeSingular_val, 3))
        
        mae12Singular_val = mae12(actualSingleTensor, predsSingleTensor)
        mae12Singular.append(round(mae12Singular_val, 3))
        
    # Append computed metrics as new columns to the dataframe.
    df['crps_gauss'] = crpsGauss
    df['central_mae'] = maeSingular
    df['central_mae<12'] = mae12Singular
        
    return df

#END: def crps_metrics()
 
def mse_metrics(df):
    """
    Custom function for caluclating relevant metrics for MSE
    
    Input: 
        df: dataframe
        
    Return:
        
        dataframe with calculations
    """

    # Value Storage Lists
    maeSingular = []
    mae12Singular = []
    rmseSingular = []
    
    for index, row in df.iterrows():
        
        # Retrieves values
        predSingular = [df.loc[index, 'Mean']]
        
        # Finds target value
        actualSingular = [df.loc[index, 'target']]

        # Tensors for singular values
        predsSingleTensor = tf.cast(tf.convert_to_tensor(predSingular), tf.float32)
        predsSingleTensor =  tf.expand_dims(predsSingleTensor, axis=-1)

        actualSingleTensor = tf.cast(tf.convert_to_tensor(actualSingular), tf.float32)
        actualSingleTensor =  tf.expand_dims(actualSingleTensor, axis=-1)

        #Grabs values and appends them to a list to be placed into dataframe
        rmseSingular.append(round(rmse_avg(actualSingleTensor, predsSingleTensor), 3))
        maeSingular.append(round(mae(actualSingleTensor, predsSingleTensor), 3))
        mae12Singular.append(round(mae12(actualSingleTensor, predsSingleTensor), 3))
    
    df['rmse_avg'] = rmseSingular
    df['central_mae'] = maeSingular
    df['central_mae<12'] = mae12Singular
    
    return df

#END: def mme_metrics()

def pnn_metrics(df):
    """
    Custom function for caluclating relevant metrics for PNN
    
    Input: 
        df: dataframe
        
    Return:
        
        dataframe with calculations
    """
    
    crpsList = []
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
    
    ######## Code To Run ########
    save = True
    cycles = [1]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    leadTimes = [12, 48, 96]
    architectures = ['mse']#['mse','PNN', "CRPS"]
    iterations = 10#100
    obsVsPred = 'val' # This should be set to either "val" or "testing"
    expanded = False
    
    ##### TO Run the Code#######
    # This line is ran so that the files needed to create the pltos are created and retrieved
    mme_mse_crps_PNN_lead_times_singlePlot(architectures, iterations, cycles, leadTimes, obsVsPred, expanded)
    
    # For loop to loop through leadtimes and create plots for each cycle (rotation)
    for leadTime in leadTimes:
        
        # This Line Will need to be ran to plot the graphs
        decentralized_graphing_driver(architectures, leadTime, cycles, obsVsPred, save)
        
else:
    
    print ("Cross_validation_visuals.py is running")