#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for visualizing hyperparameter tuning performances

Authors: Jarett Woodall and Hector Marrero-Colominas
"""
##########IMPORT STATEMENTS##########

import pandas as pd

import json

import pathlib

from pathlib import Path

import plotly.graph_objects as go 

from plotly.subplots import make_subplots

import plotly.io as pio

import plotly.express as px

import plotly

pio.renderers.default='browser'

import numpy as np

# change this wheeeee
byCycle = False

#Parameter settings
# directory name
# dataset = 'CRPS' # Used for single 
dataset = 'mse' # Used for single 
datasetLst = ['MME_MSE', 'MME_MAPE'] # Comparing multiple loss functions
#leadtime = 12
leadList = [12, 48, 96, 120] # For looping
# metrics =['crps', 'val_crps', 'ssrat', 'val_ssrat', 'ssrel', 'val_ssrel', 'pitd', 'val_pitd', 'mae', 'val_mae', 'mf', 'val_mf', 'di', 'val_di']#['mae', 'val_mae', 'mse', 'val_mse', 'mae12', 'val_mae12']
metrics =['mae', 'val_mae', 'mae12', 'val_mae12', 'me', 'val_me', 'me12', 'val_me12']
# Metric selection to graph
#Used for cases when comparing more then one loss functions
multiLoss = False
metricToGraph = 'mae'
save=True
# This is for file retrieval
objective = 'val_mae'
FONTSIZE = 11 # 11 for subplots, whatever for graph not dealing with individual cycles

#cycle_list = [0, 4]
#Most likely will stay true
allTrials = True

# Training or Validation
val = True
if val: prefix = 'val_'
else:   prefix = ''

if byCycle:
    name = 'ByCycleScatterPlot'
else: 
    name = 'GenScatterPlot'

'''--------------hector-added---------------'''
import plotly.express as px
import plotly.subplots as sp
import pandas as pd

import random
random.seed(42) # Seed the random number generator

axis_font_size = 28
title_font_size = 36
font_size = 28
marker_size = 16


# Training or Validation
val = True
if val: prefix = 'val_'
else:   prefix = ''

'''--------------hector-added---------------'''


##########---Retrieval Function----##############
"""
Function for opening and returning a dataframe for visualization use.

input: leadTime, datset-identifier, objective-(usually mae)
output: dataFrame
"""
def data_reader(dataset, leadTime, objective, allTrials):
    
    #Path Name identifier to differentiate bewteen best and all trials
    if allTrials == True:
        identifier = "ALL_Trials"
    else:
        identifier = "best_trials"
        
    # save_path = "./" + str(leadTime) + 'h_hyperparametersExcel_' + objective + "_" + identifier
    # save_path = f"results/ESB_mape_tuner_results_{leadTime}h_hyperparametersExcel_val_mae_ALL_Trials"
    save_path = f"results/ESB_mse_tuner_results_{leadTime}h_hyperparametersExcel_val_mae_ALL_Trials"
    
    #Opens excel formatted file for use in visualizations
    dataFrame = pd.read_csv(save_path+'.csv')
    
    return dataFrame, save_path

# END: def data_reader()

############---Calculations----################
"""
Creates a dataframe/file that can be used for visualizations 
inputs: df - dataFrame, metric-list of metrics, objective - string
"""
def mean_stdev_calculation(df, metrics, objective, byCycle, multiLoss):
    
    # Creates a new dataframe for storing information
    newDf = pd.DataFrame()
    
    # For loop for iterating through metrics and running calculations
    for item in metrics:
        
        if byCycle == True:
            # changed cycle -> rotation, as per new terminology
            groupByList = ['Rotation:', 'layer_units_num', "input act. func.", "# of layers"]
            
        else:
            groupByList = ['layer_units_num', "input act. func.", "# of layers", ]
            
        if multiLoss == True:
            groupByList.append('Dataset')
        
        #Appends dataframe and computes the average 
        newDf[item + "_mean"] = df.groupby(groupByList, as_index=True)[item].mean()
        newDf[item + "_stdev"] = df.groupby(groupByList, as_index=True)[item].std()
       
    # Calculate average and stdevfor objective score
    newDf[objective + "_Score_mean"] = df.groupby(groupByList,as_index=True)[objective + " Score:"].mean()
    newDf[objective + "_Score_stdev"] = df.groupby(groupByList,as_index=True)[objective + " Score:"].std()
    
    #Reset index for 
    newDf = newDf.reset_index()
    newDf.fillna(0, inplace=True)
    
    return newDf
    
#END: def mean_stdev_calculation(dataFrame, cycleList)

"""
Function for grabbing data from directories, running calculations, and combining into dataset for plotting.

Return: cleanedDf

"""
def multiLossFunctions(datasetList, leadTime, objective, allTrials):
    
    # List for storing dataframes
    lst = []
    
    # Iteration structure iterates through different loss function runs
    for function in datasetLst:
        df, _ = data_reader(function, leadTime, objective, allTrials)
        
        df['Dataset'] = str(function)
        
        lst.append(df)
        
    cleanedDf = pd.concat(lst, axis=0)
        
    return cleanedDf

#END: def multiLossFunctions()
        
##########----Saving----########
"""
Creates a file to save calculated data for user

inputs: df - dataframe, objective- string, leadtime - integer, dataset - string

creates a file and saves it
"""
def stdev_mean_saver(df, objective, leadtime, dataset, allTrials, byCycle):
    #Path Name identifier to differentiate bewteen best and all trials
    if allTrials == True:
        identifier = "ALL_Trials"
    else:
        identifier = "best_trials"
        
    if byCycle == True:
        cycleFunction = "_Cycles_Separated"
        
    else:
        cycleFunction = "_"
    
    df.to_csv(dataset + "_" + str(leadtime) + "h_" + objective + "_Objective_Tuning_Averages_&_Stdev_" + identifier + cycleFunction +".csv")
    
#END: def stdev_mean_saver()

#########------Plotting------#######
"""
Boxplot grpah for hyperparameter results - will have the ability to switch between 
unique hyperparameter combinations and cycles.

input: dataframes - original dataframe and cleaned dataframe with information on hyperparameters

output: boxplot 
"""
def hyperparameter_boxplot(df, objective, leadtime, alltrials, byCycle, metricToGraph, multiLoss, save):

    if multiLoss == True:
        
        # Combines three columns into one for identification purposes
        df['Combination'] =  df['Dataset'] +"_"+ df['layer_units_num'].astype(str) + "_neurons_" + df['# of layers'].astype(str) + "_layers_" + df['input act. func.'].astype(str)+ "_act"
        
    else:
        # Combines three columns into one for identification purposes
        df['Combination'] =  df['layer_units_num'].astype(str) + "_neurons_" + df['# of layers'].astype(str) + "_layers_" + df['input act. func.'].astype(str)+ "_act"
        

    if byCycle == True:
        
        # Creates 
        fig = make_subplots(rows=2, cols=5, horizontal_spacing=0.05)
        
        # Creates a list
        cycleList = df['Cycle:'].unique().tolist()
    
        #Control for creating subplots
        row = 1
        col = 1
        
    elif byCycle == False:
        
        # Creates graph object
        fig = go.Figure()
        
        cycleList = [0]
        #This is here for code to run, does not do anything
        col=0
        
    for cycle in cycleList:

        #Controls the subplots
        if col == 6:
            
            col = 1
            row = 2
            
        if byCycle == True: 
            temp = df.loc[df['Cycle:'] == cycle,:]
            controlList = temp['Combination'].unique().tolist()
            
            print(temp.head())
            
        else:
            controlList = df['Combination'].unique().tolist()
        
    
        # Loop for iterating through and adding traces
        for item in controlList:
            
            # Crude color controller
            if "MAPE" in item:
                color = "purple"
            elif "MSE" in item:
                color = "orange"
            else:
                color = "purple"
            
            # Conditional for subplots or a main plot
            if byCycle == True:
                
                #Filters dataframe
                individual = temp.loc[temp['Combination'] == item,:]
                
                # Creates box
                fig.add_trace(go.Box(
                    x=[item],
                    q1=[individual[metricToGraph].quantile(float(25/100))],
                    q3=[individual[metricToGraph].quantile(float(75/100))],
                    median=[individual[metricToGraph].quantile(float(50/100))],
                    upperfence=[individual[metricToGraph].quantile(float(95/100))],
                    lowerfence=[individual[metricToGraph].quantile(float(5/100))],
                    showlegend=False,
                    boxpoints=False,
                    marker_color=color,
                    line=dict(width=3)
                ), row=row, col=col)  
                
                fig.add_annotation(x=item,
                       y = individual[metricToGraph].max(),
                       text = str(len(individual[metricToGraph])),
                       yshift = 10,
                       showarrow = False, row=row, col=col
                      )
                
                if len(individual[metricToGraph]) == 1:
                    
                    fig.add_trace(go.Scatter(x=[item], y=individual[metricToGraph],marker = dict(color=color)), row=row, col=col)
                    
                
            elif byCycle == False:
                
                temp = df.loc[df['Combination'] == item,:]
                
                fig.add_trace(go.Box(
                    x=[item],
                    q1=[temp[metricToGraph].quantile(float(25/100))],
                    q3=[temp[metricToGraph].quantile(float(75/100))],
                    median=[temp[metricToGraph].quantile(float(50/100))],
                    upperfence=[temp[metricToGraph].quantile(float(95/100))],
                    lowerfence=[temp[metricToGraph].quantile(float(5/100))],
                    showlegend=False,
                    boxpoints=False,
                    marker_color=color,
                    line=dict(width=3),
                ))  
                fig.add_annotation(x=item,
                       y = temp[metricToGraph].max(),
                       text = str(len(temp[metricToGraph])),
                       yshift = 10,
                       showarrow = False
                      )
                if len(temp[metricToGraph]) == 1:
                    
                    fig.add_trace(go.Scatter(x=[item], y=temp[metricToGraph],marker = dict(color=color)))

                    
        # Creates a dataframe that holds the information by Cycle and 
        # I fgoing by Cycle this is the x axis title
        if byCycle == True:
            
            # Variable for differentiation
            titleField = "ByCycle_BoxPlot"
            
            # Creates an order of medians for the boxplots to be in correct order
            ordering = temp.groupby('Combination')[metricToGraph].median().sort_values(ascending=False).index
            
            # Range of axis values
            fig.update_yaxes(range=[0, int(max(temp[metricToGraph]) + 1)], tickmode='linear')
            fig.update_xaxes(title_text= "Cycle: "+ str(cycle), row=row, col=col)
            #fig.update_xaxes(title_text="Hyper. Combos")
            
            # Yaxis label title
            fig.update_yaxes(title_text=str(metricToGraph) + " (" + chr(176)+'C)' , row=1, col=1)
            
            #Orders boxplots by median in descending order
            fig.update_xaxes(categoryorder="array", categoryarray=ordering, row=row, col=col) 
            fig.update_layout(title=titleField, font=dict(size=FONTSIZE))
            
        elif byCycle == False:
            
            #Variable for differentiation
            titleField = "Gen_BoxPlot"
            # Creates an order of medians for the boxplots to be in correct order
            ordering = df.groupby('Combination')[metricToGraph].median().sort_values(ascending=False).index
           
            fig.update_xaxes(title_text= "Hyper. Combos")
           
            # Yaxis label title
            fig.update_yaxes(title_text=str(metricToGraph) + " (" + chr(176)+'C)')
            
            #Orders boxplots by median in descending order
            fig.update_xaxes(categoryorder="array", categoryarray=ordering) 
            fig.update_layout(title=titleField,font=dict(size=FONTSIZE))
            
        col +=1
              
    # Save Path
    save_path = dataset +"_" +str(leadtime) + "h_" +titleField + "_" + str(metricToGraph) + '_mean_TunerResults'
    # Mechanism for Saving
    if save == True:
        p = pathlib.Path("BoxPlot/")
        p.mkdir(parents=True, exist_ok=True)
        fig.write_html( p / f"{save_path}.html")
    else:
        fig.show()
    
#END: def hyperparameter_boxplot()

"""
Heat map plot depicting selected metric

input: dataframe, save_path, objective - string, leadtime, allTrials  - True or False, metricToGraph - string, save-True or false

output: heatmap graph
"""
def heatmap_plot(df, objective, leadtime, alltrials, metricToGraph, multiLoss, save):
    
    if multiLoss == True:
        
        # Combines three columns into one for identification purposes
        df['Combination'] =  df['Dataset'] +"_"+ df['layer_units_num'].astype(str) + "_neurons_" + df['# of layers'].astype(str) + "_layers_" + df['input act. func.'].astype(str)+ "_act"
        
    else:
        # Combines three columns into one for identification purposes
        df['Combination'] =  df['layer_units_num'].astype(str) + "_neurons_" + df['# of layers'].astype(str) + "_layers_" + df['input act. func.'].astype(str)+ "_act"
    

    df = df.filter(['Cycle:', 'Combination', str(metricToGraph) + "_mean", str(metricToGraph) + "_stdev"])

    
    # Clean the dataframe into something easier for the heatmap to see
    meanDf = pd.DataFrame()
    stdevDf = pd.DataFrame()
    
    #Updates dataframes by adding new column
    meanDf['Combination'] = df['Combination']
    stdevDf['Combination'] = df['Combination']
    
    # Sets column as index
    meanDf.set_index('Combination', inplace=True)
    stdevDf.set_index('Combination', inplace=True)
    
    # Nested for loop for iterating through data
    for item in meanDf.index.tolist():
        for column in df['Cycle:'].unique():
            
            # Try catch for dealing with combinations that do not exist in other cycles
            try:
                # Finds corresponding data
                mean = df.loc[(df['Combination'] == item) & (df['Cycle:'] == column), metricToGraph + "_mean"].item()
                stdev = df.loc[(df['Combination'] == item) & (df['Cycle:'] == column), metricToGraph + "_stdev"].item()
                
                # Places information into specific column at a specific row
                meanDf.loc[item, str(column)] = mean
                stdevDf.loc[item, str(column)] = stdev
            except ValueError:
                print()
        
    #Plots data
    fig = px.imshow(meanDf, y=meanDf.index, labels=dict(color=metricToGraph + "_mean (" + chr(176) + 'C)'), aspect='auto')

    # Add custom data and hovertemplate
    fig.update_traces(
        customdata=stdevDf,
        hovertemplate="x: %{x}<br>y: %{y}<br>" + str(metricToGraph) + " Mean: %{z}<br>" + str(metricToGraph) + " Stdev: %{customdata}"
    )

    # Updates to graph structure i.e. titles and font, and locks in the cycles on x-axis
    fig.update_xaxes(title_text= "Cycles", tickmode='linear',  # Set tick mode to linear
                     dtick=1 )
    
    fig.update_layout(title = "HeatMap",font=dict(size=FONTSIZE))
        
    # Save Path
    save_path = dataset +"_" +str(leadtime) + "h_HeatMap_" + str(metricToGraph) + '_mean_TunerResults'
    # Mechanism for Saving
    if save == True:
        p = pathlib.Path("HeatMap/")
        p.mkdir(parents=True, exist_ok=True)
        fig.write_html( p / f"{save_path}.html")
    else:
        fig.show()
        
# END: def heatmap_plot()


"""
Scatter plot depicting selected metric

input: objective,leadtime, allTrials, byCycle, metricToGraph, standardDevLine, save,

output: scatter plot graph
"""
def hyperparameter_scatterplot(df, objective, leadtime, allTrials, byCycle, metricToGraph, save):
    
    # Combines three columns into one for identification purposes
    df['Combination'] = df['layer_units_num'].astype(str) + "_neurons_" + df['# of layers'].astype(str) + "_layers_" + df['input act. func.'].astype(str)+ "_act"

    # Create a unique color for each unique item using a color palette
    unique_items = df['Combination'].unique()

    def random_color():
        return f'#{random.randint(0, 0xFFFFFF):06x}' # Generate a random integer

    # Generate 50 random unique colors
    colors = [random_color() for _ in range(len(unique_items))]
    # colors = px.colors.qualitative.Dark24  # or any other color palette you prefer
    color_mapping = {item: colors[i % len(colors)] for i, item in enumerate(unique_items)}

    # Assuming you have a predefined mapping of cycles to symbols
    cycle_symbol_mapping = {
        1: 'circle',
        2: 'triangle-up',
        3: 'square',
        4: 'diamond',
        5: 'cross',
        6: 'x',
        7: 'pentagon',
        8: 'star',
        9: 'hexagon',
        10: 'triangle-down',
    }

    if byCycle == True:
        fig = go.Figure() # Creates graph object
        cycleList = df['Cycle:'].unique().tolist() # Creates a list
    
        #Control for creating subplots
        row = 1
        col = 1
        
    elif byCycle == False:
        fig = go.Figure() # Creates graph object
        cycleList = [0] # Creates a list

        #This is here for code to run, does not do anything
        col=0

    for cycle in cycleList:

        #Controls the subplots
        if col == 6:
            
            col = 1
            row = 2
    
        # Loop for iterating through and adding traces
        for item in df['Combination'].unique().tolist():
            
            temp = df.loc[df['Combination'] == item,:]

            # Conditional for subplots or a main plot
            if byCycle == True:
                
                titleField = 'ByCycle_Scatterplot'
                color = color_mapping[item] # Get the color for the current item
                # Check the current cycle and set the symbol
                symbol = cycle_symbol_mapping.get(cycle, 'square')  # Default to 'square' if cycle is not in mapping
                
                temp = temp.loc[temp['Cycle:'] == cycle,:]

                num_data_points = len(temp[metricToGraph])  # Get the number of data points

                fig.add_trace(go.Scatter(
                    y=[temp[metricToGraph].std()], 
                    x=[temp[metricToGraph].mean()],
                    mode='markers',
                    marker=dict(
                        size=marker_size, 
                        color=color, 
                        symbol=symbol
                        ),
                    # name="legend_name",
                    marker_symbol=symbol,  # Use the symbol based on cycle
                    text=f'Cycle:{cycle}<br>Data Points: {num_data_points}<br>{metricToGraph}: {round(temp[metricToGraph].mean(), 3)}<br>Std: {round(temp[metricToGraph].std(), 3)}<br> {item}',
                    hoverinfo='text',  # Ensure hover info shows the custom text
                    name=f"{item} - Cycle {cycle}",  # Will create legend entries
                    showlegend=True,  # Ensure it's shown in legend
                    
                ))
                
            elif byCycle == False:
                
                titleField = 'Gen_BoxPlot'
                
                # Get the color for the current item
                color = color_mapping[item]

                num_data_points = len(temp[metricToGraph])  # Get the number of data points

                fig.add_trace(go.Scatter(
                    y=[temp[metricToGraph].std()], 
                    x=[temp[metricToGraph].mean()],
                    mode='markers',
                    marker=dict(
                        size=marker_size, 
                        color=color
                        ),
                    # name="legend_name",
                    text=f'Data Points: {num_data_points}<br>{metricToGraph}: {round(temp[metricToGraph].mean(), 3)}<br>Std: {round(temp[metricToGraph].std(), 3)}<br> {item}',
                    hoverinfo='text',  # Ensure hover info shows the custom text
                    name=f"{item} - Cycle {cycle}",  # Will create legend entries
                    showlegend=True,  # Ensure it's shown in legend
                ))
        col +=1

    fig.update_layout(
            title=dict(
                text=name,  # Add your title here
                x=0.5,                    # Centers the title
                xanchor='center',          # Ensures alignment
                font=dict(size=title_font_size)         # Optional: Adjust title font size
            ),

            xaxis=dict(
                title='Mean ' + str(metricToGraph) + " (" + chr(176)+'C)',
                titlefont=dict(size=axis_font_size),  # Axis title size
                tickfont=dict(size=font_size)    # Tick label size
            ),
            yaxis=dict(
                title='Standard Deviation',
                titlefont=dict(size=axis_font_size),  # Axis title size
                tickfont=dict(size=font_size)    # Tick label size
            ),

            legend_title="Combo and Cycle",
        )

    # Save Path
    save_path = dataset +"_" +str(leadtime) + "h_" +titleField + "_" + str(metricToGraph) + '_mean_TunerResults'
    # Mechanism for Saving
    if save == True:
        p = pathlib.Path("ScatterPlot/")
        p.mkdir(parents=True, exist_ok=True)
        fig.write_html( p / f"{save_path}.html")
        
    else:
        fig.show()

# END: def scatterplot()

"""
Function for calculating mean and standard deviation and creating a new dataframe/file
"""
if __name__ == "__main__":
    
    print("Test function online")
    
    #Main functions to run
   # df, save_path = data_reader(dataset, leadList[0], objective, allTrials)
    #newDf = mean_stdev_calculation(df, metrics, objective, byCycle)
    
    #hyperparameter_scatterplot(df, objective,leadList[0], allTrials, byCycle, metricToGraph, save)
    
    
    
    #BOXPLOT-GRAPH
    #hyperparameter_boxplot(df, objective,leadtime, allTrials, byCycle, metricToGraph, multiLoss, save)
    #newDf = mean_stdev_calculation(df, metrics, objective, True)
    # Saving aggregate metrics
    #stdev_mean_saver(newDf, objective, leadtime, dataset, allTrials, byCycle)
    #heatmap_plot(newDf, save_path, objective, leadtime, True, metricToGraph, multiLoss save)
    
    # Andrew's visuals
    #cleanedDf = multiLossFunctions(datasetLst, leadtime, objective, allTrials)
    # For Heatmap
    #newDf = mean_stdev_calculation(cleanedDf, metrics, objective, True, multiLoss)
    
    for leadTime in leadList:
        
        # Andrew's visuals
        #cleanedDf = multiLossFunctions(datasetLst, leadTime, objective, allTrials)
        # For Heatmap
        #newDf = mean_stdev_calculation(cleanedDf, metrics, objective, byCycle, multiLoss)
        
        #stdev_mean_saver(newDf, objective, leadTime, dataset, allTrials, byCycle)
        
        ####Christians######
        df, save_path = data_reader(dataset, leadTime, objective, allTrials)
        newDf = mean_stdev_calculation(df, metrics, objective, byCycle, False)
        stdev_mean_saver(newDf, objective, leadTime, dataset, allTrials, byCycle)
        
        # for metric in metrics:
        #     hyperparameter_boxplot(df, objective, leadTime, allTrials, byCycle, metric, multiLoss, save)
            
        #     #Commented out when doing combination only boxplot
        #     #heatmap_plot(newDf, objective, leadTime, True, metric,multiLoss, save)
        #     print()


else:
    print('MainFunction Offline')
    print("Armageddon is here please check paths and function names")  
