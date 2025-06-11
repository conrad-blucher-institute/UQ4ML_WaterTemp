#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:39:01 2025

This file serves as the driver for the cross validation visuals and 
aggregate table code. PLease follow the instructions in order to run this code 
effectively.

@author: Jarett Woodall

"""

####### Imports ########

from cross_validation_visuals_paper import mme_mse_crps_PNN_lead_times_singlePlot, decentralized_graphing_driver

from aggregate_tables import aggregateTable

from boxplot_figures import figure_4_plot, figure_5_6_plot, figure_12_plot, existance_checker

######## Variables ########
"""
This variable will be set to false to run the needed files for the aggregate table.
If this variable is set to True and throwing an error, make sure you have 
run this file with this variable set to False so that the tables can then be 
created once these files are created.

IMPORTANT:
Set to False first to ensure the files are there, 
this exists so you dont have to run the intensive functions again
"""
runAggregateCode = False #False

# Variable to save the plots will be set to True, otherwise false
save = True

# List of cycles to create necessary files for plotting and aggregate tables 
cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# List of leadTimes to make visuals and tables for
leadTimes = [12, 48, 96]

# Architecture lists; code will only work if you use these three types, any deviation will require refactoring
architectures = ['mse','PNN', "CRPS"]

# This should match the number of iterations you ran while training, you can have this number set to something smaller
iterations = 100

"""
Should be set to either "val", "test", or "train" depending on what information
you want to visualize. If you ran the 2021 or 2024 testing years please use 
'2021' or '2024' to retrieve relevant information.
"""
obsVsPred = 'val' 

"""
Set to true if you want csvs outputted that contain all of the predictions 
instead of having just summary statistics.

WARNING if this is set to True, the files will grow to very large sizes, 
particularly for CRPS
"""
expanded = False

###################################################################

##### CODE EXECUTION for Plotting and Calculation Files

if runAggregateCode == False:
    
    # This line is ran so that the files needed to create the pltos are created and retrieved
    # If you want to modify the graph code you can uncomment this line if the files have been created; it will save time
    # The loop below assumes that the files created by this function exist.
    mme_mse_crps_PNN_lead_times_singlePlot(architectures, iterations, cycles, leadTimes, obsVsPred, expanded)
    
    # For loop to loop through leadtimes and create plots for each cycle (rotation)
    for leadTime in leadTimes:
        
        # This Line Will need to be ran to plot the graphs
        decentralized_graphing_driver(architectures, leadTime, cycles, obsVsPred, save)
        
else:
    
    ##### Aggregate Table Execution Code #######
    byCycle = True # Set to false if you wish to run byCombo code
    
    # Chnage this variable if you wish to see metrics for predictions below a certain threshold specified
    threshold = 12
    
    # Controls how many hours before and after Winter Storm URI should be included in calculations
    padding = 24
    
    # Conditional to run function if you are working with the 2021 year; obsVsPred must be set to '2021' to pass if
    if obsVsPred == '2021':
        
        # Driver function call to include files for URI
        aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred, True , padding)
        
        # Outputs Figure 12 
        figure_12_plot(padding)
        
    elif obsVsPred == '2024':
    
        # Runs the Function helper
        aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)
        
        # Figure 6 is ran and outputted
        figure_5_6_plot(obsVsPred, threshold)
    
    elif obsVsPred == 'test':
    
        # Runs the Function helper
        aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)
        
        # Figure 5 is ran and outputted
        figure_5_6_plot(obsVsPred, threshold)
        
    else:
        # Runs the Function helper
        aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)
        
########################################################################
""" This code checks to make sure the aggregate tables for val and test
 exist so that Figure 4 can be created. If the aggregate tables do not exist
 please run the above script to output the correct outputs so that figure 4 
code can run."""
if existance_checker():
    
    # Function to run figure 4
    figure_4_plot()



 