#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spaghetti Graph Runner
Script to generate spaghetti plots for all 30 model iterations across cycles and lead times

@author: Generated for UQ4ML_WaterTemp
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from helper.utils import fanGraph
from evaluations.cross_validation_visuals_paper import model_parser

######## Configuration ########

# Cycles to process
cycles = [0, 1, 2, 3]

# Lead times to process
leadTimes = [12, 48, 96, 120]

# Architecture (using mape as example, change as needed)
# Options: "mape", "mse", "crps", "PNN"
architectures = ["mape"]

# Number of iterations (30 models)
iterations = 30

# Data type: 'val', 'test', or 'train'
obsVsPred = 'test'

# Directory where model predictions are stored
# This should point to results/{architecture}_results
# Examples: 
#   "results/mape_results" for mape architecture
#   "results/mse_results" for mse architecture
#   "results/crps_results" for crps architecture
MAIN_DIRECTORY = "results/mape_results"

# Save plots or just show them
save_plots = True

# Y-axis limits (for water temperature in Celsius)
y_min = 2
y_max = 20

######## Helper function ########

def create_onset_offset_placeholders(df_length):
    """
    Create placeholder onset/offset dictionaries for the fanGraph function.
    Modify this if you have specific onset/offset times.
    """
    onset = {
        "Earliest": [100],
        "Latest": [200]
    }
    offset = {
        "Earliest": [500],
        "Latest": [600]
    }
    return onset, offset


def run_spaghetti_graphs():
    """
    Main function to generate spaghetti plots for all configurations
    """
    
    print("Starting Spaghetti Graph Generation...")
    print(f"Processing {len(cycles)} cycles, {len(leadTimes)} lead times, {iterations} iterations each")
    
    # Auto-determine main directory from architecture if using default
    directory_map = {
        "mape": "results/mape_results",
        "mse": "results/mse_results",
        "crps": "results/crps_results",
    }
    
    # Loop through each configuration
    for cycle in cycles:
        for leadTime in leadTimes:
            for architecture in architectures:
                
                # Get the correct directory for this architecture
                main_dir = directory_map.get(architecture.lower(), MAIN_DIRECTORY)
                
                try:
                    print(f"\nProcessing: Cycle {cycle}, LeadTime {leadTime}h, Architecture {architecture}")
                    
                    # Get the model name for this configuration
                    model_names = get_model_selection(leadTime, architecture)
                    
                    if not model_names:
                        print(f"  No model found for {architecture} at {leadTime}h - skipping")
                        continue
                    
                    model = model_names[0]  # Use the first (or only) model
                    
                    # Load data for all 30 iterations
                    df_combined = model_parser(
                        MAIN_DIRECTORY=main_dir,
                        model=model,
                        architecture=architecture,
                        obsVsPred=obsVsPred,
                        iterations=iterations,
                        cycle=cycle,
                        leadTime=leadTime
                    )
                    
                    # Prepare data for fanGraph
                    df_for_plot = prepare_data_for_fanGraph(df_combined)
                    
                    # Create placeholder onset/offset
                    onset, offset = create_onset_offset_placeholders(len(df_for_plot))
                    
                    # Define plot file naming
                    plot_name = f"{obsVsPred}_{architecture}_{leadTime}h_cycle_{cycle}"
                    
                    # Generate spaghetti plot (fan=False for spaghetti)
                    fanGraph(
                        df_allMembers=df_for_plot,
                        saver=save_plots,
                        start_date=df_for_plot['dateAndTime'].min().strftime("%Y-%m-%d"),
                        end_date=df_for_plot['dateAndTime'].max().strftime("%Y-%m-%d"),
                        onset=onset,
                        offset=offset,
                        line_count="Total Model Median",
                        y_min=y_min,
                        y_max=y_max,
                        cycle=str(cycle),
                        leadtime=str(leadTime),
                        fan=False  # Set to False for spaghetti plot
                    )
                    
                    print(f"  ✓ Spaghetti plot created: {plot_name}_spaghetti.png")
                    
                except Exception as e:
                    print(f"  ✗ Error processing Cycle {cycle}, LeadTime {leadTime}h: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print("\n" + "="*60)
    print("Spaghetti Graph Generation Complete!")
    print("="*60)


def get_model_selection(leadTime, architecture):
    """
    Get the model name for a given lead time and architecture.
    Based on model_selection_conditional from cross_validation_visuals_paper.py
    """
    
    if leadTime == 12 and architecture == "mape":
        return ['1_layers-leaky_relu-100_neurons']
    elif leadTime == 48 and architecture == "mape":
        return ['3_layers-leaky_relu-32_neurons']
    elif leadTime == 96 and architecture == "mape":
        return ['1_layers-leaky_relu-256_neurons']
    elif leadTime == 120 and architecture == "mape":
        return ['3_layers-relu-256_neurons']
    
    elif leadTime == 12 and architecture == "mse":
        return ['2_layers-leaky_relu-16_neurons']
    elif leadTime == 48 and architecture == "mse":
        return ['3_layers-leaky_relu-16_neurons']
    elif leadTime == 96 and architecture == "mse":
        return ['2_layers-leaky_relu-32_neurons']
    
    elif leadTime == 12 and architecture == "CRPS":
        return ['3_layers-relu-32_neurons']
    elif leadTime == 48 and architecture == "CRPS":
        return ['3_layers-selu-64_neurons']
    elif leadTime == 96 and architecture == "CRPS":
        return ['3_layers-relu-100_neurons']
    
    else:
        return []


def prepare_data_for_fanGraph(df_combined):
    """
    Reformat the dataframe from model_parser to the format expected by fanGraph.
    
    fanGraph expects:
    - Column 'dateAndTime' with datetime values (the function will convert the index)
    - Column 'Target' with actual values
    - Other columns representing predictions from each iteration
    """
    
    df_plot = df_combined.copy()
    
    # Reset index to make date_time a column
    df_plot = df_plot.reset_index()
    
    # Rename 'date_time' to 'dateAndTime' (fanGraph expects this)
    if 'date_time' in df_plot.columns:
        df_plot.rename(columns={'date_time': 'dateAndTime'}, inplace=True)
    
    # Rename 'target' to 'Target' (fanGraph uses capital T)
    if 'target' in df_plot.columns:
        df_plot.rename(columns={'target': 'Target'}, inplace=True)
    
    # Ensure dateAndTime is datetime
    df_plot['dateAndTime'] = pd.to_datetime(df_plot['dateAndTime'])
    
    # Set index as date_time for the date range query, but keep dateAndTime column for the plotting
    # Actually, let's keep both to be safe
    
    return df_plot


if __name__ == "__main__":
    run_spaghetti_graphs()
