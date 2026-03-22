#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spaghetti Graph Runner
Script to generate spaghetti plots for all 30 model iterations across cycles and lead times.
Each individual model prediction is plotted as a separate line.

@author: AK
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from evaluations.cross_validation_visuals_paper import model_parser, model_selection_conditional

######## Helper Functions ########

def load_air_temperature(date_index, obsVsPred):
    """
    Load air temperature data from the ESB_datasets based on the dates in the predictions.
    
    Parameters:
    -----------
    date_index : pd.DatetimeIndex or pd.Index
        The date/time index from the model predictions
    obsVsPred : str
        Either a year (e.g., '2021') or a data split ('test', 'val', 'train')
    
    Returns:
    --------
    pd.Series or None
        Series indexed by date_time with air temperature values, or None if not found
    """
    
    air_temp = None
    
    try:
        # Convert date_index to datetime if it's not already
        if not isinstance(date_index, pd.DatetimeIndex):
            date_index = pd.to_datetime(date_index)
        
        # Extract the year from the first date in the index
        start_year = date_index[0].year
        
        # Determine which ESB dataset to load based on the year
        # ESB datasets are named like esb_2021_2022.csv, esb_2022_2023.csv, etc.
        esb_file = Path('data/ESB_datasets') / f'esb_{start_year}_{start_year + 1}.csv'
        
        if esb_file.exists():
            df_esb = pd.read_csv(esb_file)
            df_esb['date'] = pd.to_datetime(df_esb['date'])
            df_esb = df_esb.set_index('date')
            
            # Rename Air Average to match what we need
            df_esb = df_esb.rename(columns={'Air Average': 'air_temp'})
            
            # Reindex to match the date_index from predictions (nearest neighbor matching)
            air_temp = df_esb['air_temp'].reindex(date_index, method='nearest')
            
            print(f"    Loaded air temperature from {esb_file.name}")
        else:
            print(f"    Warning: Could not find ESB dataset at {esb_file}")
            
    except Exception as e:
        print(f"    Warning: Error loading air temperature: {e}")
        import traceback
        traceback.print_exc()
    
    return air_temp

######## Configuration ########

# Cycles to process
cycles = [0]

# Lead times to process
leadTimes = [12]

# Architecture
# Options: "mape", "mse", "crps", "PNN"
architectures = ["mape"]

# Number of iterations (30 models)
iterations = 30

# Data type: 'val', 'test', or 'train'
obsVsPred = '2021'

# Output directory for plots
output_directory = "spaghetti_plots"

# Y-axis limits (for water temperature in Celsius)
y_min = 2
y_max = 20

# Plot colors for iterations (will cycle through)
iteration_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]



def create_spaghetti_plot(df_combined, leadTime, cycle, architecture, obsVsPred):
    """
    Create a spaghetti plot showing all individual model predictions.
    Each iteration is plotted as a separate line with the ensemble mean highlighted.
    
    Parameters:
    -----------
    df_combined : pd.DataFrame
        DataFrame from model_parser with all iterations combined.
        Index is 'date_time', columns include 'target' and prediction columns
        suffixed with '_iteration_X'
    leadTime : int
        Lead time in hours
    cycle : int
        Cycle number
    architecture : str
        Model architecture name
    obsVsPred : str
        Data split ('test', 'val', 'train')
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    
    fig = go.Figure()
    
    # Extract target (water temperature)
    target = df_combined['target']
    date_time = df_combined.index
    
    # Load air temperature data
    air_temp = load_air_temperature(date_time, obsVsPred)
    
    # Get all prediction columns (those with _iteration_ in the name)
    # Sort numerically by iteration number, not alphabetically
    pred_columns = sorted(
        [col for col in df_combined.columns if '_iteration_' in col],
        key=lambda x: int(x.split('_iteration_')[-1])
    )
    
    if not pred_columns:
        print(f"    Warning: No prediction columns found in dataframe")
        return None
    
    # Add a trace for each iteration (individual model predictions)
    for idx, col in enumerate(pred_columns):
        iteration_num = col.split('_iteration_')[-1]
        color = iteration_colors[idx % len(iteration_colors)]
        
        fig.add_trace(go.Scatter(
            x=date_time,
            y=df_combined[col],
            mode='lines',
            name=f'Iteration {iteration_num}',
            line=dict(color=color, width=1),
            opacity=0.6,
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
        ))
    
    # Calculate and add ensemble mean (average of all iterations)
    ensemble_mean = df_combined[pred_columns].mean(axis=1)
    fig.add_trace(go.Scatter(
        x=date_time,
        y=ensemble_mean,
        mode='lines',
        name='Ensemble Mean',
        line=dict(color='#A8E6A1', width=5, dash='dash'),
        opacity=1.0,
        connectgaps=False,
        hovertemplate='<b>Ensemble Mean</b><br>Date: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
    ))
    
    # Add observed/target line (water temperature)
    fig.add_trace(go.Scatter(
        x=date_time,
        y=target,
        mode='lines',
        name='Observed Water Temp',
        line=dict(color='black', width=3),
        opacity=1.0,
        hovertemplate='<b>Observed Water Temp</b><br>Date: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
    ))
    
    # Add air temperature line if available
    if air_temp is not None:
        fig.add_trace(go.Scatter(
            x=date_time,
            y=air_temp,
            mode='lines',
            name='Observed Air Temp',
            line=dict(color='#7c0a02', width=3),
            opacity=1.0,
            hovertemplate='<b>Observed Air Temp</b><br>Date: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
        ))
    
    # Add sea turtle threshold line
    fig.add_hline(
        y=8,
        line_dash="dot",
        line_color="red",
        annotation_text="Sea Turtle Threshold",
        annotation_position="top left",
        annotation_font_size=14,
        annotation_font_color="red"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Spaghetti Plot: {architecture.upper()} - {leadTime}h Lead Time - Cycle {cycle} ({iterations} Models)',
            font=dict(size=20)
        ),
        xaxis_title='Date/Time',
        yaxis_title='Temperature (°C)',
        height=700,
        width=1600,
        template='plotly_white',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        ),
        font=dict(size=12)
    )
    
    return fig


def run_spaghetti_graphs():
    """
    Main function to generate spaghetti plots for all configurations
    """
    
    # Create output directory if it doesn't exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SPAGHETTI GRAPH GENERATION - PLOTLY VERSION")
    print("="*70)
    print(f"Processing {len(cycles)} cycles × {len(leadTimes)} lead times × {len(architectures)} architecture(s)")
    print(f"Total plots to generate: {len(cycles) * len(leadTimes) * len(architectures)}")
    print(f"Output directory: {Path(output_directory).resolve()}")
    print("="*70 + "\n")
    
    plot_count = 0
    error_count = 0
    
    # Loop through each configuration
    for cycle in cycles:
        for leadTime in leadTimes:
            for architecture in architectures:
                
                try:
                    print(f"Processing: Cycle {cycle}, LeadTime {leadTime}h, Architecture {architecture}")
                    
                    # Get model name for this configuration
                    model_names = model_selection_conditional(leadTime, architecture)
                    
                    if not model_names:
                        print(f"  ⚠ No model found for {architecture} at {leadTime}h - skipping\n")
                        continue
                    
                    model = model_names[0]
                    
                    # Determine main directory based on architecture
                    main_dir = f"results/{architecture.lower()}_results"
                    
                    print(f"  Loading {iterations} iterations from: {main_dir}/{leadTime}h/")
                    
                    # Load data for all iterations combined
                    df_combined = model_parser(
                        MAIN_DIRECTORY=main_dir,
                        model=model,
                        architecture=architecture,
                        obsVsPred=obsVsPred,
                        iterations=iterations,
                        cycle=cycle,
                        leadTime=leadTime
                    )
                    
                    print(f"  Data loaded: {len(df_combined)} time steps, {len([c for c in df_combined.columns if '_iteration_' in c])} iterations")
                    
                    # Create spaghetti plot
                    fig = create_spaghetti_plot(df_combined, leadTime, cycle, architecture, obsVsPred)
                    
                    if fig is None:
                        print(f"  ✗ Failed to create plot\n")
                        error_count += 1
                        continue
                    
                    # Define output filename and path
                    filename = f"{obsVsPred}_{architecture}_{leadTime}h_cycle_{cycle}_spaghetti.html"
                    filepath = Path(output_directory) / filename
                    
                    # Save HTML plot
                    fig.write_html(str(filepath))
                    
                    print(f"  ✓ Plot saved: {filepath.name}")
                    print(f"    Full path: {filepath.resolve()}\n")
                    
                    plot_count += 1
                    
                except FileNotFoundError as e:
                    print(f"  ✗ File not found: {str(e)}\n")
                    error_count += 1
                    
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print()
                    error_count += 1
    
    print("="*70)
    print(f"GENERATION COMPLETE")
    print(f"  Successfully created: {plot_count} plots")
    print(f"  Errors encountered: {error_count}")
    print(f"  Output directory: {Path(output_directory).resolve()}")
    print("="*70)


if __name__ == "__main__":
    run_spaghetti_graphs()
