#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 13:18:42 2025

@author: Jarett Woodall

This file serves as a host for all of the functions needed for the various
figures created for the UQ Paper. (Figure 4, 5, 6, and 12). The figures created
using this file are outputted to a folder called "Paper_Figures".
"""

######### Imports #########
import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np

from matplotlib.ticker import FuncFormatter

import os

import matplotlib.patches as mpatches

from pathlib import Path

####### PLotting Functions ########

def figure_4_plot():
    """
     This function creates a plot folling the structure of Figure 4 in the UQ
     Paper. IT assumes that the relevant aggregate table files exist for both
     validation and testing.
    """
    aggregate_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "Aggregate_Tables" 
    
    # Read in two dataframes for concatenation and preparation for plotting
    valDf = pd.read_csv(aggregate_dir / 'val_cold_metrics_results_performance_byCycleAggregateTable.csv')
    testDf = pd.read_csv( aggregate_dir / 'test_cold_metrics_results_performance_byCycleAggregateTable.csv')
    df = pd.concat([valDf, testDf], axis=0, ignore_index=True)
    
    # Flip sign of ME and ME12 to reflect prediction - observation
    df["me"] = -df["me"]
    df["me12"] = -df["me12"]
    
    # Capitalize architecture names
    df["architecture"] = df["architecture"].str.upper()
    
    # Create architecture-dataset label (e.g., PNN_val)
    df["arch_dataset"] = df["architecture"] + "_" + df["dataset"]
    
    # Metrics to plot
    metrics = ["crps", "ssrat", "ssrel", "pit", "mae", "mae12", "me", "me12"]
    temp_metrics = {"crps", "mae", "mae12", "me", "me12"}
    
    # Define color palette
    custom_palette = {
        "CRPS_val": '#E399DA',
        "CRPS_test": '#A02B93',
        "PNN_val": 'lightskyblue',
        "PNN_test": "dodgerblue",
        "MSE_val": 'yellowgreen',
        "MSE_test": 'forestgreen'
    }
    
    arch_dataset_order = ["PNN_val", "PNN_test", "MSE_val", "MSE_test", "CRPS_val", "CRPS_test"]
    
    # Setup plot
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 12))
    axes = axes.flatten()
    
    # Sorted lead times
    lead_times = sorted(df["leadTime"].unique())
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
    
        sns.boxplot(
            data=df,
            x="leadTime",
            y=metric,
            hue="arch_dataset",
            palette=custom_palette,
            ax=ax,
            order=lead_times,
            hue_order=arch_dataset_order,
            width=0.7,
            fliersize=3,
            linewidth=1,
            dodge=True,
            flierprops=dict(marker='D', markerfacecolor='black', markersize=4)
        )
        for line in ax.lines:
            ydata = line.get_ydata()
            # Check if line is horizontal and short (median line)
            if len(ydata) == 2 and ydata[0] == ydata[1]:
                line.set_linewidth(3) 
        # Titles and axis labels
    
        if metric == "me":
    
            metric = "BIAS"
            
        elif metric == 'me12':
            
            metric = "BIAS<12"
            
        ax.set_title(metric.upper(), fontsize=22, weight='bold')
        unit = " (°C)" if metric in temp_metrics else ""
        ax.set_ylabel(f"{metric.upper()}{unit}", fontsize=20)
        ax.set_xlabel("")
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_xticklabels([f"{lt}-HR" for lt in lead_times], fontsize=18, fontstyle='italic')  # Set labels

    
        # Add vertical separators between lead times
        for idx in range(1, len(lead_times)):
            ax.axvline(x=idx - 0.5, color='gray', linestyle='--', linewidth=1)
    
        # Remove individual legends
        ax.get_legend().remove()
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.18, 1, 1])  # Make room for two-row legend
    
    # Custom two-row legend
    validation_labels = [
        "PNN-MME (Validation)", "MSE-MME (Validation)", "CRPS-MME (Validation)"
    ]
    testing_labels = [
        "PNN-MME (Testing)", "MSE-MME (Testing)", "CRPS-MME (Testing)"
    ]
    validation_colors = ['lightskyblue', 'yellowgreen', '#E399DA']
    testing_colors = ['dodgerblue', 'forestgreen', '#A02B93']
    
    # Create two rows of legend handles
    validation_lines = [Line2D([0], [0], color=c, lw=10) for c in validation_colors]
    testing_lines = [Line2D([0], [0], color=c, lw=10) for c in testing_colors]
    
    # Validation legend (top row)
    fig.legend(
        handles=validation_lines,
        labels=validation_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        fontsize=22,
        frameon=False,
    )
    
    # Testing legend (bottom row)
    fig.legend(
        handles=testing_lines,
        labels=testing_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=22,
        frameon=False,

    )
    
    save_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" "Paper_Figures"

    # Create it if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_dir / 'Figure4.png', dpi=900)

# END: def figure_4_plots()

def figure_5_6_plot(obsVsPred, threshold):
    
    """
    This function is designed to output the figures 5 and 6 based off of user 
    selections.
    
    Inputs:
        
        obsVsPred: string - use '2021', '2024', 'val' or 'testing'
        threshold: integer representing threshold
    """
    # Specifies Path to Aggregate Tables Folder
    aggregate_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "Aggregate_Tables"
    
    # Read in two dataframes for concatenation and preparation for plotting
    df1 = pd.read_csv(aggregate_dir / f'{obsVsPred}_cold_metrics_results_performance_byCycleAggregateTable.csv')
    df2 = pd.read_csv(aggregate_dir / f'{obsVsPred}_temp<{threshold}_metrics_results_performance_byCycleAggregateTable.csv')
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    
    # Flip sign of ME and ME12 to reflect prediction - observation
    df["me"] = -df["me"]
    df["me12"] = -df["me12"]
    
    # Renames metrics for visuals
    df.rename(columns={'me': 'BIAS'}, inplace=True)
    df.rename(columns={'pit': 'PITD'}, inplace=True)

    # Change architecture 'mse' to 'MSE' for consistency in labels
    df['architecture'] = df['architecture'].replace('mse', 'MSE')
    
    df['selection'] = df['selection'].str.lower()
    
    # Plot settings
    metrics = ['crps', 'ssrat', 'ssrel', 'PITD', 'mae', 'BIAS']
    lead_times = sorted(df['leadTime'].unique())
    architectures = ['PNN', 'CRPS', 'MSE']
    datasets = ['cold', 'temp<' + str(threshold)]
    
    # Architecture-based colors
    architecture_colors = {
        'PNN': 'dodgerblue',
        'CRPS': '#A02B93',
        'MSE': 'forestgreen',
    }
    
    # Custom group column: dataset-architecture
    df['group'] = df.apply(lambda row: f"{row['selection']}-{row['architecture']}", axis=1)
    group_order = [f"{ds}-{arch}" for ds in datasets for arch in architectures]
    palette = {f"{ds}-{arch}": architecture_colors[arch] for ds in datasets for arch in architectures}
    
    # Plot
    fig, axes = plt.subplots(len(metrics), len(lead_times), figsize=(6.5, 2.0 * len(metrics)), sharey='row')
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, metric in enumerate(metrics):
    
        for j, lt in enumerate(lead_times):
            ax = axes[i, j] if len(metrics) > 1 else axes[j]
            subset = df[df['leadTime'] == lt].copy()
    
            sns.boxplot(
                data=subset,
                x='group',
                y=metric,
                hue='group',  # Set hue to the same as x
                order=group_order,
                palette=palette,
                width=0.6,
                linewidth=0.8,
                ax=ax,
                legend=False,  # Avoid duplicate legend
                flierprops=dict(marker='D', markerfacecolor='black', markersize=4)
            )

    
            # Only set titles on the first row
            if i == 0:
                ax.set_title(f'{lt}hr', fontsize=22, weight='bold')
            else:
                ax.set_title('')
    
            ax.set_xlabel('')
    
            # Y-label
            if metric in ['crps', 'mae', 'BIAS']:
                ax.set_ylabel(f'{metric.upper()}(°C)', fontsize=18,weight='bold')
    
            else:
                ax.set_ylabel(metric.upper(), fontsize=18,weight='bold')
    
            # Only show x-axis tick labels on the last row
            if i == len(metrics) - 1:
                ax.set_xticks([1, 4])
                ax.set_xticklabels(['Cold\nSeason', 'Sub-\n' +str(threshold) +'°C'], fontsize=15)
            else:
                ax.set_xticks([])  # Remove x ticks for other rows

    
            ax.tick_params(axis='x', which='both', length=4)
            #ax.tick_params(axis='x', labelsize=22)
            ax.tick_params(axis='y', labelsize=15)
    
            ax.yaxis.grid(True, color='grey', linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
    
            # Vertical separator between Cold and Temps<12
            ax.axvline(2.5, color='grey', linestyle='-', linewidth=1)
            
            sig_fig_formatters = {
            'crps': lambda x, _: f'{x:.2f}',
            'ssrat': lambda x, _: f'{x:.2f}',
            'ssrel': lambda x, _: f'{x:.2f}',
            'PITD': lambda x, _: f'{x:.2f}',
            'mae': lambda x, _: f'{x:.2f}',
            'BIAS': lambda x, _: f'{x:.2f}',
            }
            ax.yaxis.set_major_formatter(FuncFormatter(sig_fig_formatters[metric]))
            
    # Legend (by architecture)
    custom_lines = [
        plt.Line2D([0], [0], color=architecture_colors[arch], lw=8)
        for arch in architectures
    ]
    legend_labels = [f"{arch}-MME" for arch in architectures]
    
    # Move legend slightly inside the figure and reduce padding
    fig.legend(
        custom_lines,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.00),  # slightly inside
        ncol=3,
        fontsize=18,
        frameon=False
    )
    
    # Use tight_layout for better spacing, but restrict rect so it doesn't interfere with the legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # top, bottom space adjusted
    
    # Changes file name pending on what is being output
    if obsVsPred == '2024':
    
        figname = 'Figure6.png'
        
    elif obsVsPred == 'test':
        figname = 'Figure5.png'
        
    else:
        figname = 'Fig5_6_Plot_' + obsVsPred

    
    save_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "Paper_Figures"

    # Create it if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Save with bounding box tight to remove unnecessary white space
    fig.savefig(save_dir / figname, dpi=900, bbox_inches='tight')
    
# END: def figure_5_6_plot()

def figure_12_plot(padded):
    """
    This function is designed to output the figure 12; if the user is using a
    different padded value it will be taken into the function.
    
    Inputs:
        
        padded: integer representing hours before and after URI
    """
    aggregate_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "Aggregate_Tables"
    # Reads in the correct Csv containing the relevant information
    df = pd.read_csv(aggregate_dir / '2021_URI_padded_metrics_results_performance_byCycleAggregateTable.csv') 
    
    # Flip sign of ME and ME12 to reflect prediction - observation
    df["me"] = -df["me"]
    df["me12"] = -df["me12"]
    
    uriVal = 'URI' + str(padded)
    uriLab = "URI-" + str(padded)
    
    # Set up colors
    architecture_colors = {
        ('CRPS', uriVal): '#A02B93',
        ('mse', uriVal): 'forestgreen',
        ('PNN', uriVal): 'dodgerblue',
    }
    
    metrics = ["crps", "ssrat", "ssrel", "pit", "mae", "me", "rmse"]
    temp_metrics = {"me", "mae12", "me12", "mae", "rmse", "crps"}
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    axes = axes.flatten()
    
    # Loop through first 6 metrics
    for i, metric in enumerate(metrics[:-1]):
        ax = axes[i]
    
        box_data = []
        box_colors = []
        box_positions = []
        pos = 0
    
        leadtime_positions = {}
    
        space_within_group = 0
        space_between_arch = 0.4
        space_between_leadtime = 1
    
        for leadtime in sorted(df["leadTime"].unique()):
            leadtime_positions[leadtime] = []
    
            for arch in ["PNN", "CRPS", "mse"]:
                for dataset in [uriVal]:
                    data = df[
                        (df["leadTime"] == leadtime) &
                        (df["architecture"] == arch) &
                        (df["selection"] == dataset)
                    ][metric].dropna()
    
                    if not data.empty:
                        box_data.append(data)
                        box_colors.append(architecture_colors[(arch, dataset)])
                        box_positions.append(pos)
                        leadtime_positions[leadtime].append(pos)
                        pos += space_within_group + 1
    
                pos += space_between_arch
    
            pos += space_between_leadtime
    
        # Draw boxplots
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.8,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5, color='black'),
            flierprops=dict(marker='D', markerfacecolor='black', markersize=4)
        )
    
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
    
        for lt in list(leadtime_positions.keys())[1:]:
            sep_idx = min(leadtime_positions[lt]) - 1.2
            ax.axvline(sep_idx, color='grey', linestyle='--', linewidth=1)
    
        for lt, positions in leadtime_positions.items():
            center_pos = np.mean(positions)
            ax.text(center_pos, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                    f"{lt}-HR", ha='center', va='top', fontsize=26, fontstyle='italic')
    
        ax.set_xticks([])
        if box_positions:
            min_pos = min(box_positions)
            max_pos = max(box_positions)
            buffer = 0.8  # Adjust if needed for more space
            ax.set_xlim(min_pos - buffer, max_pos + buffer)
        if metric == "me":
            display_metric = "BIAS"
        elif metric == "pit":
            display_metric = "PITD"
        elif metric == "crps":
            display_metric = "CRPS"
        else:
            display_metric = metric.upper()
    
        unit = " (°C)" if metric in temp_metrics else ""
        ax.set_title(display_metric, fontsize=30, weight='bold')
        ax.set_ylabel(f"{display_metric}{unit}", fontsize=26)
        ax.tick_params(axis='y', labelsize=26)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='grey')
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=architecture_colors[('CRPS', uriVal)], label='CRPS (' + uriLab + ")"),
        mpatches.Patch(color=architecture_colors[('mse', uriVal)], label='MSE (' + uriLab + ")"),
        mpatches.Patch(color=architecture_colors[('PNN', uriVal)], label='PNN (' + uriLab + ")"),
    ]
    
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=6,
        frameon=True,
        fontsize=26,
        bbox_to_anchor=(0.5, -0.03)
    )
    
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.93, bottom=0.12,
        wspace=0.25, hspace=0.35
    )

    # Build Path
    save_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "Paper_Figures"

    # Create it if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Saves Plot
    plt.savefig(save_dir / 'Figure12.png', dpi=900) 
    
# END: def figure_12_plot()

####### Helper Function Checking For Files ######

def existance_checker():
    
    """
    Function to check for the relevant files needed to run Figure_4 code.
    """
    aggregate_dir = Path("UQ4ML_WaterTemp") / "src" / "UQ_Visuals_Tables_Files" / "Aggregate_Tables"
    files = {
        'val': aggregate_dir / 'val_cold_metrics_results_performance_byCycleAggregateTable.csv',
        'test': aggregate_dir / 'test_cold_metrics_results_performance_byCycleAggregateTable.csv'
    }

    # Check for missing files
    missing = [name for name, path in files.items() if not os.path.isfile(path)]
    
    # Print Message for Missing Files
    if missing:
        print("Missing file(s):", ", ".join(f"{name}: {files[name]}:" for name in missing))
        print("If missing test or val, make sure you have the files needed to create the aggregate tables and the aggregate tables themselves.")
        return False
    
    return True

# END: def existance_checker()

######## DEBUGGING CODE ###########
if __name__ == "__main__":
    
    # Variables that are changeable
    figure_4_plot()
    obsVsPred = "2021"
    padded = 24
    #figure_5_6_plot(obsVsPred, 12)
    #figure_12_plot(padded)
    
else:
    
    print("AggregateTables is running")