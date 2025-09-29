import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime
import os

# Add project root (parent of src) to sys.path BEFORE any src import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluations.cross_validation_visuals_paper import model_selection_conditional, model_parser, visualization_metric_calcs, crps_metrics, pnn_metrics, mse_metrics

from evaluations.evaluation_functions import *

from evaluations.aggregate_tables import aggregateTableOpsTest

from evaluations.utils_visuals import file_retriever

# Variables for plots
models = ['PNN', 'CRPS']
#start = '12/23/2022 16:00'
#end = '12/27/2022 17:00'
start = '01/16/2024 06:00'
end = '01/21/2024 15:00'
#start = '02/14/2021 06:00'
#end = '02/19/2021 22:00'
leadTimes = [12, 48, 96, 120] # 12, 48, 96, 120
padding = 48
cycle = 8 
threshold = 8 # Threshold for low temperature detection
min_duration = 24 # Minimum duration for low temperature periods in
block_size = 6 # Number of rows per composite block
obsVsPred_list = ['MM-EX', 'MM-IN', 'ALL', 'PP']#['Median', 'min','max', 1, 5, 25, 75, 95, 99]

#obsVsPred = 'test' # 'val', 'test', or 'train'
expanded = False
composite = True
save = True
uncertainty = True

# Composite metrics driver
def runner(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred):

    df_container = {}
    for model in models:

        for leadTime in leadTimes:

            model_list = model_selection_conditional(leadTime, model)
                
            # Loop over each hyperparameter combo.
            for spec in model_list:

                if composite:

                    # Calls function to retrieve plotting data
                    print(f"Processing {model} for lead time {leadTime} and cycle {cycle}")
                    df = file_retriever(leadTime, cycle, model, obsVsPred, start, end)

                    df = create_composite_metrics(df, block_size)

                    start_dt = datetime.strptime(start, '%m/%d/%Y %H:%M')
                    end_dt = datetime.strptime(end, '%m/%d/%Y %H:%M')

                    # Format for filename or variable name
                    results_path = f"TWC_results_{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}"

                    input_path = Path("src") / results_path / "TWC_composite_UQ_Files"
                    filename = f"{obsVsPred}_{leadTime}h_{model}_Cycle_{cycle}_Model_{spec}.csv"
                    full_path = input_path / filename

                    # Ensure the directory exists
                    input_path.mkdir(parents=True, exist_ok=True)

                    # Save the DataFrame
                    df.to_csv(full_path, index=False)

                    metrics = ["crps_gauss", 'mean_error', 'central_mae', 'pit', 'ssrat', 'ssrel']

                    plot_type = "_composite"

                    df_container[f"{leadTime}h_{model}"] = df

                else:
                    # Calls function to retrieve plotting data
                    df = file_retriever(leadTime, cycle, model, obsVsPred, start, end)
                    metrics = ["crps_gauss", 'mean_error', 'central_mae']

                    df_container[f"{leadTime}h_{model}"] = df
                    plot_type = "_hourly"
                    print(start)
                    print(end)

                for metric in metrics:

                    heatmap_plot_b(df, leadTime, cycle, model, metric, start, end, "heatmap_b" + plot_type, composite, obsVsPred)

    for metric in metrics:
        line_plot_b(df_container, leadTimes, cycle, metric, start, end, "line_plot_b" + plot_type, composite, obsVsPred)
# END: def runner()

def runner_uncertainty(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred_list):

    """
    This function serves as a driver that will retrieve created files in order to create plots.

    inputs:
        leadTimes : list of integers, 
        cycle: integer, 
        models: list of strings, 
        start: string representing start date and time, 
        end: string representing end date and time,
        block_size: integer representing number of rows per composite block,
        composite: boolean (True or False)
    """

    df_container = {}
    for model in models:

        for leadTime in leadTimes:

            model_list = model_selection_conditional(leadTime, model)
                
            # Loop over each hyperparameter combo.
            for spec in model_list:

                for obsVsPred in obsVsPred_list:

                    if composite:

                        # Calls function to retrieve plotting data
                        print(f"Processing {model} for lead time {leadTime} and cycle {cycle}")
                        df = file_retriever(leadTime, cycle, model, obsVsPred, start, end, "TWC_composite_UQ_Files")

                        metrics = ["crps_gauss", 'mean_error', 'central_mae', 'pit', 'ssrat', 'ssrel']

                        plot_type = "_composite"

                        df_container[f"{leadTime}h_{model}_{obsVsPred}"] = df

                    else:
                        # Calls function to retrieve plotting data
                        df = file_retriever(leadTime, cycle, model, obsVsPred, start, end)
                        metrics = ["crps_gauss", 'mean_error', 'central_mae']
                        df_container[f"{leadTime}h_{model}_{obsVsPred}"] = df
                        plot_type = "_hourly"
                        print(start)
                        print(end)
    name = "min_max_included_excluded_all_members_PP" 
    for metric in metrics:
        line_plot_b(df_container, leadTimes, cycle, metric, start, end, "line_plot_b" + plot_type, composite, name)
        # Line to remove PP from dictionary for pairwise and architecture comparisons
        cleaned_dict = {k: v for k, v in original_dict.items() if 'PP' not in k}

        # Changes name back
        name = "min_max_included_excluded_all_members"
        plot_pairwise_differences_by_leadtime(df_container, leadTimes, cycle, metric, start, end, "line_plot_c_differences_approaches" + plot_type, composite, name)
        plot_architecture_differences_by_leadtime(df_container, leadTimes, cycle, metric, start, end, "line_plot_d_differences_archiectures" + plot_type, composite, name)
# END: def runner_uncertainty()

def decentralized_graphing_driver(architectures, leadTime, cycles, obsVsPred, save, composite, start, end, uncertainty):
    """
    This function serves as a driver that will retrieve the created files and 
    plot standard deviation plots.

    inputs: 
    architectures : list of strings, 
    leadTime : integer, 
    cycles: list of integers, 
    obsVsPred: string to differentiate between data, 
    save: boolean (True or False)

    outputs: 
    plots made using plotly
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

                if uncertainty == True:
                    obsVsPred_list = ['MM-EX', 'MM-IN', 'ALL', 'PP']
                else:
                    obsVsPred_list = [obsVsPred]

                for obsVsPred1 in obsVsPred_list:

                    if composite == True:
                        identifier = "TWC_composite_UQ_Files"
                        saveIdentifier = "_composite"
                    else:
                        identifier = "TWC_hourly_UQ_Files"
                        saveIdentifier = "_hourly"

                    start_dt = datetime.strptime(start, '%m/%d/%Y %H:%M')
                    end_dt = datetime.strptime(end, '%m/%d/%Y %H:%M')

                    # Format for filename or variable name
                    results_path = f"TWC_results_{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}"

                    # Utilizes Path for cross compatability regardless of macOs or Windows
                    input_path = Path("src") / results_path / identifier/ f"{obsVsPred1}_{leadTime}h_{architecture}_Cycle_{cycle}_Model_{model}.csv"
                    df = pd.read_csv(input_path)

                    df['date_time'] = pd.to_datetime(df["date_time"])

                    df = df.set_index('date_time')

                    # Create a key that encodes the combo, architecture, and leadTime.
                    key = f"{architecture}-{leadTime}h_{obsVsPred1}"
                    modelsDict[key] = df
                    
                # Update the cycle-level dictionary with the lead time–specific data.
                modelsDict_cycle.update(modelsDict)
        
        # Combines architectures to make an effective title
        if len(architectures) > 1:
            
            arch_title = ", ".join(architectures)
            
        else:
            
            arch_title = architectures[0]

        if uncertainty == True:
            obsVsPred2 = "min_max_included_excluded_all_members_PP"
        else:
            obsVsPred2 = obsVsPred1
        
        # Call the boxplot function with the aggregated data.
        fig = standardDeviationFan_leadTime_plot(modelsDict_cycle, leadTime, arch_title, cycle, obsVsPred2, save, composite, uncertainty)

        save_figure(fig, start, end, leadTime, cycle, arch_title, '', "plot_a_stdev_plot" + saveIdentifier, obsVsPred2,"html")
        print(f"Plot_Created_{obsVsPred2}_{leadTime}h_{architecture}_Cycle_{cycle}")    
# END: def decentralized_graphing_driver()

def standardDeviationFan_leadTime_plot(dfDict, leadTime, arch_title, cycle, obsVsPred, save, composite, uncertainty):
    
    """
    This function serves as the plotting function for a standard deviation fan.
    
    Inputs:
        dfDict: dictionary holding dataframes for plotting,
        leadTime: integer representing leadtime,
        arch_title: string for the save file naming convention,
        cycle: integer denoting cycle being plotted,
        obsVsPred: string to denote data being used,
        save: boolean to save or display

    Output:
        
        A fan plot
    """
    
    # Creation of Figure 
    fig = go.Figure()

    # Storage for traces for effective layering
    fan_traces = []
    mean_traces = []
    SD1_fan_traces = []

    # Sorts the Keys in an Effective Manner
    sorted_keys = sorted(dfDict.keys(), key=lambda x: ('mse' not in x, 'CRPS' not in x, 'PNN' not in x))
    print(f"Sorted Keys: {sorted_keys}")
    # Loops through the dictionary of information for plotting
    for key in sorted_keys:
        
        # Retrieves Dataframe
        df = dfDict[key]
        
        # Model Name Changes
        #model_name = key.split("-")[0] + "-MME"
        model_name = insert_mme(key)

        model_name = model_name.upper()
        
        # Hover Text Templates and Colors
        if 'CRPS' in key and composite == False:

            if uncertainty == True and 'MM-EX' in key:
                color = "#B504C8"#"#D8BFD8"  # Dark Orange for min_max_excluded

            elif uncertainty == True and 'ALL' in key:
                color = "#FF80AB"
            elif uncertainty == True and 'MM-IN' in key:
                color = "#4A148C"

            else:
                color = "#9C27B0"
            customda = df[['target', 'central_mae', 'mean_error', 'crps_gauss', 'init_time', 'P5', 'P95', "Median",  'BelowStd', 'AboveStd', 'Below2Std', 'Above2Std']]
            hovertemp = "<br>".join([
                "Valid date_time: %{x}",
                "Initialization time: %{customdata[4]}",
                'Lower Prediction Limit (P5): %{customdata[5]}',
                'Upper Prediction Limit (P95): %{customdata[6]}',
                '1SD Below: %{customdata[8]}',
                '1SD Above: %{customdata[9]}',
                '2SD Below: %{customdata[10]}',
                '2SD Above: %{customdata[11]}',
                f"Model: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                'Median Predicted Temperature (°C): %{customdata[7]}',
                "Actual temperature (°C): %{customdata[0]}",
                "CRPS (°C): %{customdata[3]}",
                "Central_MAE (°C): %{customdata[1]}",
                "ME (°C):%{customdata[2]}"
            ])

            # Hover Text Templates and Colors
        if 'PNN' in key and composite == False:

            if uncertainty == True and 'MM-EX' in key:
                color = "#84E291"

            elif uncertainty == True and 'ALL' in key:
                color = "#C0CA33"
            elif uncertainty == True and 'MM-IN' in key:
                color = "#1B5E20"
            else:
                color = "#66BB6A"  # Medium green with a vibrant, minty feel
                
            customda = df[['target', 'central_mae', 'mean_error', 'crps_gauss', 'init_time', 'P5', 'P95', "Median",  'BelowStd', 'AboveStd', 'Below2Std', 'Above2Std']]
            hovertemp = "<br>".join([
                "Valid date_time: %{x}",
                "Initialization time: %{customdata[4]}",
                'Lower Prediction Limit (P5): %{customdata[5]}',
                'Upper Prediction Limit (P95): %{customdata[6]}',
                '1SD Below: %{customdata[8]}',
                '1SD Above: %{customdata[9]}',
                '2SD Below: %{customdata[10]}',
                '2SD Above: %{customdata[11]}',
                f"Model: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                'Median Predicted Temperature (°C): %{customdata[7]}',
                "Actual temperature (°C): %{customdata[0]}",
                "CRPS (°C): %{customdata[3]}",
                "Central_MAE (°C): %{customdata[1]}",
                "ME (°C):%{customdata[2]}"
            ])

        elif 'CRPS' in key and composite == True:     
            print("Passed Here")
            if uncertainty == True and 'MM-EX' in key:
                color = "#B504C8"#"#D8BFD8"  # Dark Orange for min_max_excluded

            elif uncertainty == True and 'ALL' in key:
                color = "#FF80AB"
            elif uncertainty == True and 'MM-IN' in key:
                color = "#4A148C"

            else:
                color = "#9C27B0"
            customda = df[['target', 'central_mae', 'mean_error', 'crps_gauss', 'init_time', 'P5', 'P95', "Median",  'BelowStd', 'AboveStd', 'Below2Std', 'Above2Std', 'ssrat', 'ssrel', 'pit', 'date_time_range', 'init_time_range']]
            hovertemp = "<br>".join([
                'Valid date_time range: %{customdata[15]}',
                "Initialization time range: %{customdata[16]}",
                'Lower Prediction Limit (P5): %{customdata[5]}',
                'Upper Prediction Limit (P95): %{customdata[6]}',
                '1SD Below: %{customdata[8]}',
                '1SD Above: %{customdata[9]}',
                '2SD Below: %{customdata[10]}',
                '2SD Above: %{customdata[11]}',
                f"Model: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                'Median Predicted Temperature (°C): %{customdata[7]}',
                "Actual temperature (°C): %{customdata[0]}",
                "CRPS (°C): %{customdata[3]}",
                "Central_MAE (°C): %{customdata[1]}",
                "ME (°C):%{customdata[2]}",
                'SSRAT (°C): %{customdata[12]}',
                'SSREL (°C): %{customdata[13]}',
                'PIT: %{customdata[14]}'
            ])       
        elif 'PNN' in key:
            if uncertainty == True and 'MM-EX' in key:
                color = "#84E291"

            elif uncertainty == True and 'ALL' in key:
                color = "#C0CA33"
            elif uncertainty == True and 'MM-IN' in key:
                color = "#1B5E20"
            else:
                color = "#66BB6A"  # Medium green with a vibrant, minty feel
            customda = df[['target', 'central_mae', 'mean_error', 'crps_gauss', 'init_time', 'P5', 'P95', "Median",  'BelowStd', 'AboveStd', 'Below2Std', 'Above2Std', 'ssrat', 'ssrel', 'pit', 'date_time_range', 'init_time_range']]
            hovertemp = "<br>".join([
                'Valid date_time range: %{customdata[15]}',
                "Initialization time range: %{customdata[16]}",
                'Lower Prediction Limit (P5): %{customdata[5]}',
                'Upper Prediction Limit (P95): %{customdata[6]}',
                '1SD Below: %{customdata[8]}',
                '1SD Above: %{customdata[9]}',
                '2SD Below: %{customdata[10]}',
                '2SD Above: %{customdata[11]}',
                f"Model: {model_name}",
                "Mean Predicted Temperature (°C): %{y}",
                'Median Predicted Temperature (°C): %{customdata[7]}',
                "Actual temperature (°C): %{customdata[0]}",
                "CRPS (°C): %{customdata[3]}",
                "Central_MAE (°C): %{customdata[1]}",
                "ME (°C):%{customdata[2]}",
                'SSRAT (°C): %{customdata[12]}',
                'SSREL (°C): %{customdata[13]}',
                'PIT: %{customdata[14]}'
            ])    
            
        # Store fan traces
        fan_traces.append(go.Scatter(
            x=df.index,
            y=df['Below2Std'],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            legendgroup=model_name,
            opacity=0.3,
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
            opacity=0.3,
            connectgaps=False
        ))
        SD1_fan_traces.append(go.Scatter(
            x=df.index,
            y=df['BelowStd'],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            legendgroup=model_name,
            opacity=1,
            connectgaps=False
        ))
        SD1_fan_traces.append(go.Scatter(
            x=df.index,
            y=df['AboveStd'],
            mode='lines',
            line=dict(color=color, width=2),
            fill='tonexty',
            #fillcolor=fill_rgba,
            name=f"{model_name} ±1SD",
            showlegend=True,
            legendgroup=model_name,
            #legendrank=1,
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
            #legendrank=2,
            opacity=1,
            connectgaps=False
        ))


    # Code to add Traces to the Plot in an order to clearly see the differentiation
    for trace in fan_traces:
        fig.add_trace(trace)
    for trace in SD1_fan_traces:
        fig.add_trace(trace)
    for trace in mean_traces:
        fig.add_trace(trace)

    rep_df = next(iter(dfDict.values()))
    fig.add_trace(go.Scatter(x=rep_df.index, y=rep_df['target'].round(3), name="Observed Water Temperature", marker=dict(color='black'), mode='lines', showlegend=True,line=dict( width=3)))

    # Threshold Line
    fig.add_hline(y=8, line_dash="dot", line_color="red", annotation_text="Turtle Threshold", annotation_position="top left", annotation_font_size=26, annotation_font_color="red")

    # Labeling 
    title_text = f"Stdev_plot_{leadTime}h_Cycle_{cycle}_{obsVsPred}"
    save_path = f"{obsVsPred}_{arch_title}_{leadTime}h_Cycle_{cycle}"

    # Plot adjustments
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
            font=dict(size=20),
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
        return fig
    else:
        fig.show()

def mme_mse_crps_PNN_lead_times_singlePlot(architectures, cycles, leadTimes, obsVsPred, expanded, composite, uncertainty, start, end):
    
    """
        Process the hyperparameter combinations and metrics, and then for each cycle produce one
        boxplot that shows results for all lead times and for all architectures.
        
        Parameters:
          hyper_combos : list
              List of hyperparameter combinations.

          architectures : list
              List of model architectures.

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
            
            #MAIN_DIRECTORY = os.path.join(
                #r"C:\Users\woody\Work\UQ4ML_WaterTemp\src\TWC_results",
                #f"{architecture.lower()}_results")
            
            #MAIN_DIRECTORY = 'TWC_results/' + str(architecture.lower()) + "_results"
            start_dt = datetime.strptime(start, '%m/%d/%Y %H:%M')
            end_dt = datetime.strptime(end, '%m/%d/%Y %H:%M')

            # Format for filename or variable name
            results_path = f"TWC_results_{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}"

            MAIN_DIRECTORY = Path(results_path) / f"{architecture.lower()}_results"

            # For every lead time, grab data for each hyperparameter combo.
            for leadTime in leadTimes:
                
                # Calls helper function for conditional tree for models
                model_list = model_selection_conditional(leadTime, architecture)
                
                # Loop over each hyperparameter combo.
                for model in model_list:
                    uncertainty_collection = {}
                    if uncertainty == True:
                        uncertainty_list = ['Median', 'min', 'max', 1, 5, 25, 75, 95, 99]
                        df_list = []

                        for selection in uncertainty_list:
                            df = model_parser(MAIN_DIRECTORY, model, architecture, selection, 0, cycle, leadTime)
                            df_list.append(df)

                        # Concatenate horizontally
                        combined_df = pd.concat(df_list, axis=1)

                        combined_df = deduplicate_columns(combined_df)

                        uncertainty_collection['MM-IN'] = combined_df
                        
                        # Excluding Min Max
                        uncertainty_list = ['Median', 1, 5, 25, 75, 95, 99]
                        df_list = []

                        for selection in uncertainty_list:
                            df = model_parser(MAIN_DIRECTORY, model, architecture, selection, 0, cycle, leadTime)
                            df_list.append(df)

                        # Concatenate horizontally
                        combined_df2 = pd.concat(df_list, axis=1)

                        combined_df2 = deduplicate_columns(combined_df2)

                        uncertainty_collection['MM-EX'] = combined_df2

                        uncertainty_list = [f"member_{i}" for i in range(0, 100)]
                        df_list = []

                        for selection in uncertainty_list:
                            df = model_parser(MAIN_DIRECTORY, model, architecture, selection, 0, cycle, leadTime)
                            df_list.append(df)
                        
                        combined_df3 = pd.concat(df_list, axis=1)
                        combined_df3 = deduplicate_columns(combined_df3)
                        uncertainty_collection['ALL'] = combined_df3

                        uncertainty_list = ['pprog']
                        df_list = []

                        for selection in uncertainty_list:
                            df = model_parser(MAIN_DIRECTORY, model, architecture, selection, 0, cycle, leadTime)
                            df_list.append(df)
                        
                        combined_df4 = pd.concat(df_list, axis=1)
                        combined_df4 = deduplicate_columns(combined_df4)
                        uncertainty_collection['PP'] = combined_df4

                    else:
                        # Grab and process the data for this model, cycle, leadTime, and architecture.
                        df = model_parser(MAIN_DIRECTORY, model, architecture, obsVsPred, 0, cycle, leadTime)
                        uncertainty_collection['Median'] = df
                    # Remove any duplicate indices. # Unnecessary
                    df = df[~df.index.duplicated(keep='first')]
                    
                    for key, df in uncertainty_collection.items():
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

                        if composite == True:
                            identifier = "TWC_composite_UQ_Files"
                        else:
                            identifier = "TWC_hourly_UQ_Files"
                        # To ensure cross compatability
                        base_dir = Path("src") / results_path  / identifier

                        # Create the directories if they do not exist
                        base_dir.mkdir(parents=True, exist_ok=True)

                        # Now define the output path
                        output_path = base_dir / f"{key}_{leadTime}h_{architecture}_Cycle_{cycle}_Model_{model}.csv"
                        modDf2.to_csv(output_path)

def create_composite_metrics(df, block_size):

    """
    Splits the DataFrame into blocks of `block_size` rows and computes metrics on each block.
    
    Parameters:
    - df: original hourly DataFrame
    - block_size: number of rows per composite block
    - metric_columns: list of columns to compute metrics on
    
    Returns:
    - composite_df: DataFrame with one row per block and computed metrics
    """
    composite_rows = []

    print(df.columns)

    # Total number of blocks
    num_blocks = (len(df) + block_size - 1) // block_size  # rounds up

    for i in range(num_blocks):
        block = df.iloc[i * block_size : (i + 1) * block_size]

        print(block.head(block_size))

        print(block.columns)

        # Compute metrics for the block
        metrics = {
            "date_time": block["date_time"].iloc[-1],
            'date_time_range': (block["date_time"].iloc[0].strftime('%m/%d/%Y %H:%M'), block["date_time"].iloc[-1].strftime('%m/%d/%Y %H:%M')),
            "target": block["target"].iloc[-1],
            "init_time": block["init_time"].iloc[-1],
            'init_time_range': (block["init_time"].iloc[0], block["init_time"].iloc[-1]),
            'Median': block['Median'].iloc[-1],
            'Mean': block['Mean'].iloc[-1],
            'Stdev': block['Stdev'].iloc[-1],
            '2Stdev': block['2Stdev'].iloc[-1],
            'BelowStd': block['BelowStd'].iloc[-1],
            'AboveStd': block['AboveStd'].iloc[-1],
            'Below2Std': block['Below2Std'].iloc[-1],
            'Above2Std': block['Above2Std'].iloc[-1],
            'P5': block['P5'].iloc[-1],
            'P95': block['P95'].iloc[-1]
        }

        actualShaped = block['target'].values.astype(float)
        averageReshaped = block["Mean"].values.astype(float)
        stdReshaped = block["Stdev"].values.astype(float)

        # Reshaped to Tensors
        actualReshaped = block['target'].values.astype(float).reshape(-1, 1)
        averageShapedTens = averageReshaped.reshape(-1, 1)

        # Reshapes for CRPS Calculations
        actualReshapedTens = block['target'].values.astype(np.float32).reshape(-1, 1)
        averageReshapedTens = block["Mean"].values.astype(np.float32).reshape(-1, 1)
        stdReshapedTens = block["Stdev"].values.astype(np.float32).reshape(-1, 1)

        # Evaluation Calculations
        pitAverageCalc = get_pit_points(actualShaped, averageReshaped, stdReshaped).round(3)
        print("Pit Calculated")
        
        ssrelCalcAVG = get_spread_skill_points(actualShaped, averageReshaped, stdReshaped).round(3)
        print("SSREL Calculated")
        
        crpsCalc_gauss = crps_gaussian_tf(averageReshapedTens, stdReshapedTens, actualReshapedTens).numpy().round(3)
        print("CRPS Calculated")
        
        ssratAverage = ssrat_avg(actualShaped, averageReshaped, stdReshaped).round(3)
        print("SSRAT Calculated")
        
        meCalc = me(actualReshaped, averageShapedTens).round(3)
        print("ME Calculated")

        maeCalc = mae(actualReshaped, averageShapedTens).round(3)
        print('MAE calculated')

        # Calculations for metrics
        metrics["crps_gauss"] = crpsCalc_gauss
        metrics["mean_error"] = meCalc
        metrics["central_mae"] = maeCalc
        metrics["pit"] = pitAverageCalc
        metrics["ssrat"] = ssratAverage
        metrics['ssrel'] = ssrelCalcAVG

        composite_rows.append(metrics)

    composite_df = pd.DataFrame(composite_rows)

    return composite_df
# END: def create_composite_metrics()

def detect_low_temperature_periods(df, threshold=8, min_duration=24):
    """
    Detects periods where 'target' stays below the threshold for at least min_duration hours.
    Assumes 'date_time' column is datetime and rows are spaced regularly (e.g., every 6 hours).
    Returns a list of (start_time, end_time) datetime objects.
    """
    low_periods = []

    # Ensure datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Calculate time delta between rows (assumes regular spacing)
    time_deltas = df['date_time'].diff().dropna()
    avg_delta_hours = time_deltas.dt.total_seconds().mean() / 3600

    if avg_delta_hours == 0:
        raise ValueError("Time intervals between rows are zero or inconsistent.")

    # Convert min_duration (in hours) to row count
    required_rows = int(min_duration / avg_delta_hours)

    # Create mask and group consecutive low values
    df['below'] = (df['target'] < threshold).astype(int)
    df['group'] = (df['below'] != df['below'].shift()).cumsum()

    for _, group_df in df.groupby('group'):
        if group_df['below'].iloc[0] == 1 and len(group_df) >= required_rows:
            start_time = group_df['date_time'].iloc[0]
            end_time = group_df['date_time'].iloc[-1]
            low_periods.append((start_time, end_time))

    df.drop(columns=['below', 'group'], inplace=True)
    return low_periods
# END: def detect_low_temperature_periods()

def heatmap_plot_b(df, leadTime, cycle, architecture, metric, start, end, plot_type, compoiste, obsVsPred):

    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = df.pivot(index='init_time', columns='date_time', values=metric)

    # Detect low-temperature periods
    low_periods = detect_low_temperature_periods(df)

    # Convert columns to datetime
    heatmap_data.columns = pd.to_datetime(heatmap_data.columns)

    if metric == "crps_gauss":
        title = "CRPS"
    elif metric == "mean_error":
        title = "ME"
    elif metric == "central_mae":
        title = "MAE"
    elif metric == "pit":
        title = "PIT"
    elif metric == "ssrat":
        title = "SSRAT"
    elif metric == "ssrel":
        title = "SSREL"


    if composite:

        # Map init_time to init_range directly
        init_range_map = df.set_index('init_time')['init_time_range']

        # Map valid time to validrange directly
        valid_range_map = df.set_index('date_time')['date_time_range']

        hovertext = [
            [
                f"Init: {y}<br>Valid: {x}<br>{title}: {z:.3f}<br>Init Range: {init_range_map[y]} <br>Valid Range: {valid_range_map[x]}"
                for x, z in zip(heatmap_data.columns, row)
            ]
            for y, row in zip(heatmap_data.index, heatmap_data.values)
        ]
    else:

        # Customizes hover text labels
        hovertext = [
            [
                f"Init: {y}<br>Valid: {x}<br>{title}: {z:.3f}"
                for x, z in zip(heatmap_data.columns, row)
            ]
            for y, row in zip(heatmap_data.index, heatmap_data.values)
        ]

    # Create heatmap using go.Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        text=hovertext,
        hoverinfo="text", 
        colorscale='Viridis',
        colorbar=dict(title=title
    )))

    if low_periods: 
         # Add vertical lines for each low-temperature period
       
            for i, (low_start, low_end) in enumerate(low_periods):
                fig.add_shape(
                    type="line",
                    x0=low_start,
                    x1=low_start,
                    y0=heatmap_data.index.min(),
                    y1=heatmap_data.index.max(),
                    line=dict(color="red", width=2),
                    name=f"Start Low {i+1}"
                )
                fig.add_shape(
                    type="line",
                    x0=low_end,
                    x1=low_end,
                    y0=heatmap_data.index.min(),
                    y1=heatmap_data.index.max(),
                    line=dict(color="red", width=2),
                    name=f"End Low {i+1}"
                )


    # Assuming leadtime and model are defined variables
    full_title = f"{title} Performance Heatmap\nInitialization Vs. Valid Time\nModel: {architecture} | Lead Time: {leadTime} | {obsVsPred}"

    fig.update_layout(
        title=full_title,
        xaxis_title="Valid Time",
        yaxis_title="Initialization Time"
    )

    save_figure(fig, start, end, leadTime, cycle, architecture, title, plot_type, obsVsPred)

    #fig.show()
# END: def heatmap_plot_b()

def line_plot_b(df_dict, leadTimes, cycle, metric, start, end, plot_type, composite, obsVsPred):
    """
    Creates a line plot for the specified metric over time.
    
    Parameters:
    - df: DataFrame containing the data to plot.
    - leadTime: Lead time for the plot title.
    - cycle: Cycle number for the plot title.
    - architecture: Model architecture for the plot title.
    - metric: Metric to plot.
    - start: Start datetime for the plot.
    - end: End datetime for the plot.
    - plot_type: Type of plot (e.g., "line_plot_b").
    
    Returns:
    - None
    """
    fig = go.Figure()

    # Name conditional based on metric
    if metric == "crps_gauss":
        title = "CRPS"
    elif metric == "mean_error":
        title = "ME"
    elif metric == "central_mae":
        title = "MAE"
    elif metric == "pit":
        title = "PIT"
    elif metric == "ssrat":
        title = "SSRAT"
    elif metric == "ssrel":
        title = "SSREL"

    seen_architectures = []
    # Loop through each DataFrame in the list
    for key, df in df_dict.items():

        architecture = key.split("_")[1]
        combined = architecture
        if architecture not in seen_architectures:
            seen_architectures.append(architecture)
            if len(seen_architectures) == 2:
                combined = "-".join(seen_architectures)
                print(f"New combo detected: {combined}")

        
        # Extract model name and leadTime from df metadata if available
        # Otherwise, use placeholder names

        if 'MM-EX' in key:

            dashType = "dash"
        elif 'ALL' in key:
            dashType = "dot"
        elif 'MM-IN' in key:
            dashType = "solid"
        else:
            dashType = "dashdot"

        if composite:
            customda = df[['target', 'Mean', 'init_time', 'Stdev', '2Stdev', 'date_time_range', 'init_time_range']]
            hovertemp = "<br>".join([
                "valid_time_range: %{customdata[5]}",
                "init_time_range: %{customdata[6]}",
                f"Model: {key}",
                "Mean Predicted Temperature (°C): %{customdata[1]}",
                "Actual temperature (°C): %{customdata[0]}",
                '1SD (°C): %{customdata[3]}',
                '2SD (°C): %{customdata[4]}',
                f"{title} (°C):" + "%{y}",
            ])    

        else:
            customda = df[['target', 'Mean', 'init_time', 'Stdev', '2Stdev']]
            hovertemp = "<br>".join([
                "date_time: %{x}",
                "init_time: %{customdata[2]}",
                f"Model: {key}",
                "Mean Predicted Temperature (°C): %{customdata[1]}",
                "Actual temperature (°C): %{customdata[0]}",
                '1SD (°C): %{customdata[3]}',
                '2SD (°C): %{customdata[4]}',
                f"{title} (°C):" + "%{y}",
            ])

        leadTime_id = key.split("_")[0]  # Extracts "12h"
        leadTime_int = int(leadTime_id.replace("h", ""))  # Converts to 12

        # Add line to plot
        fig.add_trace(go.Scatter(
            x=df["date_time"],
            y=df[metric],
            mode='markers+lines',
            #marker=dict(symbol='circle', size=10),
            name=key,
            customdata=customda,
            hovertemplate=hovertemp,
            line=dict(dash = dashType, color=get_color(architecture, leadTime_int))

        
        ))

        # Optional: highlight low-temperature periods
        low_periods = detect_low_temperature_periods(df)
        # You can annotate or shade these if needed

        if low_periods:
            for i, (low_start, low_end) in enumerate(low_periods):
                # Start line
                fig.add_shape(
                    type="line",
                    x0=low_start,
                    x1=low_start,
                    y0=df[metric].min(),
                    y1=df[metric].max(),
                    line=dict(color="red", width=2, dash="dot"),
                    name=f"Start Low {i+1}"
                )

                # End line
                fig.add_shape(
                    type="line",
                    x0=low_end,
                    x1=low_end,
                    y0=df[metric].min(),
                    y1=df[metric].max(),
                    line=dict(color="orange", width=2, dash="dot"),
                    name=f"End Low {i+1}"
                )


    # Layout customization
    fig.update_layout(
        title=f"Line_Plot — {title} — {combined} Cycle — ({cycle}) | {obsVsPred}",
        xaxis_title="Valid Time",
        yaxis_title=title + " (°C)",
        legend_title="Model & Lead Time",
        template="plotly_white"
    )

    lead_time_str = "_".join(str(lt) for lt in leadTimes)

    save_figure(fig, start, end, lead_time_str, cycle, combined, title, plot_type, obsVsPred)
    #fig.show()
# END: def line_plot_b()

def plot_pairwise_differences_by_leadtime(df_dict, leadTimes, cycle, metric, start, end, plot_type, composite, name):
    """
    Creates a Plotly line plot showing pairwise differences between obsVsPred scenarios
    for each architecture and lead time, with composite-aware hover and dash styling.

    Parameters:
    - df_dict: Dictionary of DataFrames keyed by model identifiers.
    - leadTimes: List of lead times to include in the plot.
    - cycle: Cycle number for the plot title.
    - metric: Metric column to compute differences on.
    - start, end: Start and end datetime for the plot.
    - plot_type: Type of plot (e.g., "pairwise_diff_plot").
    - composite: Whether to use composite hover data.
    - name: Scenario label for the plot title.

    Returns:
    - None
    """
    from collections import defaultdict
    fig = go.Figure()
    
    # Creates a container list to hold DataFrames for export
    export_by_lead = defaultdict(list)

    # Should be a function but its fine
    title_map = {
        "crps_gauss": "CRPS",
        "mean_error": "ME",
        "central_mae": "MAE",
        "pit": "PIT",
        "ssrat": "SSRAT",
        "ssrel": "SSREL"
    }

    # Uppercases metric for title
    title = title_map.get(metric, metric.upper())

    # Hardcoded obsVsPred scenarios and their pairs due to pairs but also because these are what we want to see now
    obsVsPred_list = ['MM-EX', 'MM-IN', 'ALL']
    pairs = [
        ('MM-EX', 'ALL'),
        ('MM-IN', 'ALL'),
        ('MM-EX', 'MM-IN')
    ]

    # Group DataFrames by (leadTime, architecture, scenario)
    grouped = defaultdict(lambda: defaultdict(dict))
    for key, df in df_dict.items():
        parts = key.split("_")
        if len(parts) < 3:
            continue
        lead_time = parts[0].replace("h", "")
        arch = parts[1]
        scenario = "_".join(parts[2:])
        grouped[lead_time][arch][scenario] = df

    # Plot pairwise differences
    for lead_time in leadTimes:
        lead_key = str(lead_time)

        for arch, scenario_dict in grouped[lead_key].items():
            for scenario_a, scenario_b in pairs:
                if scenario_a in scenario_dict and scenario_b in scenario_dict:
                    df_a = scenario_dict[scenario_a]
                    df_b = scenario_dict[scenario_b]

                    df_diff = df_a[['date_time']].copy()
                    df_diff[metric] = df_a[metric] - df_b[metric]

                    if composite:
                        index_cols = ['date_time_range', 'init_time_range']
                        shared_index = df_a[index_cols]
                    else:
                        index_cols = ['date_time', 'init_time']
                        shared_index = df_a[index_cols]
                    df_export = pd.DataFrame({
                        f"({scenario_a})-({scenario_b})_{arch}": df_a[metric] - df_b[metric],
                        f"{scenario_a}_{arch}": df_a[metric],
                        f"{scenario_b}_{arch}": df_b[metric],
                        "target": df_a["target"]
                    })

                    # Assign index after data is populated
                    df_export[index_cols] = shared_index
                    df_export.set_index(index_cols, inplace=True)

                    export_by_lead[lead_key].append(df_export)

                    # Dash type logic per pair
                    pair_key = f"{scenario_a}__{scenario_b}"

                    # Should be a function but its fine
                    dash_map = {
                        "MM-EX__ALL": "longdash",
                        "MM-IN__ALL": "dot",
                        "MM-EX__MM-IN": "longdashdot",
                        "ALL__MM-EX": "longdash",
                        "ALL__MM-IN": "dot",
                        "MM-IN__MM-EX": "longdashdot"
                    }
                    # Gets the dash type
                    dashType = dash_map.get(pair_key, "longdashdot")
                    # Composite-aware hover
                    if composite:

                        df_diff['customdata'] = list(zip(
                            df_a['target'],
                            df_a['date_time_range'],
                            df_a['init_time_range'],
                            df_a[metric],
                            df_b[metric]
                        ))
                        customdata=df_diff['customdata']
                        hovertemp = "<br>".join([
                            "valid_time_range: %{customdata[1]}",
                            "init_time_range: %{customdata[2]}",
                            f"Diff: {scenario_a} - {scenario_b}",
                            f"{scenario_a} {title} (°C): " + "%{customdata[3]}",
                            f"{scenario_b} {title} (°C): " + "%{customdata[4]}",
                            f"{title} (°C): " + "%{y}",
                            "Actual temperature (°C): %{customdata[0]}",
                        ])
                    else:
                        df_diff['customdata'] = list(zip(
                            df_a['target'],
                            df_a['init_time'],
                            df_a[metric],
                            df_b[metric]
                        ))
                        customdata=df_diff['customdata']
                        hovertemp = "<br>".join([
                            "valid_time: %{x}",
                            "init_time: %{customdata[1]}",
                            f"Diff: {scenario_a} - {scenario_b}",
                            f"{scenario_a} {title} (°C): " + "%{customdata[2]}",
                            f"{scenario_b} {title} (°C): " + "%{customdata[3]}",
                            f"{title} (°C): " + "%{y}",
                            "Actual temperature (°C): %{customdata[0]}"
                        ])


                    fig.add_trace(go.Scatter(
                        x=df_diff["date_time"],
                        y=df_diff[metric],
                        mode='lines',
                        name=f"{lead_key}h | {arch}: {scenario_a} - {scenario_b}",
                        customdata=customdata,
                        hovertemplate=hovertemp,
                        line=dict(dash=dashType, color=get_color(arch, int(lead_key)))
                    ))

                    # Detects the beginning and end of cold stunning events
                    low_periods = detect_low_temperature_periods(df_a)

                    # Code to place lines at beginning and and of cold stunnning event
                    if low_periods:
                        for i, (low_start, low_end) in enumerate(low_periods):
                            # Start line
                            fig.add_shape(
                                type="line",
                                x0=low_start,
                                x1=low_start,
                                y0=df_diff[metric].min(),
                                y1=df_diff[metric].max(),
                                line=dict(color="red", width=2, dash="dot"),
                                name=f"Start Low {i+1}"
                            )

                            # End line
                            fig.add_shape(
                                type="line",
                                x0=low_end,
                                x1=low_end,
                                y0=df_diff[metric].min(),
                                y1=df_diff[metric].max(),
                                line=dict(color="orange", width=2, dash="dot"),
                                name=f"End Low {i+1}"
                            )

    # Integral for saving and view of plots
    combined_archs = "-".join(sorted({arch for lt in leadTimes if str(lt) in grouped for arch in grouped[str(lt)]}))
    lead_time_str = "_".join(str(lt) for lt in leadTimes)

    fig.update_layout(
        title=f"Pairwise Differences — {title} — {combined_archs} Cycle — ({cycle}) | {name}",
        xaxis_title="Valid Time",
        yaxis_title=f"{title} (°C)",
        legend_title="Lead Time | Architecture | Scenario Pair",
        template="plotly_white"
    )
    save_figure(fig, start, end, lead_time_str, cycle, combined_archs, title, plot_type, name)

    # Everything below here is for outputting and storing csv files for the differences
    if composite == True:
        string = "_composite"
    else:
        string = "_hourly"
    
    for lead_key, df_list in export_by_lead.items():
        df_combined = pd.concat(df_list, axis=1)

        # Remove duplicate columns
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

        # Reset index so time info appears in CSV
        df_combined.reset_index(inplace=True)

        save_figure(df_combined, start, end, lead_key, cycle, combined_archs, title, "difference_table_c_differences" + string, name, "csv")


def plot_architecture_differences_by_leadtime(df_dict, leadTimes, cycle, metric, start, end, plot_type, composite, name):
    from collections import defaultdict
    fig = go.Figure()
    all_shapes = []

    # Can probably be placed in a function but for now its fine
    title_map = {
        "crps_gauss": "CRPS",
        "mean_error": "ME",
        "central_mae": "MAE",
        "pit": "PIT",
        "ssrat": "SSRAT",
        "ssrel": "SSREL"
    }

    # Capiotalizes metric 
    title = title_map.get(metric, metric.upper())

    # Hardcoded because we only want these three scenarios
    obsVsPred_list = ['MM-EX', 'MM-IN', 'ALL']

    grouped = defaultdict(lambda: defaultdict(dict))
    for key, df in df_dict.items():
        parts = key.split("_")
        if len(parts) < 3:
            continue
        lead_time = parts[0].replace("h", "")
        arch = parts[1]
        scenario = "_".join(parts[2:])
        grouped[lead_time][arch][scenario] = df

    for scenario_fixed in obsVsPred_list:
        for lead_time in leadTimes:
            lead_key = str(lead_time)
            if lead_key not in grouped:
                continue

            archs = list(grouped[lead_key].keys())
            for i in range(len(archs)):
                for j in range(i + 1, len(archs)):
                    arch_a, arch_b = archs[i], archs[j]
                    scenario_dict_a = grouped[lead_key][arch_a]
                    scenario_dict_b = grouped[lead_key][arch_b]

                    if scenario_fixed in scenario_dict_a and scenario_fixed in scenario_dict_b:
                        df_a = scenario_dict_a[scenario_fixed]
                        df_b = scenario_dict_b[scenario_fixed]

                        df_diff = df_a[['date_time']].copy()
                        df_diff[metric] = df_a[metric] - df_b[metric]

                        key = f"{arch_a}_{arch_b}_{scenario_fixed}"
                        dashType = "dash" if 'MM-EX' in key else "dot" if 'ALL' in key else "solid"

                        if composite:
                            df_diff['customdata'] = list(zip(
                                df_a['target'],
                                df_a['date_time_range'],
                                df_a['init_time_range'],
                                df_a[metric],
                                df_b[metric]
                            ))
                            customdata = df_diff['customdata']
                            hovertemp = "<br>".join([
                                "valid_time_range: %{customdata[1]}",
                                "init_time_range: %{customdata[2]}",
                                f"Diff: {arch_a} - {arch_b}",
                                f"{arch_a} {title} (°C): " + "%{customdata[3]}",
                                f"{arch_b} {title} (°C): " + "%{customdata[4]}",
                                f"{title} (°C): " + "%{y}",
                                "Actual temperature (°C): %{customdata[0]}",
                            ])
                        else:
                            df_diff['customdata'] = list(zip(
                                df_a['target'],
                                df_a['init_time'],
                                df_a[metric],
                                df_b[metric]
                            ))
                            customdata = df_diff['customdata']
                            hovertemp = "<br>".join([
                                "valid_time: %{x}",
                                "init_time: %{customdata[1]}",
                                f"Diff: {arch_a} - {arch_b}",
                                f"{arch_a} {title} (°C): " + "%{customdata[2]}",
                                f"{arch_b} {title} (°C): " + "%{customdata[3]}",
                                f"{title} (°C): " + "%{y}",
                                "Actual temperature (°C): %{customdata[0]}"
                            ])

                        leadCleaned = lead_key.replace("h", "")
                        fig.add_trace(go.Scatter(
                            x=df_diff["date_time"],
                            y=df_diff[metric],
                            mode='lines',
                            name=f"{lead_key}h | {arch_a} - {arch_b}: {scenario_fixed}",
                            customdata=customdata,
                            hovertemplate=hovertemp,
                            line=dict(dash=dashType, color=get_color("", int(leadCleaned)))
                        ))

                        # Detect and annotate low-temperature periods
                        low_periods = detect_low_temperature_periods(df_a)

                        # Code responsible for placing lines at the beginning and end of the cold stunning events
                        if low_periods:
                            for i, (low_start, low_end) in enumerate(low_periods):
                                all_shapes.append(dict(
                                    type="line",
                                    x0=low_start,
                                    x1=low_start,
                                    y0=df_diff[metric].min(),
                                    y1=df_diff[metric].max(),
                                    xref="x",
                                    yref="y",
                                    line=dict(color="red", width=2, dash="dot")
                                ))
                                all_shapes.append(dict(
                                    type="line",
                                    x0=low_end,
                                    x1=low_end,
                                    y0=df_diff[metric].min(),
                                    y1=df_diff[metric].max(),
                                    xref="x",
                                    yref="y",
                                    line=dict(color="orange", width=2, dash="dot")
                                ))
    # Naming for saving purposes
    arch_combo = "-".join(sorted({arch for lt in leadTimes if str(lt) in grouped for arch in grouped[str(lt)]}))
    lead_time_str = "_".join(str(lt) for lt in leadTimes)

    # This ugly blob of code is for setting the y-axis range with padding
    # This is needed for the inserting of background shading
    y_vals = []
    for trace in fig.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_vals.extend(trace.y)

    y_min = min(y_vals)
    y_max = max(y_vals)
    padding = 0.05 * (y_max - y_min)
    y_min_padded = y_min - padding
    y_max_padded = y_max + padding

    # Add background shading (i.e., green for negative, purple for positive)
    all_shapes += [
        dict(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=y_min_padded,
            y1=0,
            fillcolor="rgba(200, 255, 200, 0.3)",
            line=dict(width=0),
            layer="below"
        ),
        dict(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=0,
            y1=y_max_padded,
            fillcolor="rgba(230, 200, 255, 0.3)",
            line=dict(width=0),
            layer="below"
        )
    ]

    fig.update_layout(
        title=f"Architecture Differences — {title} — {arch_combo} Cycle — ({cycle}) | All Scenarios",
        xaxis_title="Valid Time",
        yaxis_title=f"{title} (°C)",
        legend_title="Lead Time | Architecture Pair | Scenario",
        template="plotly_white",
        shapes=all_shapes
    )

    save_figure(fig, start, end, lead_time_str, cycle, arch_combo, title, plot_type, name)
        
def save_figure(fig, start, end, leadTime, cycle, architecture, metric, plot_type, obsVsPred, filetype="html"):
    
    # Ensure start and end are datetime objects
    start = pd.to_datetime(start, format='%m/%d/%Y %H:%M')
    end = pd.to_datetime(end, format='%m/%d/%Y %H:%M')

    # Convert and format timestamps
    time_folder = f"{start.strftime('%Y%m%d_%H%M')}_to_{end.strftime('%Y%m%d_%H%M')}"

    # Create folder path
    base_dir = Path("TWC_Experiments") / time_folder / plot_type 
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    if metric == "":
        filename = f"{obsVsPred}_{leadTime}h_Cycle_{cycle}_Model_{architecture}.{filetype}"
    else:
        filename = f"{obsVsPred}_{metric}_{leadTime}h_Cycle_{cycle}_Model_{architecture}.{filetype}"
    full_path = base_dir / filename

    if filetype == "html":
        fig.write_html(full_path)
        print(f"Figure saved to: {full_path}")
    elif filetype == "png":
        fig.write_image(full_path, width=1200, height=800)
        print(f"Image saved to: {full_path}")
    elif filetype == "csv":
        if isinstance(fig, pd.DataFrame):

            full_path = base_dir / f"{obsVsPred}_{metric}_{leadTime}h_Cycle_{cycle}.{filetype}"
            fig.to_csv(full_path, index=False)
            print(f"CSV saved to: {full_path}")
        else:
            print("Error: 'fig' must be a pandas DataFrame when saving as CSV.")
    else:
        print(f"Unsupported filetype: {filetype}")

    print(f"Figure saved to: {full_path}")
# END: def save_figure()

def get_color(architecture, leadTime):

    """
    Returns a color based on the architecture and lead time.
    Uses distinct color palettes for different architectures.
    """

    purple_shades = {
    12:  "#D8BFD8",  # Thistle (lightest)
    48:  "#9370DB",  # Medium purple
    96:  "#800080",  # Purple
    120: "#4B0082"   # Indigo (darkest)
    }

    blue_shades = {
    12:  "#ADD8E6",  # Light Blue
    48:  "#6495ED",  # Cornflower Blue
    96:  "#0000FF",  # Blue
    120: "#00008B"   # Dark Blue
    }
    green_shades = {
    12:  "#98FB98",  # Pale Green
    48:  "#3CB371",  # Medium Sea Green
    96:  "#008000",  # Green
    120: "#006400"   # Dark Green
    }

    if architecture == "CRPS":
        return purple_shades.get(leadTime, "#CCCCCC")  # default gray if not found

    elif architecture == "PNN":
        return green_shades.get(leadTime, "#CCCCCC")  # default gray if not found
    else:
        return blue_shades.get(leadTime, "#CCCCCC")  # default blue for other architecture
 #END: def get_color()

def deduplicate_columns(df):
    # Drop duplicate column names (keep first)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Drop columns with identical content
    df = df.loc[:, ~df.T.duplicated(keep='first')]

    return df
# END: def deduplicate_columns()

def insert_mme(key):
    """
    Horrendous evil regex function to insert MME into the key string
    for proper labeling of multi-model ensemble data.
    """
    import re

    return re.sub(r"^(.+?)(-\d+h_)", r"\1-MME\2", key)
# END: def insert_mme()

def runner_all_lines_heats_stdevs(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred_list, uncertainty):
    if composite == True:
        
        if uncertainty == True:

            runner_uncertainty(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred_list)
        else:

            # This is for if you want to run some plots that dont require grouping needed for uncertainty
            for obsVsPred in obsVsPred_list:
                runner(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred)

        # Creates standard deviation plots for each leadtime
        for leadTime in leadTimes:
            decentralized_graphing_driver(models, leadTime, [cycle], obsVsPred, save, composite, start, end, uncertainty)

        # Runs aggregate table code 
        aggregateTableOpsTest(leadTimes, [cycle], models,12 , True, obsVsPred_list, start, end,  composite, padding = 24)

    else:

        # Creates files needed for calculations
        for obsVsPred in obsVsPred_list: 
            mme_mse_crps_PNN_lead_times_singlePlot(models, [cycle], leadTimes, obsVsPred, expanded, composite, uncertainty, start, end)

            # Code for the runner function that creates line plots and heatmaps when not looking at uncertainty
            if uncertainty == False:
                runner(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred)

            # Runs code for creating standard deviation plots for each leadtime
            for leadTime in leadTimes:
                decentralized_graphing_driver(models, leadTime, [cycle], obsVsPred, save, composite, start, end, uncertainty)

        # Aggregate table code
        aggregateTableOpsTest(leadTimes, [cycle], models, 12, True, obsVsPred_list, start, end,  composite, padding = 24)

        # Runs uncertainty code if specified, will output a suite of plots
        if uncertainty == True:

            runner_uncertainty(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred_list)
# END: def runner_all_lines_heats_stdevs()

runner_all_lines_heats_stdevs(leadTimes, cycle, models, start, end, block_size, composite, obsVsPred_list, uncertainty)