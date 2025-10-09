"""
This file is designed to test our models using the TWC air tempeature data within our models.
"""
import sys
from pathlib import Path

# Add project root (parent of src) to sys.path BEFORE any src import
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Now do all imports
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import re
from datetime import timedelta

from tensorflow.keras.models import load_model

# Then your custom module import should work
from src.helper.utils_mse_crps import creatingAdditionalColumns, dateTimeRetriever, crps_loss, crps


import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# important variables
# TWC_VARIABLE = r"C:\Users\woody\Work\twc-sbirdisland\IBM"
TWC_VARIABLE = r"C:\Users\hmarrero\Downloads\cool-turtle-data-drop\twc-sbirdisland\IBM"  

# NDFD_VARIABLE = r"C:\Users\woody\Work\ndfd-sbirdisland\NDFD"
NDFD_VARIABLE = r"C:\Users\hmarrero\Downloads\cool-turtle-data-drop\ndfd-sbirdisland\NDFD"

# repo_project_path = r"C:\Users\woody\Work\UQ4ML_WaterTemp"
repo_project_path = r"C:\Users\hmarrero\Documents\GitHub_Repos\UQ4ML_WaterTemp\UQ4ML_WaterTemp"

# model = 'CRPS'#'CRPS'
model = 'PNN'#'CRPS'
start = '01/16/2024 06:00'
end = '01/21/2024 15:00'
# leadTimes = [12, 48, 96, 120]
leadTimes = [12, 48]
padding = 48
# was hard coded to descending causing a 62 vs 60 descrepency in the PNN runs
ascend_or_descend = 'descending' #'ascending' #descending

verbose = 1 #0

uncertainty_test_list = ['Median', 'min','max', 1, 5, 25, 75, 95, 99]
start_iteration = 1
# end_iteration = 10
end_iteration = 2
cycle = 8 #Purely for naming purposes, it just denotes the cycle the model was trained on.
TWC = True
# allModels = True
allModels = False

# --------Helper Functions----------


# NEED to use the itensive way of testing every single model :(
# Weather Company Data Specific Function
def dataframe_retriever(dt, percentile, origin=TWC_VARIABLE):

    # Offset timestamp
    dt_offset = dt + timedelta(minutes=10)
    
    if verbose >= 2:
        print(f"Init Time: {dt_offset}")

    # Time components
    year = dt_offset.strftime("%Y")
    month = dt_offset.strftime("%m")
    day = dt_offset.strftime("%d")
    time = dt_offset.strftime("%H%M")

    # File path construction
    fName = os.path.join(origin, 'sbirdisland', year, f"ibm-response.{year}{month}{day}-{time}.json")

    # Load JSON
    with open(fName, "r") as data:
        dataretrieve = json.load(data)

    fcst = dataretrieve['forecasts1Hour']
    valid_times = fcst['fcstValid']
    ensemble_data_by_member = fcst['prototypes'][0]['forecast']  # shape: [100][241]

    # Convert forecast times
    forecast_date_times = [utcDateTimeConverter(valid_times[i])[0] for i in range(241)]

    # Initialize dataframe and insert timestamps first
    df = pd.DataFrame({'forecast_date_time': forecast_date_times})

    # Now add ensemble data column-wise
    for j in range(100):
        df[f'member_{j}'] = [far_to_cel(val) for val in ensemble_data_by_member[j]]

    # Drop first row (current obs), reindex
    df = df.iloc[1:].copy()
    df.set_index('forecast_date_time', inplace=True)

    if percentile == "Median":
        if verbose >= 2:
            print("Median Running")
        df['Median'] = df.median(axis=1)
    elif percentile == "max":
        if verbose >= 2:
            print('Max running')
        df['max'] = df.max(axis=1)
    elif percentile == "min":
        if verbose >= 2:
            print('Min running')
        df['min'] = df.min(axis=1)
    elif isinstance(percentile, int) and 1 <= percentile <= 99:
        if verbose >= 2:
            print("Percentiles running")
        q = percentile / 100
        df[percentile] = round(df.quantile(q, axis=1), 2)

    df['RowIndex'] = range(1, len(df) + 1)

    # Optional save
    #df.to_csv('testAirTWC.csv', index=True)

    return df, fName, year
#END: dataframe_retriever()

def dataframe_retriever_NDFD(dt, origin=NDFD_VARIABLE):
    
    # Offset timestamp
    dt_offset = dt + timedelta(minutes=10)
    if verbose >= 2:
        print(f"Init Time: {dt_offset}")

    # Time components
    year = dt_offset.strftime("%Y")
    month = dt_offset.strftime("%m")
    day = dt_offset.strftime("%d")
    time = dt_offset.strftime("%H%M")

    # File path construction
    fName = os.path.join(origin, 'sbirdisland', year, f"ndfd-predictions.{year}{month}{day}-{time}.json")

    if verbose >= 2:
        print(fName)
    # Load JSON
    with open(fName, "r") as data:
        dataretrieve = json.load(data)

    # Create DataFrame
    df = pd.DataFrame(dataretrieve, columns=['timestamp_ms', 'Median'])

    if verbose >= 2:
        print(df['timestamp_ms'])

    if verbose >= 2:
        print(df.head())

    def convert_epoch_ms_to_datetime(epoch_ms):
        """Convert epoch time in milliseconds to a readable naive UTC datetime."""
        return datetime.utcfromtimestamp(epoch_ms/1000)


    df['forecast_date_time'] = df['timestamp_ms'].apply(convert_epoch_ms_to_datetime)

    if verbose >= 2:
        print(df.head())
    # Optional: set datetime as index
    #df.set_index('forecast_date_time', inplace=True)

    #df = df.drop('timestamp_ms', axis=1)

    readable_dt = pd.to_datetime(1670846400000, unit='ms', utc=True)
    if verbose >= 2:
        print(readable_dt)

    # Display the result
    if verbose >= 2:
        print(df.head())

    df.to_csv('testAirNDFD.csv', index=True)

def utcDateTimeConverter(dateTime, menuDateConvert=False):
    """
    Funcitons purpose is to convert time from UTC to a regular datetime.

    Input: list of utc times OR dictionary of files OR single integer from dataFrame Converter,

        menuDateConvert; converts from nonformatted utc time to formatted time for menu access,
                False is default and will 

    Return: formatted date/ or a string

    Author: Jarett T. Woodall
    """
        
    #Formatting Date time to our format for the menu
    if menuDateConvert == False:
        
        #UTC Time Zone
        UTC = tz.gettz('UTC')
        
        # Central Time Zone
        #central = tz.gettz('US/Central')

        #Convert from utc time to new datetime
        formattedDate = datetime.fromtimestamp(dateTime, tz=UTC)
        
        #date = datetime.fromtimestamp(dateTime, tz=central)
        
        # String
        stringFormatDate = formattedDate.strftime('%m-%d-%Y %H%M')
    
    # Structure used for converting for menu purposes in GUI
    else:
                        
        # To file specific date time that is used to reference weather company files
        #Date Time type
        formattedDate = datetime.strptime(dateTime, '%m-%d-%Y %H%M')
    
        #String format
        stringFormatDate = formattedDate.strftime('%Y-%m-%d %H%M')
    
    return stringFormatDate, formattedDate
#END: utcDateTimeConverter()

def far_to_cel(temp):
    """
    Farenheit to Celsius

    input: temperature (float or int)

    return: formatted Temperature
    """
        
    return (temp - 32) * 5/9
#END: def far_to_cel()

def select_hyperparams(model_name, lead_time):
    num_layers = None
    act_func = None
    neurons= None


    # combination is needed for CRPS
    if lead_time == 12:
        combination = 1

    elif lead_time == 48:
        combination = 2

    elif lead_time == 96 or lead_time == 120:
        combination = 2
    # ---


    # ------------------------------
    if lead_time == 12:
        if model_name == "CRPS":
            if combination == 1:
                num_layers = 3
                act_func = 'relu'
                neurons = 32

            elif combination == 2:
                num_layers = 2
                act_func = 'leaky_relu'
                neurons = 256 

        elif model_name == "MSE":
            if combination == 1:
                num_layers = 3
                act_func = 'leaky_relu'
                neurons = 32

        elif model_name == "PNN":
            num_layers = 3
            act_func = 'leaky_relu'
            neurons = 256


    # ------------------------------
    elif lead_time == 48:
        if model_name == "CRPS":
            if combination == 1:
                num_layers = 3
                act_func = 'relu'
                neurons = 32

            elif combination == 2:
                num_layers = 3
                act_func = 'selu'
                neurons = 64

        elif model_name == "MSE":
            if combination == 2:
                num_layers = 2
                act_func = 'leaky_relu'
                neurons = 16
        
        elif model_name == "PNN":
            num_layers = 2
            act_func = 'leaky_relu'
            neurons = 64


    # ------------------------------
    elif lead_time == 96 or lead_time == 120:

        if model_name == "CRPS":
            if combination == 1:
                num_layers = 3
                act_func = 'selu'
                neurons = 32 

            elif combination == 2:
                num_layers = 3
                act_func = 'relu'
                neurons = 100
                
        elif model_name == "MSE":
            if combination == 2:
                num_layers = 2
                act_func = 'leaky_relu'
                neurons = 16
        
        elif model_name == "PNN":
            num_layers = 1
            act_func = 'leaky_relu'
            neurons = 128


    return num_layers, act_func, neurons

def data_retriever_TWC_combiner(startTime='12/23/2022 16:00', endTime='12/27/2022 17:00', leadTime=12, padding = 48, TWC = True, percentile = "Median"):
    """
    Aligns TWC and model data based on forecast datetime intervals.
    Uses the full additional columns dataframe for filtering within the loop.
    """
    # Applies padding hours to the start and end of the event to ensure we have enough data before and after the event.
    parsed_start_date = datetime.strptime(startTime, '%m/%d/%Y %H:%M') - timedelta(hours=padding)
    parsed_end_date = datetime.strptime(endTime, '%m/%d/%Y %H:%M') + timedelta(hours=padding)

    # This accounts for the relevant offset time needed for data in predictions
    start_offset_reference = parsed_start_date - timedelta(hours=leadTime)
    end_offset_reference = parsed_end_date - timedelta(hours=leadTime)

    current_year = start_offset_reference.year
    if verbose >= 2:
        print(f"Current Year: {current_year}")
    extraColsFullDf = additional_columns_retriever(start_offset_reference, leadTime)

    current_time = start_offset_reference
    combined_df = pd.DataFrame()

    while current_time <= end_offset_reference:
        if verbose >= 2:
            print(f"Current Time: {current_time}")

        if TWC:
            airTempsdf = dataframe_retriever(current_time, percentile)[0]

        else:
            airTempsdf = dataframe_retriever_NDFD(current_time)
        new_forecast_df = build_median_forecast_row(airTempsdf, leadTime, percentile)

        # Ensure forecast data is a single row
        new_forecast_df = new_forecast_df.reset_index(drop=True)

        time_key = pd.to_datetime(current_time) - pd.Timedelta(hours=1)
        if time_key in extraColsFullDf.index:
            extraColsDf = extraColsFullDf.loc[[time_key]].reset_index(drop=True)
            if verbose >= 2:
                print(f"Found extra columns for {current_time}")
        else:
            extraColsDf = pd.DataFrame(columns=extraColsFullDf.columns)
            if verbose >= 2:
                print(f"Warning: No extra columns found for {current_time}, using empty row.")

        # Ensure both DataFrames have one row
        if verbose >= 2:
            print("Build Median Forecast Row:")
        if verbose >= 2:
            print(new_forecast_df)
        if verbose >= 2:
            print("Extra Columns Dataframe:")
        if verbose >= 2:
            print(extraColsDf)
        if len(new_forecast_df) != 1 or len(extraColsDf) != 1:
            if verbose >= 2:
                print(f"Warning: Skipping {current_time} due to unexpected row count.")
            current_time += timedelta(hours=1)
            continue

        # Avoid column name collisions
        forecast_cols_clean = [col for col in new_forecast_df.columns if col not in extraColsDf.columns]

        # Concatenate horizontally
        combined_row = pd.concat([extraColsDf, new_forecast_df[forecast_cols_clean]], axis=1)

        # Insert timestamp
        combined_row.insert(0, "dateAndTime", current_time)

        # Reorder columns if target exists
        target_col = "packeryATP_lighthouse"
        if target_col in extraColsDf.columns:
            insert_index = extraColsDf.columns.get_loc(target_col) + 1
            extra_cols = extraColsDf.columns.tolist()
            new_order = (
                ["dateAndTime"] +
                extra_cols[:insert_index] +
                forecast_cols_clean +
                extra_cols[insert_index:]
            )
            combined_row = combined_row.reindex(columns=new_order)
        else:
            if verbose >= 2:
                print(f"Warning: '{target_col}' not found at {current_time}, skipping reorder.")

        combined_df = pd.concat([combined_df, combined_row], axis=0, ignore_index=True)
        current_time += timedelta(hours=1)

    combined_df.to_csv('ColdStunningEventProto.csv')
    return combined_df
# End: def data_retriever_TWC_combiner()

def model_loader_tester(model_name, startTime, endTime, lead_times, start_iteration, end_iteration, cycle, padding, TWC, percentiles, all_models = False, base_path=repo_project_path):

    import time

    start_time = time.time()

    for lead_time_index, lead_time in enumerate(lead_times):
        if all_models == True:
            if verbose >= 1:
                print(f"\n\n\n\n\n\n ALL MODELS TRUE \n\n\n\n\n\n")
            percentiles = [f"member_{i}" for i in range(0, 100)]

        
        for percentile_index, percentile in enumerate(percentiles):
            

            testDf = data_retriever_TWC_combiner(startTime, endTime, lead_time, padding, TWC, percentile)

            initTimes = testDf['dateAndTime'].tolist()

            testingDates = dateTimeRetriever(testDf, lead_time)

            testingAirTemps = testDf['packeryATP_lighthouse'].tolist()

            testDf.to_csv('checking_column_order_in_TWC_Incorp.csv')

            # x_test, y_test = reshape_testing_only("descending", testDf, model_name)
            x_test, y_test = reshape_testing_only(ascend_or_descend, testDf, model_name)

            if model_name == "CRPS":
                output_units = 100    
            elif model_name == "PNN":
                output_units = 1

            prediction_column_names = []
            for k in range(output_units):
                prediction_column_names.append(f'pred_{k+1}')   

            for iteration in range(start_iteration, end_iteration + 1, 1):

                
                if verbose >= 1:
                    print(f"\n\n\n\n\n\n ")
                    print(f"index {lead_time_index}, lead_time : {lead_time}, in lead_times out of : {len(lead_times)}")
                    print(f"index {percentile_index}, percentile : {percentile}, in percentiles out of : {len(percentiles)}")
                    print(f"index {iteration}, in iteration start : {start_iteration} out of end : {end_iteration}")
                    print(f"\n\n\n\n\n\n ")

                num_layers, act_func, neurons = select_hyperparams(model_name=model_name, lead_time=lead_time) # select the hyperparameters based on model_name and lead_time 
                    
                combo_name = f"{model_name.lower()}-{num_layers}_layers-{act_func}-{neurons}_neurons"
                if model_name == "PNN":
                    combo_name = 'noCombo'

                # Path to folder for visualization results
                if TWC == True:
                    start_dt = datetime.strptime(start, '%m/%d/%Y %H:%M')
                    end_dt = datetime.strptime(end, '%m/%d/%Y %H:%M')

                    # Format for filename or variable name
                    results_name = f"TWC_results_{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}"

                else:
                    results_name = "NDFD_results"

                save_path = Path(base_path) / "src" / results_name / f"{model_name.lower()}_results" / f"{lead_time}h" / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"
                save_path.mkdir(parents=True, exist_ok=True)

                model = find_and_load_keras_model(base_path, lead_time, iteration, cycle, combo_name, model_name)

                if model_name == 'CRPS':

                    if verbose >= 2:
                        print(x_test.shape)
                    test_predictions = model.predict(x_test)
                elif model_name == 'PNN':
                    test_predictions, test_predictions_sigma = model.predict(list((x_test, x_test)))

                """SAVING PREDICTIONS AND OBSERVATIONS"""
                test_vs_preds = pd.DataFrame(columns=prediction_column_names, data=test_predictions)
                
                test_vs_preds.insert(loc=0, column='date_time', value=testingDates)

                test_vs_preds.insert(loc=1, column='target', value=y_test)

                # Creates a column with the observed air temperatures for reference
                test_vs_preds.insert(loc=2, column='obsv_airTemp', value=testingAirTemps)

                #Creates a column with the initialization times for reference
                test_vs_preds.insert(loc=3, column='init_time', value=initTimes)

                # add
                if model_name == 'PNN':
                    test_vs_preds.insert(loc=4, column='sigma_1', value=test_predictions_sigma)

                test_path = save_path / f"{percentile}_datetime_obsv_predictions.csv"

                if verbose >= 2:
                    print(test_path)
                test_vs_preds.to_csv(test_path)

    
    end_time = time.time()
    if verbose >= 2:
        print(f"Execution time: {end_time - start_time:.4f} seconds")

#END: def model_loader_tester()

def find_and_load_keras_model(base_path, leadTime, iteration, cycle, combo_name, model_name):
    if model_name == "CRPS":
        return find_and_load_keras_model_crps(base_path, leadTime, iteration, cycle, combo_name, model_name)
    elif model_name == "PNN":
        return find_and_load_keras_model_pnn(base_path, leadTime, iteration, cycle, combo_name, model_name)

def find_and_load_keras_model_crps(base_path, leadTime, iteration, cycle, combo_name, model_name):
    base = Path(base_path)
    leadTime = str(leadTime) + 'h'
    model_lowered = model_name.lower()
    model_path = model_lowered + '_results'
    target_dir = base / 'src' / 'results' / model_path / leadTime / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"

    if verbose >= 2:
        print(target_dir)

    # Search for any file ending in "_keras" with no extension
    model_path = next((f for f in target_dir.iterdir() if f.name.endswith('.keras') and f.is_file()), None)

    if model_path:
        if verbose >= 2:
            print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        if verbose >= 2:
            print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"No matching Keras model file found in {leadTime} directory.")

def find_and_load_keras_model_pnn(base_path, leadTime, iteration, cycle, combo_name, model_name):
    
    from pathlib import Path

    # base_path = r"C:\Users\woody\Work\UQ4ML_WaterTemp"
    base_path = r"C:\Users\hmarrero\Documents\GitHub_Repos\UQ4ML_WaterTemp\UQ4ML_WaterTemp"
    leadTime_str = f"{leadTime}h"
    model_dir = f"pnn-{combo_name}-cycle_{cycle}-iteration_{iteration}"

    path_to_csv = Path(base_path) / "src" / "results" / "pnn_results" / leadTime_str / model_dir / "model.keras"
    
    if path_to_csv:
        if verbose >= 1:
            print(f"Loading model from: {path_to_csv}")

        import keras
        import tensorflow as tf
        @keras.saving.register_keras_serializable(package="hector_pnn", name="sigma_activation")
        def SigmaActivation(x):
            return tf.nn.elu(x)+1.1
        
        # model = tf.keras.models.load_model(model_path)

        model = load_model(path_to_csv)
        if verbose >= 1:
            print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"No matching Keras model file found in {leadTime} directory.")

def reshape_testing_only(input_structure, testing, model):
    '''reshape_testing_only() reshapes only the testing dataset for model inference'''

    # Determine where to start grabbing input features
    if input_structure == "descending":
        input_column_start = 1
    elif input_structure == "ascending":
        input_column_start = 3
    else:
        raise ValueError("input_structure must be 'ascending' or 'descending'")

    #if verbose >= 2:
        print(testing.columns)

    all_columns = testing.columns.tolist()

    selected_columns = testing.iloc[:, input_column_start:-1].columns.tolist()

    removed_columns = [col for col in all_columns if col not in selected_columns]
    
    if verbose >= 2:
        print("Removed columns:")
    if verbose >= 2:
        print(removed_columns)

    # Extract inputs and target from testing set
    if verbose >= 2:
        print("\n\n\n\n")
    if verbose >= 2:
        print(testing)
    if verbose >= 2:
        print(testing.shape)
    if verbose >= 2:
        print(testing.columns)

    testingData = testing.iloc[:, input_column_start:-1].values.astype(float)

    if verbose >= 2:
        print(testingData.columns)
    testingTarget = testing.iloc[:, -1].values.astype(float)

    # Reshape based on model type
    if model == "LSTM":
        x_test = np.reshape(testingData, (testingData.shape[0], 1, testingData.shape[1]))
        y_test = np.expand_dims(testingTarget, axis=-1)
    else:
        x_test = testingData
        y_test = testingTarget

    return x_test, y_test
#END: def reshape_testing_only()

def file_retrieval(target_date):
    # target_dir = r"C:\Users\woody\Work\UQ4ML_WaterTemp\data\June_May_Datasets"
    target_dir = r"C:\Users\hmarrero\Documents\GitHub_Repos\UQ4ML_WaterTemp\data\June_May_Datasets"
    
    # Ensure target_date is a datetime object
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m")  # e.g., "2021-02"
    elif isinstance(target_date, int):
        target_date = datetime(target_date, 1, 1)

    # Adjust year logic: if month is before June, use previous year as start
    if target_date.month < 6:
        target_year = target_date.year - 1
    else:
        target_year = target_date.year

    # Pattern to match filenames like "2020_2021.csv"
    pattern = re.compile(r"(20\d{2})_(20\d{2})")

    for filename in os.listdir(target_dir):
        if os.path.isfile(os.path.join(target_dir, filename)):
            match = pattern.search(filename)
            if match and int(match.group(1)) == target_year:
                file_path = os.path.join(target_dir, filename)
                if verbose >= 2:
                    print("Found file:", file_path)
                df = pd.read_csv(file_path)
                if verbose >= 2:
                    print("Loaded:", df.shape[0], "rows ×", df.shape[1], "columns")
                return df

    if verbose >= 2:
        print(f"No file found in {target_dir} containing year pair starting with '{target_year}'")
    return None

def additional_columns_retriever(time, leadtime):

    """
    Retrieves the full additional columns dataframe for a given year and lead time.
    Filters out unnecessary columns and sets datetime index.
    """
    df = file_retrieval(time)
    df = creatingAdditionalColumns(df, 'descending', leadtime, 24, 24, 1, 0.0)

    if verbose >= 2:
        print(df.columns)
    if verbose >= 2:
        print(len(df))
    #exit()
    df["dateAndTime"] = pd.to_datetime(df["dateAndTime"])
    df.set_index("dateAndTime", inplace=True)
    
    # Remove columns that are not needed
    df = df.loc[:, ~df.columns.str.contains("airTemperature_pred__")]

    return df
#END: def additional_columns_retriever()

def build_median_forecast_row(source_df, leadtime, percentile):
    """
    Constructs a single-row DataFrame with leadtime-based forecast columns
    filled with values from the 'Median' column of source_df.

    The 'forecast_date_time' is set to one hour before the first index timestamp.
    """

    df = source_df.copy()
    df.index = pd.to_datetime(df.index)

    # Create dynamic column names
    col_names = [f"airTemperature_pred__{str(h+1).zfill(2)}h_forecast" for h in range(leadtime)]

    if percentile == "Median":
        # Pull median values (up to leadtime count)
        median_values = df["Median"].iloc[:leadtime].values
    elif percentile == "max":
        median_values = df["max"].iloc[:leadtime].values
    elif percentile == "min":
        median_values = df["min"].iloc[:leadtime].values
    else:
        median_values = df[percentile].iloc[:leadtime].values

    # Build row dictionary
    forecast_row = dict(zip(col_names, median_values))

    # Shift forecast time back by one hour
    adjusted_time = df.index[0] - pd.Timedelta(hours=1)
    forecast_row["forecast_date_time"] = adjusted_time

    # Create final DataFrame
    forecast_df = pd.DataFrame([forecast_row])
    forecast_df.set_index("forecast_date_time", inplace=True)

    return forecast_df
#END: def time_reconcilliation_function():

#---------OLD CODE BELOW THIS LINE----------------
def additional_columns_retriever_old(time, year, leadtime):
    """
    This function retrieves a row from the additional columns file based on a specific timestamp.
    It builds the dataframe from model data and aligns it using the forecast datetime index.
    """

    # Build the file path safely
    df = file_retrieval(year)

    df = creatingAdditionalColumns(df, 'descending', leadtime, 24, 24, 1, 0.0)
    df["dateAndTime"] = pd.to_datetime(df["dateAndTime"])
    df.set_index("dateAndTime", inplace=True)

    # Remmovces the columns that are not needed
    df = df.loc[:, ~df.columns.str.contains("airTemperature_pred__")]

    # Ensure `time` is a datetime object if it isn’t already, also ensures data lines up
    time = pd.to_datetime(time) - pd.Timedelta(hours=1)


    # Retrieves the row at the specified time, also checks for existence
    if time in df.index:
        result_df = df.loc[[time]]  # preserves DataFrame structure
    else:
        result_df = pd.DataFrame()  # empty DataFrame if time not found

    return result_df
# End: def additional_columns_retriever()

def data_retriever_TWC_combiner_old(startTime='12/23/2022 16:00', endTime='12/27/2022 17:00', leadTime=12):
    """
    This function is designed to reconcile the time between the TWC data and the model data.
    It will take the TWC data and align it with the model data based on the forecast date times.
    """

    parsed_start_date = datetime.strptime(startTime, '%m/%d/%Y %H:%M')
    parsed_end_date = datetime.strptime(endTime, '%m/%d/%Y %H:%M')

    start_offset_reference = parsed_start_date - timedelta(hours=leadTime)
    end_offset_reference = parsed_end_date - timedelta(hours=leadTime)

    current_time = start_offset_reference
    combined_df = pd.DataFrame()

    while current_time <= end_offset_reference:

        if verbose >= 2:
            print(f"Current Time: {current_time}")
        airTempsdf = dataframe_retriever(current_time)[0]
        new_forecast_df = build_median_forecast_row(airTempsdf, leadTime)

        current_year = current_time.year
        extraColsDf = additional_columns_retriever(current_time, current_year, leadTime)

        combined_row = pd.concat([new_forecast_df, extraColsDf], axis=1)

        forecast_cols = new_forecast_df.columns.tolist()
        extra_cols = extraColsDf.columns.tolist()

        target_col = "packeryATP_lighthouse"
        if target_col in extra_cols:
            insert_index = extra_cols.index(target_col) + 1
        else:
            raise ValueError(f"'{target_col}' not found in extraColsDf")

        new_order = (
            extra_cols[:insert_index] +
            forecast_cols +
            extra_cols[insert_index:]
        )

        combined_row = combined_row[new_order]
        combined_row.insert(0, "dateAndTime", current_time)

        combined_df = pd.concat([combined_df, combined_row], axis=0, ignore_index=True)

        current_time += timedelta(hours=1)

    #combined_df.to_csv('ColdStunningEventProto.csv')
    return combined_df
# End: def data_retriever_TWC_combiner_old()

def find_and_load_keras_model_old(base_path, leadTime, iteration, cycle, combo_name, model_name):
    base = Path(base_path)
    leadTime = str(leadTime) + 'h'
    model_lowered = model_name.lower()
    model_path = model_lowered + '_results'
    target_dir = base / 'src' / 'results' / model_path / leadTime / f"{combo_name}-cycle_{cycle}-iteration_{iteration}"

    if verbose >= 2:
        print(target_dir)

    # Search for any file ending in "_keras" with no extension
    model_path = next((f for f in target_dir.iterdir() if f.name.endswith('.keras') and f.is_file()), None)

    if model_path:
        if verbose >= 2:
            print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        if verbose >= 2:
            print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"No matching Keras model file found in {leadTime} directory.")

def file_retrieval_old(year):
    target_dir = r"C:\Users\woody\Work\UQ4ML_WaterTemp\data\June_May_Datasets"
    year = str(year)

    # Pattern to match filenames that include two years like "2022_2023"
    # and extract the first year to compare against
    pattern = re.compile(r"(20\d{2})_(20\d{2})")

    for filename in os.listdir(target_dir):
        if os.path.isfile(os.path.join(target_dir, filename)):
            match = pattern.search(filename)
            if match and match.group(1) == year:
                file_path = os.path.join(target_dir, filename)
                if verbose >= 2:
                    print("Found file:", file_path)

                if verbose >= 2:
                    print(file_path)
                df = pd.read_csv(file_path)
                if verbose >= 2:
                    print("Loaded:", df.shape[0], "rows ×", df.shape[1], "columns")
                return df

    if verbose >= 2:
        print(f"No file found in {target_dir} containing year pair starting with '{year}'")
#END: def file_retrieval(year)
# Runs the code to test the model with the TWC air temperature data
model_loader_tester(model, start, end, leadTimes, start_iteration, end_iteration, cycle, padding, TWC, 
uncertainty_test_list, allModels)