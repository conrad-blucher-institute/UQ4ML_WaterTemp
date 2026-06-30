import tensorflow as tf

class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_message = f"Epoch {epoch + 1}: " + ", ".join([f"{key}={value:.4f}" for key, value in logs.items()]) + "\n"
        
        # Print to console (optional)
        print(log_message, end="")
        
        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(log_message)

'''  
-------------------------------------------------------------------------
                            def preparingData
input:
        path_to_data - a path to the folder that contains the csv files
        input_structure - 
        independent_year - 
        input_hours_forecast -
        atp_hours_back - 
        wtp_hours_back -
        pred_atp_intervals - 
        IPPOffset = 0.0 - 
        cycle = 0 -
        model = "MLP" - 
        verbose = 0 - 
purpose:
        this is the main function to create an input vector for the ML model. 
output: 
        x_train - input training data
        y_train - target training data
        x_val - input validation data
        y_val - target validation data
        x_test - input testing data
        y_test - target testing data
        training_dates - list of datetime objects for training data
        validation_dates - list of datetime objects for validation data
        testingDates - list of datetime objects for testing data
        testingAirTemps - list of air temperatures for testing data
------------------------------------------------------------------------- '''
def preparingData(path_to_data, input_structure, independent_year, input_hours_forecast, atp_hours_back, 
                  wtp_hours_back, pred_atp_interval, IPPOffset = 0.0, cycle = 0, model="MLP", verbose=0):
    
    '''preparingData() is the driver function'''
    # Importing libraries
    from datetime import datetime
    import pandas as pd
    import numpy as np

    # Function call to read the data
    # excluding data_year1, which is the independent testing data
    data_year2, data_year3, data_year4, data_year5 = readingData(path_to_data)

    #verbose determines how much info we see 
    if verbose == 3:
        for i, v in enumerate([data_year2, data_year3, data_year4, data_year5]):
            print(v.columns)

    # with open('time_for_offsetcreator', 'w') as file:
    #     totaltime = end_time - start_time
    #     file.write(str(totaltime))
     # to alternate between the two independent years as testing years

    # if independent_year == 'cycle', then that means that we are doing the regular cycle year as testing
    
    if independent_year != 'cycle':

        if independent_year == '2021':
            data_independent_year = pd.read_csv("data/ESB_datasets/esb_2020_2021.csv")
        # elif independent_year == '2024':
        #     data_independent_year = pd.read_csv("../UQ4ML_WaterTemp/data/June_May_Datasets/june_atp_and_wtp_2023_2024_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv")

        
        year_independent = creatingAdditionalColumns(data_independent_year, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
        
        # Debug: inspect year_independent
        print("year_independent shape:", year_independent.shape)
        print("year_independent columns:", year_independent.columns.tolist())
        print("year_independent head:\n", year_independent.head())
        print("year_independent tail:\n", year_independent.tail())
        print("Missing values in year_independent:\n", year_independent.isnull().sum())
        
        # Export to CSV for inspection
        year_independent.to_csv('debug_year_independent.csv', index=False)

        # return
    
    elif independent_year == 'cycle':
        year_independent = independent_year


    # Function call to create additional columns
    start_time = datetime.now()

    # excluding the independent testing set 
    year2 = creatingAdditionalColumns(data_year2, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year3 = creatingAdditionalColumns(data_year3, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year4 = creatingAdditionalColumns(data_year4, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year5 = creatingAdditionalColumns(data_year5, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)

    print('finished input construction')
    end_time = datetime.now()

    
    if verbose == 3:
        for i, v in enumerate([year2, year3, year4, year5]):
            print(v.columns)
    
    #with open('time_for_inputconstruction', 'w') as file:
    #    totaltime = end_time - start_time
    #    file.write(str(totaltime) + ',\t\t\tlt,' + str(input_hours_forecast) + ',temp,' + str(IPPOffset) + ',cycle,' + str(cycle))

    # training_data, testing_data, validation_data = splittingData(IPPYear1, IPPYear2, IPPYear3, IPPYear4, IPPYear5, IPPYear6, IPPYear7, IPPYear8, IPPYear9, IPPYear10, cycle)
    # year2.to_csv('year2.csv')


    year_independent = cycle

    # Debug logger: writes to debug_preparingData.txt when verbose >= 2
    # The log file path comes from config's 'debug_log_dir' key (set by the tuner's
    # output directory), falling back to the current working directory.
    _debug_log_path = None
    def _dlog(msg):
        nonlocal _debug_log_path
        if verbose < 2:
            return
        print(f'\n  [preparingData] {msg}\n')
        if _debug_log_path is None:
            import os
            _debug_log_dir = os.environ.get('TUNER_DEBUG_LOG_DIR', '.')
            _debug_log_path = os.path.join(_debug_log_dir, 'debug_preparingData.txt')
        with open(_debug_log_path, 'a') as _f:
            _f.write(f'{msg}\n\n')

    _dlog(f'params: input_structure={input_structure}, input_hours_forecast={input_hours_forecast}, '
          f'atp_hours_back={atp_hours_back}, wtp_hours_back={wtp_hours_back}, cycle={cycle}, model={model}')

    _dlog(f'raw data shapes: year2={data_year2.shape}, year3={data_year3.shape}, '
          f'year4={data_year4.shape}, year5={data_year5.shape}')

    _dlog(f'after creatingAdditionalColumns: year2={year2.shape}, year3={year3.shape}, '
          f'year4={year4.shape}, year5={year5.shape}')

    _dlog(f'year2 columns ({len(year2.columns)}): {list(year2.columns)[:5]}...{list(year2.columns)[-3:]}')

    # FIX: was passing raw data_year2..5 (3 columns) instead of
    # year2..5 (with engineered features). Models were training on only
    # 1 feature (Air Average) instead of ~62 lag features.
    # R1 fix (esb_refactor, Stage B): splittingData() requires 6 args
    # (year2..year5, year_independent, cycle) but was called with 5, raising a
    # TypeError the moment preparingData() ran. Pass year_independent (set to the
    # int `cycle` above) so the call is valid. With an int year_independent the
    # split leaves the test set empty, which downstream tolerates (Quirk Q8) and
    # the grid-search tuner ignores (it uses data[:4] + its own 2021 eval).
    training_data, testing_data, validation_data = splittingData(year2, year3, year4, year5, year_independent, cycle)

    print('finished splitting the data')

    print("Testing data shape:", testing_data.shape)
    print("train data shape:", training_data.shape)
    print("val data shape:", validation_data.shape)
    _dlog(f'after splittingData: train={training_data.shape}, test={testing_data.shape}, val={validation_data.shape}')

    # Function call to count the number of missing values
    training_numMissingValues, training_percMissVal = countingMissingValues(training_data)
    testing_numMissingValues, testing_percMissVal = countingMissingValues(testing_data)
    validation_numMissingValues, validation_percMissVal = countingMissingValues(validation_data)

    #  Printing the number of missing values
    print()
    print('Training Missing Values: ', training_numMissingValues)
    print('Training Percentatge of Missing Values: ', round(training_percMissVal,4), ' %')
    print()
    print('Testing Missing Values: ', testing_numMissingValues)
    print('Testing Percentatge of Missing Values: ', round(testing_percMissVal,4), ' %')
    print()
    print('Validation Missing Values: ', validation_numMissingValues)
    print('Validation Percentatge of Missing Values: ', round(validation_percMissVal,4), ' %')
    print()

    dataframe_checker(-999, [training_data, testing_data, validation_data]) # checking for any rogue number less than -999

    # Function call to delete the rows that at least one of the columns contain a missing value (-999)
    print("\n*** BEFORE DELETION ***")
    print(f"Training rows with -999: {(training_data == -999).any(axis=1).sum()}")
    print(f"Testing rows with -999: {(testing_data == -999).any(axis=1).sum()}")
    print(f"Validation rows with -999: {(validation_data == -999).any(axis=1).sum()}")
    
    training = deletingMissingValues(training_data)
    testing = deletingMissingValues(testing_data)
    validation = deletingMissingValues(validation_data)

    print("\n*** AFTER DELETION ***")
    print(f"Training rows with -999: {(training == -999).any(axis=1).sum()}")
    print(f"Testing rows with -999: {(testing == -999).any(axis=1).sum()}")
    print(f"Validation rows with -999: {(validation == -999).any(axis=1).sum()}")
    print(f"Training size: {training.shape[0]} (was {training_data.shape[0]})")
    print(f"Testing size: {testing.shape[0]} (was {testing_data.shape[0]})")
    print(f"Validation size: {validation.shape[0]} (was {validation_data.shape[0]})\n")

    # Save cleaned data for verification
    training.to_csv('debug_training_data_CLEANED.csv')
    testing.to_csv('debug_testing_data_CLEANED.csv')
    validation.to_csv('debug_validation_data_CLEANED.csv')
    print("Saved cleaned data to debug_*_CLEANED.csv files for verification\n")

    dataframe_checker(-100, [training, testing, validation]) # checking for any rogue number less than -100
    _dlog(f'after deletingMissingValues: train={training.shape}, test={testing.shape}, val={validation.shape}')

    # For new calculations created in the Summer of 2023
    training_dates = dateTimeRetriever(training, input_hours_forecast) if not training.empty else []
    validation_dates = dateTimeRetriever(validation, input_hours_forecast) if not validation.empty else []
    if not testing.empty:
        testingDates = dateTimeRetriever(testing, input_hours_forecast)
        testingAirTemps = testing['Air Average'].tolist()
    else:
        testingDates = []
        testingAirTemps = []

    print()

    # Function call to reshpe the dataset and prepare it to be used as in input for the neural network
    x_train, y_train, x_val, y_val, x_test, y_test = reshaping(input_structure, training, testing, validation, model)

    _dlog(f'after reshaping: x_train={x_train.shape}, y_train={y_train.shape}, '
          f'x_val={x_val.shape}, y_val={y_val.shape}, x_test={x_test.shape}, y_test={y_test.shape}')

    if verbose == 3:
        print("NaNs in x_train:", np.isnan(x_train).sum())
        print("NaNs in y_train:", np.isnan(y_train).sum())
        print("Infs in x_train:", np.isinf(x_train).sum())
        print("Infs in y_train:", np.isinf(y_train).sum())
        print("NaNs in x_val:", np.isnan(x_val).sum())
        print("NaNs in y_val:", np.isnan(y_val).sum())
        print("NaNs in x_test:", np.isnan(x_test).sum())
        print("NaNs in y_test:", np.isnan(y_test).sum())
        print("Infs in x_val:", np.isinf(x_val).sum())
        print("Infs in y_val:", np.isinf(y_val).sum())
        print("Infs in x_test:", np.isinf(x_test).sum())
        print("Infs in y_test:", np.isinf(y_test).sum())

    return x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAirTemps

'''  
-------------------------------------------------------------------------
                            def readingData
input:
        path_to_data - a path to the folder that contains the csv files                        
purpose:
        find all of the csv files from the given path and sort them.
        print the number of files found, and their names. 
        set the first file aside for independent testing year.
        load each csv file into a dataframe and store them in a list.
output: 
        data_list - a list of dataframes for each year 
------------------------------------------------------------------------- '''
def readingData(path_to_data):


    import os
    import glob
    import pandas as pd

    # find all CSV files and sort them 
    csv_files = sorted(glob.glob(os.path.join(path_to_data, "*.csv")))

    print(f"Found {len(csv_files)} CSV files.")
    print("Files:", [os.path.basename(f) for f in csv_files])

    # skip the first file; the first is our winter storm uri independent testing year
    csv_files_to_read = csv_files[1:]

    # Load each CSV into a DataFrame
    data_list = [pd.read_csv(f) for f in csv_files_to_read]

    return data_list

# r treats this string as a RAW STRING 
r'''  
-------------------------------------------------------------------------
                            def creatingAdditionalColumns
input:
        df - a dataframe that contains the data for one year
        input_structure - 
        input_hours_forecast - 
        atp_hours_back - 
        wtp_hours_back -
        pred_atp_interval - 
        IPPOffset = 0.0 -                  
purpose:
        creating columns for the past and future (perfect prognosis) hours
output: 
        
------------------------------------------------------------------------- '''
def creatingAdditionalColumns(df, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset=0.0):
    """Optimized implementation using pandas.shift and concat.

    This replaces the original per-column insertion loops with vectorized
    shift operations and a single concat, which is much faster on large
    DataFrames.
    """
    import warnings
    import pandas as pd
    import numpy as np
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    interval = pred_atp_interval
    new_columns = {}

    # Work on a copy and apply IPP offset vectorized
    df_work = df.copy()
    if IPPOffset != 0.0:
        mask = df_work['Air Average'] != -999
        df_work.loc[mask, 'Air Average'] = df_work.loc[mask, 'Air Average'] + IPPOffset

    # Vectorized: create lagged air temperature columns using pandas.shift()
    for lag in range(1, atp_hours_back + 1):
        col_name = f'airTemperature__{lag}h_ago'
        new_columns[col_name] = df_work['Air Average'].shift(lag).fillna(-999).values

    # Vectorized: create lagged water temperature columns using pandas.shift()
    for lag in range(1, wtp_hours_back + 1):
        col_name = f'waterTemperature__{lag}h_ago'
        new_columns[col_name] = df_work['Water Average'].shift(lag).fillna(-999).values

    # Vectorized: create forward-looking air temperature columns (perfect prognosis)
    for hours_ahead in range(interval, input_hours_forecast + 1, interval):
        col_name = f'airTemperature_pred__{hours_ahead}h_forecast'
        shifted = df_work['Air Average'].shift(-hours_ahead)
        new_columns[col_name] = shifted.fillna(-999).values

    # Vectorized: create target (forward-looking water temperature)
    target_col = f'waterTemperature_{input_hours_forecast}h_forecast'
    shifted_target = df_work['Water Average'].shift(-input_hours_forecast)
    new_columns[target_col] = shifted_target.fillna(-999).values

    # Add all new columns at once via concat
    df = pd.concat([df_work, pd.DataFrame(new_columns)], axis=1)

    if input_structure == "descending":
        water_temp_columns = [col for col in df.columns if "waterTemperature__" in col]
        air_temp_columns = [col for col in df.columns if "airTemperature__" in col and "_ago" in col]
        forecast_columns = [col for col in df.columns if "forecast" in col]
        other_columns = [col for col in df.columns if col not in water_temp_columns + air_temp_columns + forecast_columns]

        if "Water Average" in other_columns:
            other_columns.remove("Water Average")
        if "Air Average" in other_columns:
            other_columns.remove("Air Average")

        reordered_columns = (
            other_columns
            + water_temp_columns[::-1]
            + ["Water Average"]
            + air_temp_columns[::-1]
            + ["Air Average"]
            + forecast_columns
        )

        df = df[reordered_columns]
        return df

    elif input_structure == "ascending":
        return df

import pandas as pd
from typing import Union, Tuple


'''  
-------------------------------------------------------------------------
                            def splittingData
input:
        year2 - dataframe for year 2 (2021-2022)    
        year3 - dataframe for year 3 (2022-2023)
        year4 - dataframe for year 4 (2023-2024)
        year5 - dataframe for year 5 (2024-2025)
        cycle - an integer that indicates which rotation we are on.
                need to refactor all mentions of cycle to rotation.

process: last item becomes testing
        second to last item becomes validation
        everything else is training
        rotates the order of the list/years based on which cycle we want,
        rotates "cycle" number of times
        
output: 
        training - dataframe for training data
        testing - dataframe for testing data
        validation - dataframe for validation data
------------------------------------------------------------------------- '''
def temp_splittingData(year2, year3, year4, year5, cycle):
    '''splittingData() groups the data into training, testing, and validation
    --Will rotate through the years as the cycle changes'''
    import pandas as pd

    # maybe in future refactoring we can put in a list instead of the years individually
    yearList = [year2,year3,year4,year5]

    training = pd.DataFrame()
    testing = pd.DataFrame()
    validation = pd.DataFrame()
    #"""
    
    # loop until we get to the version we are trying to make
    for j in range(cycle+1):

        for i in range(len(yearList)):

            # move everything in list right by 1 index, then slice off the last value
            if i > 0:
                yearList = [yearList[-1]] + yearList[:-1]

        # loop through our 10 years
        for index in range(len(yearList)):
            # seperate years 1-8 into training
            if index < len(yearList)-2:
                training = pd.concat([training, yearList[index]])
            # seperate 9th year into validation
            if index == len(yearList)-2:
                validation = pd.concat([validation, yearList[index]])
            # seperate 10th year into testing
            if index == len(yearList)-1:
                testing = pd.concat([testing, yearList[index]])

                
        if j == cycle:
            return training, testing, validation
        
        ### reset lists to empty
        training = pd.DataFrame()
        testing = pd.DataFrame()
        validation = pd.DataFrame()

'''  
-------------------------------------------------------------------------
                            def splittingData
input:
        year2 - dataframe for year 2
        year3 - dataframe for year 3
        year4 - dataframe for year 4
        year5 - dataframe for year 5
        cycle - an integer that indicates which cycle we are on. aka rotation.
purpose: 
        this is identical to temp_splittingData; only this one utilizes
        independent testing year. may need to rid the parameters and just use
        a list of dataframes instead of manually inputting each year. thus
        there will be no need for two functions.
output: 
        training - dataframe for training data
        testing - dataframe for testing data
        validation - dataframe for validation data
------------------------------------------------------------------------- '''
def splittingData(year2, year3, year4, year5, year_independent, cycle):
    import pandas as pd

    yearList = [year2, year3, year4, year5]

    training = pd.DataFrame()
    testing = pd.DataFrame()
    validation = pd.DataFrame()

    for j in range(cycle+1):
        for i in range(len(yearList)):
            if i > 0:
                yearList = [yearList[-1]] + yearList[:-1]

        for index in range(len(yearList)):
            if index < len(yearList)-2:
                training = pd.concat([training, yearList[index]])
            elif index == len(yearList)-2:
                validation = pd.concat([validation, yearList[index]])
            elif index == len(yearList)-1:
                if isinstance(year_independent, pd.DataFrame):
                    print("USING INDEPENDENT TEST YEAR")
                    testing = pd.concat([testing, year_independent])
                elif year_independent == "cycle":
                    print("USING REGULAR CYCLE TESTING")
                    testing = pd.concat([testing, yearList[index]])

        if j == cycle:
            return training, testing, validation

        training = pd.DataFrame()
        testing = pd.DataFrame()
        validation = pd.DataFrame()


'''  
-------------------------------------------------------------------------
                         def countingMissingValues
input:
        df - 
purpose: 
        count the rows in the dataframe that contain a missing values in 
        at least one of the columns.
output: 
        numMissValues - 
        percMissValues - 
------------------------------------------------------------------------- '''
def countingMissingValues(df):
    missing_standard = df.isna().any(axis=1)
    missing_custom = df.isin([-999]).any(axis=1)
    missingValues = df[missing_standard | missing_custom]
    
    numMissValues = len(missingValues)
    percMissValues = (numMissValues/len(df))*100

    return numMissValues, percMissValues


'''  
-------------------------------------------------------------------------
                        def og_deletingMissingValues
input:
        df - 
purpose: 
        delete the rows where at least one of the columns contain a missing value.
output: 
        df 
------------------------------------------------------------------------- '''
def og_deletingMissingValues(df):
    valueRemove = [-999]
    # Exclude 'date' column from missing value checks
    cols_to_check = [col for col in df.columns if col != 'date']
    mask = (df[cols_to_check].isin(valueRemove)).any(axis=1) | df[cols_to_check].isna().any(axis=1)
    df = df[~mask]
    df = df.dropna(subset=cols_to_check)
    return df

'''  
-------------------------------------------------------------------------
                        def deletingMissingValues
input:
        df - dataframe to clean
purpose: 
        delete the rows where at least one of the columns contain a missing value (-999).
        This removes rows with incomplete sequences that were created during feature engineering.
output: 
        df - cleaned dataframe with rows containing -999 removed
------------------------------------------------------------------------- '''
def deletingMissingValues(df):
    """
    Robustly removes rows containing -999 sentinel values used for missing data.
    Also removes rows with NaN values.
    """
    import pandas as pd
    
    if df.empty:
        return df.copy()
    
    # Exclude 'date' column from missing value checks
    cols_to_check = [col for col in df.columns if col != 'date']
    
    # Debug: show initial state
    initial_rows = len(df)
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Remove rows with -999 or NaN in any column
    # Using multiple conditions for robustness
    mask_nan = df[cols_to_check].isna().any(axis=1)
    mask_missing = (df[cols_to_check] == -999).any(axis=1)
    
    # Combine masks
    rows_to_remove = mask_nan | mask_missing
    
    # Keep only rows without missing values
    df = df[~rows_to_remove]
    
    # Final check: drop any remaining NaNs just to be safe
    df = df.dropna(subset=cols_to_check, how='any')
    
    # Debug: show final state
    final_rows = len(df)
    rows_removed = initial_rows - final_rows
    
    if initial_rows > 0:
        pct_removed = (rows_removed / initial_rows * 100)
        rows_with_999_after = (df[cols_to_check] == -999).any(axis=1).sum() if len(df) > 0 else 0
        print(f"  Rows: {initial_rows} → {final_rows} (removed {rows_removed}, {pct_removed:.1f}%)")
        if rows_with_999_after > 0:
            print(f"  ⚠️  WARNING: {rows_with_999_after} rows still contain -999 after deletion!")
    
    return df



'''  
-------------------------------------------------------------------------
                        def reshaping
input:
        input_structure - 
        training - 
        testing - 
        validation - 
        model - 
purpose: 
        reshape the training, testing, and validation datasets to be able to 
        use them as an input for the model.
output: 
        x_train -
        y_train - 
        x_val -
        y_val -
        x_test - 
        y_test -  
------------------------------------------------------------------------- '''
def reshaping(input_structure, training, testing, validation, model):
    import numpy as np

    if input_structure == "descending":
        input_column_start = 1
    elif input_structure == "ascending":
        input_column_start = 3

    # handle empty dataframes
    if training.empty:
        trainingData = np.empty((0, 0))
        trainingTarget = np.empty((0,))
    else:
        trainingData = training.iloc[:,input_column_start:-1].values.astype(float)
        trainingTarget = training.iloc[:,-1].values.astype(float)

    if validation.empty:
        validationData = np.empty((0, 0))
        validationTarget = np.empty((0,))
    else:
        validationData = validation.iloc[:,input_column_start:-1].values.astype(float)
        validationTarget = validation.iloc[:,-1].values.astype(float)

    if testing.empty:
        testingData = np.empty((0, 0))
        testingTarget = np.empty((0,))
    else:
        testingData = testing.iloc[:,input_column_start:-1].values.astype(float)
        testingTarget = testing.iloc[:,-1].values.astype(float)

    if(model == "LSTM"):
        x_train = np.reshape(trainingData, (trainingData.shape[0], 1, trainingData.shape[1]))
        x_test = np.reshape(testingData, (testingData.shape[0], 1, testingData.shape[1]))
        x_val = np.reshape(validationData, (validationData.shape[0], 1, validationData.shape[1]))

        y_train = np.expand_dims(trainingTarget, axis = -1)    
        y_test = np.expand_dims(testingTarget, axis = -1)    
        y_val = np.expand_dims(validationTarget, axis = -1)
    else:
        x_train = trainingData
        x_test = testingData
        x_val = validationData

        y_train = trainingTarget
        y_test = testingTarget
        y_val = validationTarget

    return x_train, y_train, x_val, y_val, x_test, y_test


def prepare_independent_year(csv_path, input_structure, lead_time,
                             atp_hours_back, wtp_hours_back,
                             pred_atp_interval=1, IPPOffset=0.0):
    """Prepare the independent test year (e.g. esb_2020_2021.csv) for evaluation.

    Runs the same feature-engineering pipeline as preparingData() but on a single
    CSV that readingData() intentionally skips. Returns numpy arrays ready for
    model.evaluate() or model.predict(), plus datetime labels.

    Returns:
        (X, y, dates) where X has shape (n_samples, n_features),
        y has shape (n_samples,), and dates is a list of datetime strings.
    """
    import pandas as pd

    df_raw = pd.read_csv(csv_path)
    df_features = creatingAdditionalColumns(
        df=df_raw,
        input_structure=input_structure,
        input_hours_forecast=lead_time,
        atp_hours_back=atp_hours_back,
        wtp_hours_back=wtp_hours_back,
        pred_atp_interval=pred_atp_interval,
        IPPOffset=IPPOffset,
    )
    df_clean = deletingMissingValues(df_features)

    # dateTimeRetriever mutates the 'date' column, so pass a copy
    dates = dateTimeRetriever(df_clean.copy(), lead_time)

    col_start = 1 if input_structure == 'descending' else 3
    X = df_clean.iloc[:, col_start:-1].values.astype(float)
    y = df_clean.iloc[:, -1].values.astype(float)

    return X, y, dates


'''
-------------------------------------------------------------------------
                        def dateTimeRetriever
input:
        dataset 
        input_hours_forecast
purpose: 
        reshape the training, testing, and validation datasets to be able to 
        use them as an input for the model.
output: 
        x_train -
        y_train - 
        x_val -
        y_val -
        x_test - 
        y_test -  
------------------------------------------------------------------------- '''
def dateTimeRetriever(dataset, input_hours_forecast):
    import pandas as pd
    '''This function is designed to grab the date times from the testing data set for computations.'''

    # Parse ISO 8601 dates automatically
    dataset['date'] = pd.to_datetime(dataset['date'], utc=True) + pd.DateOffset(hours=input_hours_forecast)
    dates = dataset['date'].tolist()    
    return dates

def offSetCreator(dataYear, IPPOffset, input_hours_forecast):
    '''This function will be in charge of creating offsets for the given dataset and 
    replacing atp withthe real.'''


    # Adding in for efficieny so the function wont iterate through all of perfect prog data(NOT TESTED YET)
    if IPPOffset != 0.0:
        print('insideipp')
        #print(dataYear.head())
        # Loops through dataset in relevant column
        #print('# of rows;',len(dataYear))
        for i, row in dataYear.iterrows():
            
            # Grabs value from spot 
            value = dataYear.loc[i].at["Air Average"]

            # Updates Value
            if value != -999:
                dataYear.at[i, "Air Average"] = value + IPPOffset
                
        # # Loop to take care of predicted air temps depending on leadtime
        # print('a= ',input_hours_forecast,', b=',len(dataYear), ', c=',input_hours_forecast*len(dataYear))
        # for j in range(input_hours_forecast):
        #     for i, row in dataYear.iterrows():
            
        #         value2 = dataYear.loc[i].at["airTemperature_pred__" + str(j+1) + "h_forecast"]
                
        #         dataYear.at[i, "airTemperature_pred__" + str(j+1) + "h_forecast"] = value2 + IPPOffset

        # #print(dataYear.head())
    
    return dataYear

def dataframe_checker(checkNum, dfList):
    '''Checking for any number less than checkNum'''

    for df in dfList:
        if df.empty:
            print("Warning: One of the DataFrames is empty, skipping check.")
            continue
        # Check for any rogue number less than -999 in columns 2 and 3 of testing_data
        if (df.drop(columns=[df.columns[0]]) <  checkNum).any().any():
            exit("Error! DataFrame testing contains numbers lower than " + str(checkNum))


# NOTE: This module previously defined its own crps_loss as:
#     return tf.reduce_mean(tf.abs(y_pred - y_true))
# That is mean absolute error (MAE) math, NOT the Continuous Ranked
# Probability Score, despite the name. The real CRPS now lives in
# src/helper/losses.py and is re-exported here so existing imports keep
# working. Behavior change: callers now train against true CRPS, not MAE.
try:
    from src.helper.losses import crps_loss
except ImportError:
    from losses import crps_loss


def crps(y_true, y_pred):
    # Alias for crps_loss
    return crps_loss(y_true, y_pred)


# this is for debugging
if __name__ == "__main__":
    print("ayesha has been here")

    """ Manipulating data for AI Model """
    x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData("data/ESB_datasets",
                                                                                                                "descending",
                                                                                                                "cycle",
                                                                                                                12,
                                                                                                                24,
                                                                                                                24,
                                                                                                                1,
                                                                                                                cycle=1,
                                                                                                                model="MSE",
                                                                                                                verbose=0) # "model" variable only mattered for when we used lstm; lstm resuired a transofmration of dimensions of input shape
