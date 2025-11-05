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

""" CREATING INPUT VECTOR """
# MAIN function to create the input vector for the ml model
def preparingData(path_to_data, input_structure, independent_year, input_hours_forecast, atp_hours_back, 
                  wtp_hours_back, pred_atp_interval, IPPOffset = 0.0, cycle = 0, model="MLP", verbose=0):
    
    '''preparingData() is the driver function'''
    # Importing libraries
    from datetime import datetime
    import pandas as pd
    import numpy as np

    # Function call to read the data
    data_year2, data_year3, data_year4, data_year5 = readingData(path_to_data)

    if verbose == 3:
        for i, v in enumerate([data_year2, data_year3, data_year4, data_year5]):
            print(v.columns)

    # with open('time_for_offsetcreator', 'w') as file:
    #     totaltime = end_time - start_time
    #     file.write(str(totaltime))
     # to alternate between the two independent years as testing years
    # if independent_year == 'cycle', then that means that we are doing the regular cycle year as testing
    # if independent_year != 'cycle':

    #     if independent_year == '2021':
    #         data_independent_year = pd.read_csv("../UQ4ML_WaterTemp/data/June_May_Datasets/june_atp_and_wtp_2020_2021_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv")
    #     elif independent_year == '2024':
    #         data_independent_year = pd.read_csv("../UQ4ML_WaterTemp/data/June_May_Datasets/june_atp_and_wtp_2023_2024_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv")

        
    #     year_independent = creatingAdditionalColumns(data_independent_year, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    
    
    # elif independent_year == 'cycle':
        # year_independent = independent_year

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
    # training_data, testing_data, validation_data = splittingData(data_year1, data_year2, data_year3, data_year4, data_year5, year_independent, cycle)
    training_data, testing_data, validation_data = splittingData(data_year2, data_year3, data_year4, data_year5, cycle)
    # training_data.to_csv('training_data.csv')

    print('finished splitting the data')
    #print(training_data)

    print("Testing data shape:", testing_data.shape)
    print("train data shape:", training_data.shape)
    print("val data shape:", validation_data.shape)

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
    training = deletingMissingValues(training_data)
    testing = deletingMissingValues(testing_data)
    validation = deletingMissingValues(validation_data)

    dataframe_checker(-100, [training, testing, validation]) # checking for any rogue number less than -100
    

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

'''  
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
    '''creatingAdditionalColumns() creating columns for the past and future (perfect prog) hours'''
    
    interval = pred_atp_interval

    # Creating past air temperature values
    s = 'airTemperature__'
    s3 = 'h_ago'

    for i in range(atp_hours_back): 

        j = i + 1
        con = s + str(j) 
        con = []

        name = s + str(j) + s3

        for k in range(j):  # Creating missing values rows for the previous five days
            con.append(-999)

        for w in range(len(df)-(j)):   # Creating past columns
            temp = df['Air Average'][w]
            con.append(temp)  

        df[name] = con
        
    # Creating past water temperature values
    x1 = 'waterTemperature__'
    x3 = 'h_ago'

    for i in range(wtp_hours_back): 

        j = i + 1
        con = x1 + str(j) 
        con = []

        name = x1 + str(j) + x3

        for k in range(j):    # Creating missing values rows for the previous five days
            con.append(-999)

        for w in range(len(df)-(j)):
            temp = df['Water Average'][w]     # Creating past columns
            con.append(temp)  

        df[name] = con
        
    df = offSetCreator(df,IPPOffset, input_hours_forecast)

    # Predicted air temperature
    xPred = 'airTemperature_pred__'
    xPred3 = 'h_forecast'
        
    for i in range(interval, input_hours_forecast+1, interval): 

        j = i
        con = xPred + str(i) 
        con = []

        name = xPred + str(j) + xPred3

        for w in range(len(df) - (i)):   
            temp = df['Air Average'][w + i]    # Creating future columns
            con.append(temp)  

        for k in range(i):      # Creating missing values rows for the next five days
            con.append(-999)

        df[name] = con
    
        
    # Creating a list used to create the future columns
    kList_wtp = []
   
    for z in range(input_hours_forecast):
        
        temp_wtp = (z + 1)
        kList_wtp.append(temp_wtp)
        
        if(z == input_hours_forecast - 1):
            value_wtp = temp_wtp
               
    # Creating target
    t = 'waterTemperature_' + str(input_hours_forecast) + 'h_forecast'

    con = t + str(j)
    con = []

    name = t 

    for w in range(len(df) - (input_hours_forecast)):
        temp = df['Water Average'][w+(input_hours_forecast)] #-1
        con.append(temp)
    
    for k in range(input_hours_forecast):
        con.append(-999)
    
    df[name] = con
    
    
    # Delecting extra rows from the beginning
    # df = df.iloc[120:]
    
    # Delecting extra rows from the end
    # df = df.iloc[:-120] 
    
    if input_structure == "descending":
        
        # begining of changing the order of the input vector 
        # we want the input vector to look like this below
        # wtp_3h_ago, wtp_2h_ago, wtp_1h_ago, current_wtp, atp_3h_ago, atp_2h_ago, atp_1h_ago, current_atp, atp_1h_forecast, atp_2h_forecast, atp_3h_forecast, target_wtp_3h_forecast

        # separating the columns (water temperature xh_ago, air temperature xh_ago, and air temperature xh_forecast)
        water_temp_columns = [col for col in df.columns if "waterTemperature__" in col]
        air_temp_columns = [col for col in df.columns if "airTemperature__" in col and "_ago" in col]
        forecast_columns = [col for col in df.columns if "forecast" in col]

        
        # the datetime, current wtp and current atp
        other_columns = [col for col in df.columns if col not in water_temp_columns + air_temp_columns + forecast_columns]
        # removing current wtp and atp from list to be added back in the appropriate location

        if "Water Average" in other_columns:
            other_columns.remove("Water Average")  
        if "Air Average" in other_columns:
            other_columns.remove("Air Average")  

        # Reorder columns
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


def splittingData(year2, year3, year4, year5, cycle):
    '''splittingData() groups the data into training, testing, and validation
    --Will rotate through the years as the cycle changes'''
    import pandas as pd

    yearList = [year2,year3,year4,year5]

    training = pd.DataFrame()
    testing = pd.DataFrame()
    validation = pd.DataFrame()
    #"""
    
    # loop until we get to the version we are tring to make
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
    
def og_splittingData(year2, year3, year4, year5, year_independent, cycle):
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
    
def countingMissingValues(df):
    '''countingMissingValues() counting the rows that contains a missing value in at least one of the columns'''

    numMissValues = df.isnull().sum().sum()
    if len(df) == 0:
        return numMissValues, 0.0
    
    percMissValues = (numMissValues/len(df))*100
    return numMissValues, percMissValues

    missing_standard = df.isna().any(axis=1)
    missing_custom = df.isin([-999]).any(axis=1)
    missingValues = df[missing_standard | missing_custom]
    
    numMissValues = len(missingValues)
    percMissValues = (numMissValues/len(df))*100

    return numMissValues, percMissValues

def deletingMissingValues(df):
    '''deletingMissingValues() deleting the rows that at least one of the columns contain a missing value'''
    valueRemove = [-999]
    # Exclude 'date' column from missing value checks
    cols_to_check = [col for col in df.columns if col != 'date']
    mask = (df[cols_to_check].isin(valueRemove)).any(axis=1) | df[cols_to_check].isna().any(axis=1)
    df = df[~mask]
    df = df.dropna(subset=cols_to_check)
    return df

def reshaping(input_structure, training, testing, validation, model):
    '''reshaping() reshaping the training, testing, and validation datasets to be 
    able to use them as an input for the AI model'''
    import numpy as np

    if input_structure == "descending":
        input_column_start = 1
    elif input_structure == "ascending":
        input_column_start = 3

    # Handle empty DataFrames
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


def crps_loss(y_true, y_pred):
    import tensorflow as tf
    return tf.reduce_mean(tf.abs(y_pred - y_true))


def crps(y_true, y_pred):
    # Alias for crps_loss
    return crps_loss(y_true, y_pred)


if __name__ == "__main__":
    print("ayesha has been here")
    """ Manipulating data for AI Model """
    x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData("data/June_May_Datasets",
                                                                                                                "descending",
                                                                                                                "cycle",
                                                                                                                12,
                                                                                                                24,
                                                                                                                24,
                                                                                                                1,
                                                                                                                cycle=1,
                                                                                                                model="MSE",
                                                                                                                verbose=0) # "model" variable only mattered for when we used lstm; lstm resuired a transofmration of dimensions of input shape
