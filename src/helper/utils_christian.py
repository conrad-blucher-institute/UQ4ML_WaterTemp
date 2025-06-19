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
def preparingData(path_to_data, input_structure, independent_year, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset = 0.0, cycle = 0, model="MLP"):
    '''preparingData() is the driver function'''
    # Importing libraries
    from datetime import datetime
    import pandas as pd

    # Function call to read the data
    data_year1, data_year2, data_year3, data_year4, data_year5, data_year6, data_year7, data_year8, data_year9, data_year10 = readingData(path_to_data)


    # with open('time_for_offsetcreator', 'w') as file:
    #     totaltime = end_time - start_time
    #     file.write(str(totaltime))
     # to alternate between the two independent years as testing years
    # if independent_year == 'cycle', then that means that we are doing the regular cycle year as testing
    if independent_year != 'cycle':

        if independent_year == '2021':
            data_independent_year = pd.read_csv("../UQ4ML_WaterTemp/data/June_May_Datasets/june_atp_and_wtp_2020_2021_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv")
        elif independent_year == '2024':
            data_independent_year = pd.read_csv("../UQ4ML_WaterTemp/data/June_May_Datasets/june_atp_and_wtp_2023_2024_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv")

        
        year_independent = creatingAdditionalColumns(data_independent_year, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    
    
    elif independent_year == 'cycle':
        year_independent = independent_year

    # Function call to create additional columns
    start_time = datetime.now()
    year1 = creatingAdditionalColumns(data_year1, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year2 = creatingAdditionalColumns(data_year2, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year3 = creatingAdditionalColumns(data_year3, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year4 = creatingAdditionalColumns(data_year4, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year5 = creatingAdditionalColumns(data_year5, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year6 = creatingAdditionalColumns(data_year6, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year7 = creatingAdditionalColumns(data_year7, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year8 = creatingAdditionalColumns(data_year8, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year9 = creatingAdditionalColumns(data_year9, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year10 = creatingAdditionalColumns(data_year10, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    print('finished input construction')
    end_time = datetime.now()
    
    #with open('time_for_inputconstruction', 'w') as file:
    #    totaltime = end_time - start_time
    #    file.write(str(totaltime) + ',\t\t\tlt,' + str(input_hours_forecast) + ',temp,' + str(IPPOffset) + ',cycle,' + str(cycle))

    # training_data, testing_data, validation_data = splittingData(IPPYear1, IPPYear2, IPPYear3, IPPYear4, IPPYear5, IPPYear6, IPPYear7, IPPYear8, IPPYear9, IPPYear10, cycle)
    # year2.to_csv('year2.csv')

    training_data, testing_data, validation_data = splittingData(year1,   year2,   year3,   year4,   year5,   year6,   year7,   year8,   year9,   year10, year_independent, cycle)
    # training_data.to_csv('training_data.csv')

    print('finished splitting the data')
    #print(training_data)

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
    training_dates = dateTimeRetriever(training, input_hours_forecast)
    validation_dates = dateTimeRetriever(validation, input_hours_forecast)
    testingDates = dateTimeRetriever(testing, input_hours_forecast)
    
    # To grab air temperatures
    #trainingAirTemps = testing['packeryATP_lighthouse'].tolist()
    #validationAirTemps = testing['packeryATP_lighthouse'].tolist()
    testingAirTemps = testing['packeryATP_lighthouse'].tolist()
    
    print()
    
    # Function call to reshpe the dataset and prepare it to be used as in input for the neural network
    x_train, y_train, x_val, y_val, x_test, y_test = reshaping(input_structure, training, testing, validation, model) 
    
    return x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAirTemps

def readingData(path_to_data):
    '''readingData() reads the data for each year'''

    import glob
    import pandas as pd
    import os
    import re

    csvs = glob.glob(f"{path_to_data}/*.csv")
    # print(csvs)
    print(f"Found {len(csvs)} CSV files in the directory.")

    pattern = re.compile(r'(\d{4})_(\d{4})')

    year_dict = {}

    for csv in csvs:
        match = pattern.search(csv)
        if match:
            year_range = f"{match.group(1)}_{match.group(2)}"  # e.g., "2022_2023"
            year_dict[year_range] = pd.read_csv(csv)  # Read CSV into DataFrame


    # csvs = glob.glob(f"{path_to_data}/*csv")
    data_year1 = year_dict['2022_2023']  # Read the first CSV into a DataFrame
    data_year2 = year_dict['2012_2013']  # Read the second CSV into a DataFrame
    data_year3 = year_dict['2013_2014']  # Read the third CSV into a DataFrame
    data_year4 = year_dict['2014_2015']  # Read the fourth CSV into a DataFrame
    data_year5 = year_dict['2015_2016']  # Read the fifth CSV into a DataFrame
    data_year6 = year_dict['2016_2017']  # Read the sixth CSV into a DataFrame
    data_year7 = year_dict['2017_2018']  # Read the seventh CSV into a DataFrame
    data_year8 = year_dict['2018_2019']  # Read the eighth CSV into a DataFrame
    data_year9 = year_dict['2019_2020']  # Read the ninth CSV into a DataFrame
    data_year10 = year_dict['2021_2022']  # Read the tenth CSV into a DataFramw
    
    return data_year1, data_year2, data_year3, data_year4, data_year5, data_year6, data_year7, data_year8, data_year9, data_year10

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
            temp = df['packeryATP_lighthouse'][w]
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
            temp = df['npsbiWTP_lighthouse'][w]     # Creating past columns
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
            temp = df['packeryATP_lighthouse'][w + i]    # Creating future columns
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
        temp = df['npsbiWTP_lighthouse'][w+(input_hours_forecast)] #-1
        con.append(temp)
    
    for k in range(input_hours_forecast):
        con.append(-999)
    
    df[name] = con
    
    
    # Delecting extra rows from the beginning
    df = df.iloc[120:]
    
    # Delecting extra rows from the end
    df = df.iloc[:-120] 
    
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

        if "npsbiWTP_lighthouse" in other_columns:
            other_columns.remove("npsbiWTP_lighthouse")  
        if "packeryATP_lighthouse" in other_columns:
            other_columns.remove("packeryATP_lighthouse")  

        # Reorder columns
        reordered_columns = (
            other_columns
            + water_temp_columns[::-1]
            + ["npsbiWTP_lighthouse"]
            + air_temp_columns[::-1]
            + ["packeryATP_lighthouse"]
            + forecast_columns
        )

        df = df[reordered_columns]
        return df

    elif input_structure == "ascending":
        return df

def splittingData(year1, year2, year3, year4, year5, year6, year7, year8, year9, year10, year_independent, cycle):
    '''splittingData() groups the data into training, testing, and validation
    --Will rotate through the years as the cycle changes'''
    import pandas as pd

    yearList = [year1,year2,year3,year4,year5,year6,year7,year8,year9,year10]

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
                if isinstance(year_independent, pd.DataFrame):
                    print("USING INDEPENDENT TEST YEAR")
                    testing = pd.concat([testing, year_independent])
                elif year_independent == "cycle":
                    print("USING REGULAR CYCLE TESTING")
                    testing = pd.concat([testing, yearList[index]])


                
        if j == cycle:
            #print(testing)
            return training, testing, validation
        
        ### reset lists to empty
        training = pd.DataFrame()
        testing = pd.DataFrame()
        validation = pd.DataFrame()
    
def countingMissingValues(df):
    '''countingMissingValues() counting the rows that contains a missing value in at least one of the columns'''

    missing_standard = df.isna().any(axis=1)
    missing_custom = df.isin([-999]).any(axis=1)
    missingValues = df[missing_standard | missing_custom]
    
    numMissValues = len(missingValues)
    percMissValues = (numMissValues/len(df))*100

    return numMissValues, percMissValues

def deletingMissingValues(df):
    '''deletingMissingValues() deleting the rows that at least one of the columns contain a missing value'''

    valueRemove = [-999]   
    df = df[df.isin(valueRemove) == False]
    df = df.dropna()
    
    return df

def reshaping(input_structure, training, testing, validation, model):
    '''reshaping() reshaping the training, testing, and validation datasets to be 
    able to use them as an input for the AI model'''
    import numpy as np
    
    if input_structure == "descending":
        input_column_start = 1
    elif input_structure == "ascending":
        input_column_start = 3

    #print(testing['dateAndTime', 'packeryATP_lighthouse', 'npsbiWTP_lighthouse'].head(125))
    # Dividing the datasets between the inputs and the target
    trainingData = training.iloc[:,input_column_start:-1].values.astype(float) 
    trainingTarget = training.iloc[:,-1].values.astype(float)
    
    # print(trainingData)
    # print(trainingTarget)
        
    validationData = validation.iloc[:,input_column_start:-1].values.astype(float) 
    validationTarget = validation.iloc[:,-1].values.astype(float) 

    testingData = testing.iloc[:,input_column_start:-1].values.astype(float) 
    testingTarget = testing.iloc[:,-1].values.astype(float) 
    
    if(model == "LSTM"):
        # Reshaping the datasets
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

    import pandas as pd
    import pandas as pd
    # holds date times
    dates = []
    
    #print(testing.head())import pandas as pd
    #print(testing.head())import pandas as pd
    
    # Loop to add date times for future calculations
    dataset['dateAndTime'] = pd.to_datetime(dataset['dateAndTime'], format='%m-%d-%Y %H%M', yearfirst=False) + pd.DateOffset(hours=input_hours_forecast)
    
    #print(testing['dateAndTime'].head())
    
    # Ask if an offset is needed for the air temps
    
    #Converts series into a list
    dates = dataset['dateAndTime'].tolist()    
            
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
            value = dataYear.loc[i].at["packeryATP_lighthouse"]

            # Updates Value
            if value != -999:
                dataYear.at[i, "packeryATP_lighthouse"] = value + IPPOffset
                
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
        # Check for any rogue number less than -999 in columns 2 and 3 of testing_data
        if (df.drop(columns=[df.columns[0]]) <  checkNum).any().any():
            exit("Error! DataFrame testing contains numbers lower than " + str( checkNum))



""" METRICS / LOSS FUNCTIONS """ 
def crps(y_true, y_pred): 
    """ From Ryan Lagerquist... 

    Calculates the Continuous Ranked Probability Score (CRPS) 
    for finite ensemble members and a single target. 
    
    This implementation is based on the identity: 
        CRPS(F, x) = E_F|y_pred - y_true| - 1/2 * E_F|y_pred - y_pred'| 
    where y_pred and y_pred' denote independent random variables drawn from 
    the predicted distribution F, and E_F denotes the expectation 
    value under F. 
    
    Following the approach by Steven Brey at  
    TheClimateCorporation (formerly ClimateLLC) 
    https://github.com/TheClimateCorporation/properscoring 
    
    Adapted from David Blei's lab at Columbia University 
    http://www.cs.columbia.edu/~blei/ and 
    https://github.com/blei-lab/edward/pull/922/files 
    
    
    References 
    --------- 
    Tilmann Gneiting and Adrian E. Raftery (2005). 
        Strictly proper scoring rules, prediction, and estimation. 
        University of Washington Department of Statistics Technical 
        Report no. 463R. 
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf 
    
    H. Hersbach (2000). 
        Decomposition of the Continuous Ranked Probability Score 
        for Ensemble Prediction Systems. 
        https://doi.org/10.1175/1520-0434(2000)015%3C0559:DOTCRP%3E2.0.CO;2 
    """ 
 
    import tensorflow as tf 
    
    # Variable names below reference equation terms in docstring above 
    term_one = tf.reduce_mean(tf.abs( 
        tf.subtract(y_true, y_pred)), axis=-1) 
    
    term_two = tf.reduce_mean( 
        tf.abs( 
            tf.subtract(tf.expand_dims(y_pred, -1), 
                        tf.expand_dims(y_pred, -2))), 
        axis=(-2, -1)) 
    
    half = tf.constant(-0.5, dtype=term_two.dtype) 
    
    score = tf.add(term_one, tf.multiply(half, term_two)) 
    
    score = tf.reduce_mean(score) 
    
    return score

def mae(y_true, y_pred): 
    import tensorflow as tf 

    if y_pred.shape[1] > 1: # shape[1] > 1 means that it has multiple outputs  
        #y_true = tf.expand_dims(y_true, axis=-1) 
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1) 
    else: 
        mean_pred = y_pred 

    differences = tf.abs(tf.subtract(y_true, mean_pred)) 
    
    score = tf.reduce_mean(differences) 
    
    return score.numpy() 
  
def mae12(y_true, y_pred):
    import tensorflow as tf 
    # If there are multiple ensemble outputs, compute ensemble mean and take one target value.
    if y_pred.shape[1] > 1:
        # Compute the ensemble mean over the last axis, keeping dims so shape becomes (batch, 1)
        mean_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        # Since y_true is repeated, just take the first column (shape becomes (batch, 1))
        true_val = y_true[:, :1]
    else:
        mean_pred = y_pred 
        true_val = y_true 

    # Create mask based on the target values (now shape (batch, 1))
    mask = true_val < 12 
    filtered_y_true = tf.boolean_mask(true_val, mask)
    filtered_y_pred_mean = tf.boolean_mask(mean_pred, mask)
    
    differences = tf.abs(filtered_y_true - filtered_y_pred_mean)
    score = tf.reduce_mean(differences).numpy()
    return score

def mse(y_true, y_pred): 
    import tensorflow as tf 


    if y_pred.shape[1] > 1: # shape[1] > 1 means that it has multiple outputs 
        #y_true = tf.expand_dims(y_true, axis=-1) 
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1)
    else: 
        mean_pred = y_pred 
    
    mean_square = tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred)), axis=-1) 
    
    score = tf.reduce_mean(mean_square) 
    
    return score.numpy() 
  
def me(y_true, y_pred): 
    import tensorflow as tf 


    if y_pred.shape[1] > 1: # shape[1] > 1 means that it has multiple outputs 
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1)
    else: 
        mean_pred = y_pred 
    
    mean_square = tf.reduce_mean((tf.subtract(y_true, mean_pred)), axis=-1) 
    
    score = tf.reduce_mean(mean_square) 
    
    return score.numpy()
  
def rmse(y_true, y_pred): 
    import tensorflow as tf 

    if y_pred.shape[1] > 1: ## shape[1] > 1 means that it has multiple outputs 
        #y_true = tf.expand_dims(y_true, axis=-1) 
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1) 
    else: 
        mean_pred = y_pred 
    
    root_mean_square = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred)), axis=-1)) 
    
    
    score = tf.reduce_mean(root_mean_square) 
    
    return score.numpy()

def y_pred_std(y_true, y_pred):
    import tensorflow as tf
    return tf.math.reduce_mean(tf.math.reduce_std(y_pred, axis=-1)).numpy()

#Spread Skill Ratio; SSRAT 
def ssrat(y_true, y_pred): 
    import tensorflow as tf 

    y_pred_std = tf.math.reduce_std(y_pred, axis=-1) 
    
    ssrat_score = tf.math.reduce_mean(y_pred_std)/rmse(y_true, y_pred)

    return ssrat_score.numpy()

# PITD: Probability Integral Transgform Distance 
def pitd(y_true, y_pred):
    import numpy as np

    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()

    pit_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nBins = len(pit_bins) - 1
    nEns = y_pred.shape[-1]
    nSamples = y_true.shape[0]

    ytrueT = y_true.reshape(-1)
    ypredT = y_pred.reshape((nSamples, nEns))
    ypredTS = np.sort(ypredT, axis=1)

    ytrueTE = np.repeat(
      ytrueT[..., np.newaxis], nEns, axis=-1)
    pred_diff = np.abs(np.subtract(ytrueTE, ypredTS))
    pit_values = np.divide(np.argmin(pred_diff, axis=-1), nEns)
    weights = np.ones_like(pit_values) / nSamples

    def get_histogram(var, bins=10, density=False, weights=None):
        counts, bin_edges = np.histogram(
            var, bins=bins, density=density, weights=weights)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return counts, bin_centers
        
    pit_counts, bin_centers = get_histogram(\
        pit_values, bins=pit_bins, weights=weights)

    def get_pit_dvalue(pit_counts):
        dvalue = 0.
        nbins = pit_counts.shape[0]
        nbinsI = 1./nbins

        pitTot = np.sum(pit_counts)
        pit_freq = np.divide(pit_counts, pitTot)
        for i in range(nbins):
            dvalue += (pit_freq[i] - nbinsI) * (pit_freq[i] - nbinsI)
        dvalue = np.sqrt(dvalue/nbins)
        return dvalue

    pitd_score = get_pit_dvalue(pit_counts)

    return pitd_score



# Continuous Rank Probability Score Loss Function
def crps_loss(y_true, y_pred):
    """
    From Ryan Lagerquist...

    Calculates the Continuous Ranked Probability Score (CRPS)
    for finite ensemble members and a single target.
    
    This implementation is based on the identity:
        CRPS(F, x) = E_F|y_pred - y_true| - 1/2 * E_F|y_pred - y_pred'|
    where y_pred and y_pred' denote independent random variables drawn from
    the predicted distribution F, and E_F denotes the expectation
    value under F.

    Following the approach by Steven Brey at 
    TheClimateCorporation (formerly ClimateLLC)
    https://github.com/TheClimateCorporation/properscoring
    
    Adapted from David Blei's lab at Columbia University
    http://www.cs.columbia.edu/~blei/ and
    https://github.com/blei-lab/edward/pull/922/files

    
    References
    ---------
    Tilmann Gneiting and Adrian E. Raftery (2005).
        Strictly proper scoring rules, prediction, and estimation.
        University of Washington Department of Statistics Technical
        Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    
    H. Hersbach (2000).
        Decomposition of the Continuous Ranked Probability Score
        for Ensemble Prediction Systems.
        https://doi.org/10.1175/1520-0434(2000)015%3C0559:DOTCRP%3E2.0.CO;2
    """

    import tensorflow as tf

    # Variable names below reference equation terms in docstring above
    term_one = tf.reduce_mean(tf.abs(
        tf.subtract(y_true, y_pred)), axis=-1)
    
    term_two = tf.reduce_mean(
        tf.abs(
            tf.subtract(tf.expand_dims(y_pred, -1),
                        tf.expand_dims(y_pred, -2))),
        axis=(-2, -1))
    
    half = tf.constant(-0.5, dtype=term_two.dtype)

    score = tf.add(term_one, tf.multiply(half, term_two))
    
    score = tf.reduce_mean(score)

    return score
