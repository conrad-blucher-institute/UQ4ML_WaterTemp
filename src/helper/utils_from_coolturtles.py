"""
This file is for the use of the Coastal Dynamics Laboratory

containing helper functions for:

    - Creating Visuals (Graphing etc.) 
    - Calculating Metrics
    - Preparing Data

    
Organized by: Christian Duff
"""

'''
Custom Loss Functions Begin
'''


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


#experimental custom loss function
def weighted_mse(CS_weight):
    # Importing libraries     
    import keras.backend as K

    def loss(target, prediction):
        min_temp = K.minimum(K.abs(target), K.abs(prediction))

        # Shape is really appealing, but doesn't seem to work.
        # CS_weight can't be negative
        #se_weight = 1/(min_temp)**CS_weight
        
        # Works for CS_weight in [0, 5]
        # 0 is MSE
        se_weight = (40 - min_temp)**CS_weight
        
        se = K.abs(prediction - target) ** 2
        weighted_se = se_weight * se
        weighted_mse = K.mean(weighted_se)
        return weighted_mse
    return loss

'''
Custom Loss Functions End
'''

'''
Machine Learning Model(s) Architectures Begin

Author: Marina Vicens Miquel

Modified by: Jarett T. Woodall

'''



#### from weather_company_functions import utcDateTimeConverter



# Creating a class where the AI architecture is defined
class mlp_model():
    # Importing libraries 
    from  tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    from tensorflow import keras
    import tensorflow.keras.backend as K
    # Initializing the arguments passed
    def __init__(self, inputShape):
        self.inputShape = inputShape  
        
    # Defining the AI architecture and returning the model
    def model(self, input_units, hidden_layers = 1, weight = 0):
        # Importing libraries
        from keras.layers import Input, Dense
        from keras.models import Model

        # Input layer
        dataInput = Input(self.inputShape)

        # First Hidden Layer
        x = Dense(input_units, activation='selu', kernel_regularizer='l2')(dataInput)
        
        # Output layer
        output = Dense(1)(x)
        
        # Creates the model
        model = Model(inputs=[dataInput], outputs=output)
        model.summary()  # Print the summary of the neural network
        #model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.01), loss = weighted_mse(CS_weight=weight)) #experiment with 1, 2, .5
        
        return model

class mme_crps_model():
    # Initializing the arguments passed
    def __init__(self, inputShape):
        self.inputShape = inputShape  
        
    # Defining the AI architecture and returning the model
    def model(self, input_units, output_units, activation, regularizer):
        # Importing libraries
        from keras.layers import Input, Dense
        from keras.models import Model

        # Input layer
        dataInput = Input(self.inputShape)

        # First Hidden Layer
        x = Dense(input_units, activation=activation, kernel_regularizer=regularizer)(dataInput)
        
        # Output layer
        output = Dense(output_units)(x)
        
        # Creates the model
        model = Model(inputs=[dataInput], outputs=output)
        model.summary()  # Print the summary of the neural network
        #model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.01), loss = weighted_mse(CS_weight=weight)) #experiment with 1, 2, .5
        
        return model

'''Machine Learning Model(s) Architectures End'''



''' 
Data Preparation Begin


Author: Marina Vicens Miquel

Modified by: Hector Marrero-Colominas, Jarett T. Woodall, Christian Duff


The goal of this code is to prepare the cold stunning dataset to be used as input for the AI model:
    1) Read the dataset
    2) Creating additional columns (past values and perfect prog)
    3) Splitting the data between training, testing, and validation
    4) Counting the missing values
    5) Deleting the missing values
    6) Reshaping the dataset

'''


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

        if independent_year == '21':
            data_independent_year = pd.read_csv("../June_May_Datasets/june_atp_and_wtp_2020_2021_withExtraRows_INDEPENDENT_TEST_YEAR_MW.csv")
        elif independent_year == '24':
            data_independent_year = pd.read_csv("../June_May_Datasets/june_atp_and_wtp_2023_2024_withExtraRows_INDEPENDENT_TEST_YEAR_MW.csv")

        
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


''' Data Preparation End'''




'''Evaluation Begins

Author: Marina Vicens Miquel

Modified by: Hector Marrero-Colominas, Jarett T. Woodall, Christian Duff

'''


'''Evaluation Ends'''



'''Ensemble Visuals Begins

Author: Jarett T. Woodall

Modified by: Hector Marrero-Colominas, Christian Duff


The goal of this code is to graph ensemble visuals:
    1) This fille contains a function to graoh visuals
    2) This file contains a function to swap paths
    3) This file also contains a function to swap and augment data frames
    4) DATAFRAME SWITCHER FUNCTION MAY NEED OPTIMIZED WITH A ITERATER STRUCTURE
        IN ORDER TO REDUCE HARD CODING IN VALUES.
    
    ***NOTE: if assignment statements seem extraneous in figure creation, this
    was done intentially due to parameter conflicts.***

'''



''' Ensemble Visuals End'''



''' Ensemble Attribute Diagram Visuals Begins

Author(s): Christian Duff


GOAL: automate the process of attribute diagram graphing from ensemble  results
(automate the "visual statistics" as well ; aka the bar graphs)
once automated, move to "utils" in refactoring branch

- Turn median.ipynb into function (done)
- Read in trial vs prediction text file into dataframe 
- Remove last two columns
- Add median predicted value as third column
- run through the graphing process

reference "individual" from threshold calc to get the median of each member (15 total)

'''




def median(ensemble_visuals_all_path, target_vs_prediction):
    import pandas as pd
    import numpy as np

    """Gets the median values of desired model"""

    import pandas as pd
    import numpy as np

    target_vs_prediction = pd.read_csv(target_vs_prediction)

    target_vs_prediction = target_vs_prediction.iloc[:, :-2]

    members=list(np.arange(-3.5, 4.0, 0.5))
    individual_members = []
    individual_median = []
    df_all_members = pd.read_csv(ensemble_visuals_all_path)

    columns_to_remove = ['dateAndTime']  # List of column names to remove
    df_all_members = df_all_members.drop(columns=columns_to_remove)

    cold_members = df_all_members.loc[:, df_all_members.columns.str.startswith('-')]             # Extract negative columns
    perfprog_members = df_all_members.loc[:, df_all_members.columns.str.startswith('0.0')]       # Extract perfectProg column
    hot_members = df_all_members.loc[:, ~df_all_members.columns.str.startswith(('-', '0.0'))]    # Extract positive columns

    cold_median = cold_members.apply(np.median, axis=1)
    hot_median = hot_members.apply(np.median, axis=1)
    perf_median = perfprog_members.apply(np.median, axis=1)
    all_median = df_all_members.apply(np.median, axis=1)

    for i in range(len(members)):
        individual_members.append(df_all_members.loc[:, df_all_members.columns.str.startswith(str(members[i]))])
        individual_median.append(individual_members[i].apply(np.median, axis=1))
        
        df = target_vs_prediction.copy()
        df['Predicted Value'] = individual_median[i]

        individual_median[i] = df

    df = target_vs_prediction.copy()
    df['Predicted Value'] = cold_median
    cold_median = df

    df = target_vs_prediction.copy()
    df['Predicted Value'] = hot_median
    hot_median = df

    df = target_vs_prediction.copy()
    df['Predicted Value'] = perf_median
    perf_median = df

    df = target_vs_prediction.copy()
    df['Predicted Value'] = all_median
    all_median = df

    return cold_median, hot_median, perf_median, all_median, individual_median


def attribute_diagram_prep(median):
    """prep for attribute diagram graphing"""
    median = median.sort_values(by = 'Predicted Value', ascending=True)
    grouped = median.copy()

    grouped[' Target Value'] = grouped[' Target Value'].round(decimals=0)
    grouped[' Target Value'] = grouped[' Target Value'].add(grouped[' Target Value'].mod(2))
    grouped.rename(columns={'Predicted Value': 'Predictions'}, inplace=True)
    grouped = grouped.groupby(' Target Value').Predictions.mean().sort_values()

    return median, grouped


def bar_graphs(path , cycle, lead_times, num_trials, roc):
    import os
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np
    import matplotlib.pyplot as plt
    from math import sqrt

    import os
    import numpy as np
    import pandas as pd
    from math import sqrt
    import pandas as pd
    import matplotlib.pyplot as plt


    import os
    import numpy as np
    import pandas as pd
    from math import sqrt
    import pandas as pd
    import matplotlib.pyplot as plt


    folder = f"cycle_{cycle}_bar_graphs/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    members = list(np.arange(-3.5, 4.0, 0.5))
    
    metrics = ['ME', 'MAE', 'MAEx10', 'ME<12', 'MAE<12'] # List of evaluation metrics

    # Function to compute statistics (mean, median, std, 2se, IQR)
    def compute_statistics(data, trials):
        return pd.Series({'Mean': data.mean(),
                        'Median': data.median(),
                        'Std': data.std(),
                        '2SE': 2*(data.std()/sqrt(trials)),
                        'IQR': data.quantile(0.75) - data.quantile(0.25)})

    """ Per Lead Time """
    # Create a new DataFrame to store the computed statistics
    stats = pd.DataFrame(columns=['cycle', 'Metric', 'Mean', 'Median', 'Std', '2SE', 'IQR'])

    dfs = []
    
    for i, leadtime in enumerate(lead_times):
        csv_path = f'{path}/{leadtime}h_{roc}_results/whole_{leadtime}h_METRICS.csv' # path to "whole METRICS csv file"
        dfs.append(pd.read_csv(csv_path))
        for metric in metrics:
            metric_data = dfs[i][metric]
            trials = num_trials
            statistics = compute_statistics(metric_data, trials)
            new_row = pd.Series({'leadtime': leadtime, 'cycle': cycle, 'Metric': metric, **statistics})
            stats = pd.concat([stats, new_row.to_frame().T], ignore_index=True)

    # Function to plot metric bar graphs for each leadtime
    def leadtime_metric_bargraph(stats, metrics, lead_times):

        for m, metric in enumerate(metrics):
            plt.figure(figsize=(14, 8))
            plt.title(f'Cycle {cycle} {metric} - Mean ± 2SE', fontsize="32") 
            plt.xlabel('Lead Time', fontsize="25")
            plt.ylabel(f'{metric} (°C)', fontsize="25")

            for leadtime in lead_times:
                df = stats.loc[stats['leadtime'] == leadtime]
                mean = df[df['Metric'] == metric]['Mean'].values[0]
                se = df[df['Metric'] == metric]['2SE'].values[0]
                
                plt.bar(str(leadtime), mean, yerr=se, capsize=5, align='center', alpha=0.7, label=str(leadtime))
                
           
            if m == 0:
                m = "ME"
            if m == 1:
                m = "MAE"
            if m == 2:
                m = "MAE10"
            if m == 3:
                m = "ME12"
            if m == 4:
                m = "MAE12"
                
            plt.tick_params(axis='y', labelsize="17")
            plt.tick_params(axis='x', labelsize="17")
            plt.legend(fontsize="16")
            fig = plt.gcf()
            fig.savefig(f'{folder}cycle_{cycle}_{m}_{roc}_lead_time_statistics_bargraphs.jpeg', bbox_inches='tight')
            plt.close()
            #plt.show() 
          

    leadtime_metric_bargraph(stats, metrics, lead_times)

    """ Per Member """

    # Create a new DataFrame to store the computed statistics
    stats = pd.DataFrame(columns=['member', 'Metric', 'Mean', 'Median', 'Std', '2SE', 'IQR'])

    for i, leadtime in enumerate(lead_times):
        csv_path = f'{path}/{leadtime}h_{roc}_results/whole_{leadtime}h_METRICS.csv' # path to "whole METRICS csv file"
        dfs.append(pd.read_csv(csv_path))
        # Group the DataFrame by 'member' and 'Metric' and calculate statistics for each group
        for member in members:
            trials = (num_trials/15)

            start_of_member = dfs[i][dfs[i]['temp'] == member].index[0]
            end_of_member = dfs[i][dfs[i]['temp'] == member].index[-1]


            for metric in metrics:
                metric_data = dfs[i][metric].iloc[start_of_member:end_of_member+1]
                statistics = compute_statistics(metric_data, trials)        
                new_row = pd.Series({'leadtime': leadtime, 'member': member, 'Metric': metric, **statistics})
                stats = pd.concat([stats, new_row.to_frame().T], ignore_index=True)

    # Function to plot metric bar graphs for each member
    def member_metric_bargraph(stats, metrics, lead_times):
        for m, metric in enumerate(metrics):
            plt.figure(figsize=(14, 8))
            plt.title(f'Cycle {cycle} {metric} - Mean ± 2SE', fontsize="32") 
            plt.xlabel('Lead Time - Member', fontsize="25")
            plt.ylabel(f'{metric} (°C)', fontsize="25")

            for j, lead_time in enumerate(lead_times):
                member_means = []
                member_std_errors = []

                for member in members:
                    df = stats.loc[stats['leadtime'] == lead_time]
                    mean = df[(df['Metric'] == metric) & (df['member'] == member)]['Mean'].values[0]
                    se = df[(df['Metric'] == metric) & (df['member'] == member)]['2SE'].values[0]

                    member_means.append(mean)
                    member_std_errors.append(se)

                x = [f'{lead_time}-member{str(member)}' for member in members]
                plt.bar(x, member_means, yerr=member_std_errors, capsize=5, align='center', alpha=0.7, label=str(lead_time))

            if m == 0:
                m = "ME"
            if m == 1:
                m = "MAE"
            if m == 2:
                m = "MAE10"
            if m == 3:
                m = "ME12"
            if m == 4:
                m = "MAE12"

            plt.tick_params(axis='y', labelsize="16")
            plt.xticks(np.arange(0, 15, step=1), labels=[f'{member}' for member in members],fontsize="7")
            plt.legend(fontsize="22")
            fig = plt.gcf()
            fig.savefig(f'{folder}cycle_{cycle}_{m}_{roc}_member_statistics_bargraphs.jpeg', bbox_inches='tight')  
            #plt.show()
            plt.close()
         
            
    member_metric_bargraph(stats, metrics, lead_times)
    
''' Ensemble Attribute Diagram Visuals End'''


''' Ensemble time series grapher functions begins'''


def calcThreshold_of_CS_event(path, datetime_split):
    '''Calculates the Threshold of the beginning and ending of the cold stunning event
        Author: Hector M. Marrero-Colominas'''

    import pandas as pd
    # load in the dataset (csv) to a dataframe
    ensemble_df = pd.read_csv(path, index_col='dateAndTime', parse_dates=True)
    if(datetime_split == "dec"):
        ensemble_df = ensemble_df.iloc[7660:] # for december  
    
    if(datetime_split == "jan-mar"):
        ensemble_df = ensemble_df.iloc[:2070] # for jan-march

    # Columns to remove
    columns_to_remove = ['Target']#, '3.5', '-3.5']  # List of column names to remove
    ensemble_df = ensemble_df.drop(columns=columns_to_remove)


    startIndex = {}
    endingIndex = {}

    # print(ensemble_df.head(15))

    # Iterate over each column and find the start values
    for column in ensemble_df.columns:
        if column == "dateAndTime": continue
        # print("columns:", column)
        col_values = ensemble_df[column].tolist()  # Convert column to a list
        # print(col_values)

        start_index = None
        for i, value in enumerate(col_values):
            if float(value) < 8:
                if start_index is None:
                    start_index = i
                    startIndex[str(column)] = i
        # start_index = ensemble_df[column].apply(lambda x: x < 8).idxmax()

        end_index = None
        for i, value in enumerate(col_values[startIndex[column]:]):
            if float(value) > 8 and float(value) < 8.5:
                if end_index is None and i > 24:
                    end_index = i
                    endingIndex[str(column)] = i + startIndex[column]
        
                


    dateAndTime = ensemble_df.index


    # complicated line to transform the dateAndTime string to something nicer to read
    startMin = pd.to_datetime(dateAndTime[min(list(startIndex.values()))]).strftime("%m-%d-%Y %H%M")
    startMax = pd.to_datetime(dateAndTime[max(list(startIndex.values()))]).strftime("%m-%d-%Y %H%M")
    endMin = pd.to_datetime(dateAndTime[min(list(endingIndex.values()))]).strftime("%m-%d-%Y %H%M")
    endMax = pd.to_datetime(dateAndTime[max(list(endingIndex.values()))]).strftime("%m-%d-%Y %H%M")
    
    start_threshold = startMin, startMax
    end_threshold = endMin, endMax
    print("start ", start_threshold, max(list(startIndex.values())) - min(list(startIndex.values())), "hours range")
    print("end ", end_threshold, max(list(endingIndex.values())) - min(list(endingIndex.values())), "hours range")

    
    #with open('ensemble_threshold', 'w') as file:
    #    file.write("start " + str(start_threshold) +' '+ str(max(list(startIndex.values())) - min(list(startIndex.values()))) + " hours range")
    #    file.write('\n')
    #    file.write("end " + str(end_threshold) +' '+ str(max(list(endingIndex.values())) - min(list(endingIndex.values()))) + " hours range")

def findBounds(df): 
    
    import numpy as np

    # returns 3 series of data

    lower_bound = df.apply(min, axis=1)
    upper_bound = df.apply(max, axis=1)
    median = df.apply(np.median, axis=1)

    return lower_bound, upper_bound, median

def section_part_of_year(dates_series, start_date, end_date):

    import pandas as pd
    # Convert the start and end dates from strings to datetime objects.
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date not in dates_series.values:
        raise NameError("check the year, date is missing")

    # Find and save the index corresponding to the start and end dates
    start_index = dates_series.loc[dates_series == start_date].index[0]
    end_index = dates_series.loc[dates_series == end_date].index[0]

    return start_index, end_index

def drawRegion(start_index, end_index, dates_series, df, color, alpha=0.2, zorder=0, fan = True):
    import matplotlib.pyplot as plt
    lower_bound, upper_bound, median = findBounds(df)

    # draw the shaded polygon around our median
    if fan: 
        plt.fill_between(dates_series[start_index:end_index + 1], lower_bound[start_index:end_index + 1], upper_bound[start_index:end_index + 1], alpha=alpha, color=color, zorder=zorder)

    # draw median line
    plt.plot(dates_series[start_index:end_index + 1], median[start_index:end_index + 1], color=color, linewidth=3)#, alpha=0.75)

import numpy as np
def fanGraph(df_allMembers, saver, start_date, end_date, onset, offset, line_count="Total Model Median", y_min=2, y_max=20, members=list(np.arange(-3.5, 4.0, 0.5)), cycle="0", leadtime="12", fan = True):  
    
    import matplotlib.dates as mdates
    import pandas as pd
    import matplotlib.pyplot as plt

    colorList = ['mediumblue', 'teal', 'cadetblue', 'cornflowerblue', 'deepskyblue', 'skyblue', 'lightsteelblue', 'gold', 'pink', 'lightsalmon', 'lightcoral', 'palevioletred', 'orangered', 'indianred', 'maroon']

    save = '_cycle_' + cycle + '_' + leadtime + 'h_' # variable used to create the name while saving files
    if fan == False: 
        save += '_spaghetti'
    elif fan == True:
        save += "_fans"

    dates_series = pd.to_datetime(df_allMembers['dateAndTime'], format="%Y-%m-%d %H:%M:%S") # seperate series just for datetime
    start_index, end_index = section_part_of_year(dates_series, start_date, end_date)
    # print(start_index,'starttjjl')
    # print(end_index, 'endslkfdjl')

    columns_to_remove = ['dateAndTime']  # List of column names to remove
    df_allMembers = df_allMembers.drop(columns=columns_to_remove)

    ## --------------------------------------------
    ## Make the graph fancy and nice looking
    axis_fontsize = 36
    plt.rcParams.update({'font.size': axis_fontsize})

    fig = plt.figure(figsize=(40,10))
    ax = fig.add_subplot(1, 1, 1)  # Adjust the parameters as needed

    plt.tight_layout()

    # plt.xticks(rotation=35)
    plt.xticks(rotation=15)
    
    plt.rcParams.update({'font.size': 28})  # Change the value as desired, to change the default font size for all text

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ##
    start_date = pd.to_datetime(start_date)
    plt.title('Measurements vs NN Predictions,  Year ' + str((start_date).year))
    plt.ylabel('Water Temperature (C)', fontsize=axis_fontsize)
    # plt.ylim(2, 20)
    plt.ylim(y_min, y_max)
    plt.margins(x=0)
    plt.xlabel('Date (MM-DD)', fontsize=axis_fontsize)

    turtleThreshold=plt.axhline(8, color='red', ls='--', linewidth=3)
    turtleThreshold.set_label('Sea Turtle Threshold')

    fisheriesThreshold=plt.axhline(4.5, color='blue', ls='--', linewidth=3)
    fisheriesThreshold.set_label('Sea Turtle Threshold')

    plt.grid()
    
    plt.plot(dates_series[start_index:end_index + 1], df_allMembers['Target'][start_index:end_index + 1], color='black', linewidth=3, ls='--', zorder= 1000, label='Target')


    ##
    # plt.savefig('fantest'+save)
    ## --------------------------------------------
    onset_start = dates_series.iloc[onset["Earliest"][0]]
    onset_end = dates_series.iloc[onset["Latest"][0]]
    offset_start = dates_series.iloc[offset["Earliest"][0]]
    offset_end = dates_series.iloc[offset["Latest"][0]]

    
    if line_count == "Total Model Median":
        lower_bound, upper_bound, median = findBounds(df_allMembers)
        
        if fan: 
            plt.fill_between(dates_series[start_index:end_index + 1], lower_bound[start_index:end_index + 1], upper_bound[start_index:end_index + 1], alpha=0.4, color='yellow', zorder=3)
        
        plt.plot(dates_series[start_index:end_index + 1], median[start_index:end_index + 1], color='#ba9c00', linewidth=3, label='Ensemble Median')# , alpha=0.75)
        

    elif line_count == "Hot/Cold/Perfect Medians":
        # split allMembers into cold, hot, and perfprog
        coldMembers = df_allMembers.loc[:, df_allMembers.columns.str.startswith('-')]             # Extract negative columns
        perfprogMembers = df_allMembers.loc[:, df_allMembers.columns.str.startswith('0.0')]       # Extract perfectProg column
        hotMembers = df_allMembers.loc[:, ~df_allMembers.columns.str.startswith(('-', '0.0'))]    # Extract positive columns

        # Plot the shaded regions
        drawRegion(start_index, end_index, dates_series, coldMembers, 'blue', fan=fan)
        drawRegion(start_index, end_index, dates_series, hotMembers, 'red', fan=fan)
        drawRegion(start_index, end_index, dates_series, perfprogMembers, 'yellow', fan=fan)


    elif line_count == "15 Member Medians":
        for i in range(len(members)):
            df = df_allMembers.loc[:, df_allMembers.columns.str.startswith(str(members[i]))]
            drawRegion(start_index, end_index, dates_series, df, colorList[i], fan=fan)


    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), fancybox=True, ncol=1)
    # plt.plot(dates_series[start_index:end_index + 1], df_allMembers['Target'][start_index:end_index + 1], color='black', linewidth=3)
    
    plt.axvline(x=onset_start, color='b', label=f'Vertical Line at x={onset_start}', linestyle='--')  
    plt.axvline(x=onset_end, color='b', label=f'Vertical Line at x={onset_end}', linestyle='--')
    plt.axvline(x=offset_start, color='b', label=f'Vertical Line at x={offset_start}', linestyle='--')
    plt.axvline(x=offset_end, color='b', label=f'Vertical Line at x={offset_end}', linestyle='--')

    if saver == True:
        plt.savefig(line_count + save, bbox_inches='tight')
    elif saver == False:
        plt.show()

    plt.close()

def onset_offset_calculations(df, threshold, date_range):
    """
    Author(s): Christian Duff


    - Finding the onset and offset of the ensemble model with respect to a given threshold
    - Also will be calculating the time interval between onset and offset
    """


    """ 
        - find the first instance each member onsets the threshold
        subtract the first member and the last member to onset the threshold

        - find the last instance each member offsets the threshold
        subtract the first member and the last member that offsets the threshold

        - subtract the very first instance (first member to onset) with the last instance (last member to offset) 
    """

    if(date_range == "dec"):
        df = df.iloc[7660:] # for december  
    
    if(date_range == "jan-march"):
        df = df.iloc[:2070] # for jan-march

    def onset_offset_blocks(df, threshold):
        '''finding the contiguous blocks of values per column that have values less than the threshold given, 
            returns list containing blocks per column'''

        # dropping the first three columns (indexer, datetime, target) as to not influence the calculations
        df = df.drop(columns=df.columns[:3], axis=0) 
        num_columns = df.shape[1]
        column_data = []

        for i in range(num_columns):
            data = df.iloc[:, [i]]
            
            # setting the values less than the threshold to true and the rest to false to be used for calculations later
            mask = (data < threshold).all(axis=1)

            # creating different groups based on blocks of contiguous true values 
            # (assigning an integer value to represent a group based on each change in contiguous blocks)
            mask_diff = mask.ne(mask.shift()).cumsum()

            # df containing only the True values based on the threshold given
            filtered_mask = mask[mask]

            # list to hold the information per cold stun event
            cs_event_info = []

            # iterating through the filtered mask and grabbing each group 
            for cs_id, group in filtered_mask.groupby(mask_diff):
                
                # skipping the groups of contiguous False blocks
                if group.empty:
                    continue  
                
                # beginning of the block
                start_index = group.index[0]

                # end of the block
                end_index = group.index[-1]
                
                if((end_index - start_index) < 48):
                    # if crossing of the threshold only lasts for a short amount of time
                    # (i.e doesn't really "count" as an event)
                    break
                else:
                    # storing the information of the block into a list to later be accessed for calculating the onset and offset
                    cs_event_info.append([start_index, end_index])

            
            # Output information about each block for each member
            #print(f"Column {i}\n")
            #for j, (start, end, num_hours) in enumerate(column_data[i], 1):

            #    print(f"Block {j}: Starts at row {start}, ends at row {end}, with {num_hours} rows.")

            column_data.append(cs_event_info)
        
        return column_data 

    # getting column data
    data = onset_offset_blocks(df, threshold)
    
    # getting maximum number of threshold crossings based on a column basis
    max_num_blocks = max(len(column) for column in data)

    # gathering information on all onsets
    onset_account = {"Earliest": [],
                    "Latest": [],
                    "Hours": []}
    
    # gathering information on all offsets
    offset_account = {"Earliest": [],
                    "Latest": [],
                    "Hours": []}
    
    # Iterate over each cold stun event position (across all Columns)
    for cs_event in range(max_num_blocks):
        #print(f"Analyzing Cold-Stun Event {cs_event + 1} across all columns:")
        
        # Collect the corresponding grandchild from each child, if it exists
        corresponding_block = [column[cs_event] if len(column) > cs_event else [None, None] for column in data]


        # Separate values by their positions (first or second)
        onsets = [b[0] for b in corresponding_block if b[0] is not None]
        offsets = [b[1] for b in corresponding_block if b[1] is not None]
        
        # Determine the smallest and largest values
        if onsets:  # If there are any valid first values

            earliest_onset = min(onsets)
            latest_onset = max(onsets)

            #print(f"The earliest onset of Cold-Stun Event {cs_event + 1} is at index {earliest_onset}.\n")
            #print(f"The latest onset of Cold-Stun Event {cs_event + 1} is at index {latest_onset}.\n")
            #print(f"Number of hours: {latest_onset - earliest_onset}\n")

            onset_account["Earliest"].append(earliest_onset)
            onset_account["Latest"].append(latest_onset)
            onset_account["Hours"].append((latest_onset - earliest_onset))

        else:
            print(f"No valid integers in the first index for Cold-Stun Event {cs_event + 1}.")

        if offsets:  # If there are any valid second values

            earliest_offset = min(offsets)
            latest_offset = max(offsets)

            #print(f"The earliest offset at Cold-Stun Event {cs_event + 1} is at index {earliest_offset}.\n")
            #print(f"The latest offset at Cold-Stun Event {cs_event + 1} is at index {latest_offset}.\n")
            #print(f"Number of hours: {latest_offset - earliest_offset}\n")

            offset_account["Earliest"].append(earliest_offset)
            offset_account["Latest"].append(latest_offset)
            offset_account["Hours"].append((latest_offset - earliest_offset))
        else:
            print(f"No valid integers in the second index for Cold-Stun Event {cs_event + 1}.")

    return onset_account, offset_account

def time_series_graphing(df, start_date, end_date, onset, offset, cycle, leadtime, line_count, fan, save):
    '''
        - df of data to be graphed
        - starting date to be graphed
        - ending date to be graphed
        - cycle of data
        - leadtime of data
        - line_count can be "15 Member Medians", "Total Model Median" , "Hot/Cold/Perfect Medians", "ALL 450"
        - fan of colored in space lines graphed

        Author(s): Christian , Hector
    '''

    import pandas as pd
    import plotly.graph_objects as go 

    colorList = ['mediumblue', 'teal', 'cadetblue', 'cornflowerblue', 'deepskyblue', 'skyblue', 'lightsteelblue', 'gold', 'pink', 'lightsalmon', 'lightcoral', 'palevioletred', 'orangered', 'indianred', 'maroon']
    members = list(np.arange(-3.5, 4.0, 0.5))

    # Convert the start and end dates from strings to datetime objects.
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    save_path = '_cycle_' + cycle + '_' + leadtime + 'h_'

    if fan == True:
        save_path += '_fans'
    if fan == False:
        save_path += '_spaghetti'

    df['dateAndTime'] = pd.to_datetime(df['dateAndTime'], format="%Y-%m-%d %H:%M:%S")

    if start_date not in df['dateAndTime'].values:
        raise NameError("check the year, date is missing")
    
    if end_date not in df['dateAndTime'].values:
        raise NameError("check the year, date is missing")

    # Find and save the index corresponding to the start and end dates
    start_index = df['dateAndTime'].loc[df['dateAndTime'] == start_date].index[0]
    end_index = df['dateAndTime'].loc[df['dateAndTime'] == end_date].index[0]

    sub_df = df.drop(df.columns[[0,1]], axis=1)

    onset_start = df['dateAndTime'].iloc[onset["Earliest"][0]]
    onset_end = df['dateAndTime'].iloc[onset["Latest"][0]]
    offset_start = df['dateAndTime'].iloc[offset["Earliest"][0]]
    offset_end = df['dateAndTime'].iloc[offset["Latest"][0]]

    fig = go.Figure()
    line_mode = "lines" # "lines+markers" "lines"
    
    if line_count == "Total Model Median":
        lower_bound = sub_df.apply(min, axis=1)
        upper_bound = sub_df.apply(max, axis=1)
        median = sub_df.apply(np.median, axis=1)

        if fan == True:
            # graphing the lower region of the total model
            fig.add_trace( go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=lower_bound[start_index:end_index+1], 
                    line=dict(color='yellow'),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=upper_bound[start_index:end_index+1], 
                    line=dict(color='yellow'), 
                    fill='tonexty',
                    mode=line_mode))
        
        # graphing the total model median (1 line: median of all 450 models)
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                   y=median[start_index:end_index+1], 
                   line=dict(color='#ba9c00'), 
                   mode=line_mode, 
                   name='Total Model Median'))
    
    if line_count == "15 Member Median":
        for i in range(len(members)):
            sub_df = sub_df.loc[:, sub_df.columns.str.startswith(str(members[i]))]
            lower_bound = sub_df.apply(min, axis=1)
            upper_bound = sub_df.apply(max, axis=1)
            median = sub_df.apply(np.median, axis=1)

            if fan == True:
                # graphing the lower region of the total model
                fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                        y=lower_bound[start_index:end_index+1], 
                        line=dict(color=colorList[i]),
                        mode=line_mode))
                
                # graphing the upper region of the total model 
                fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                        y=upper_bound[start_index:end_index+1], 
                        line=dict(color=colorList[i]), 
                        fill='tonexty',
                        mode=line_mode))
                
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                        y=median[start_index:end_index+1], 
                        line=dict(color=colorList[i]),
                        mode=line_mode))
        
    if line_count == "Hot/Cold/Perfect Medians":
        # split allMembers into cold, hot, and perfprog
        coldMembers = sub_df.loc[:, sub_df.columns.str.startswith('-')]             # Extract negative columns
        perfprogMembers = sub_df.loc[:, sub_df.columns.str.startswith('0.0')]       # Extract perfectProg column
        hotMembers = sub_df.loc[:, ~sub_df.columns.str.startswith(('-', '0.0'))]    # Extract positive columns

        cold_lower_bound = coldMembers.apply(min, axis=1)
        cold_upper_bound = coldMembers.apply(max, axis=1)
        cold_median = coldMembers.apply(np.median, axis=1)

        perfprogMembers_lower_bound = perfprogMembers.apply(min, axis=1)
        perfprogMembers_upper_bound = perfprogMembers.apply(max, axis=1)
        perfprogMembers_median = perfprogMembers.apply(np.median, axis=1)

        hotMembers_lower_bound = hotMembers.apply(min, axis=1)
        hotMembers_upper_bound = hotMembers.apply(max, axis=1)
        hotMembers_median = hotMembers.apply(np.median, axis=1)

        if fan == True:
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=cold_lower_bound[start_index:end_index+1], 
                    line=dict(color="blue"),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=cold_upper_bound[start_index:end_index+1], 
                    line=dict(color="blue"), 
                    fill='tonexty',
                    mode=line_mode))
            
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=perfprogMembers_lower_bound[start_index:end_index+1], 
                    line=dict(color="yellow"),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=perfprogMembers_upper_bound[start_index:end_index+1], 
                    line=dict(color="yellow"), 
                    fill='tonexty',
                    mode=line_mode))
            
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=hotMembers_lower_bound[start_index:end_index+1], 
                    line=dict(color="red"),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=hotMembers_upper_bound[start_index:end_index+1], 
                    line=dict(color="red"), 
                    fill='tonexty',
                    mode=line_mode))
            
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                y=cold_median[start_index:end_index+1], 
                line=dict(color="blue"),
                mode=line_mode))
        
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                y=perfprogMembers_median[start_index:end_index+1], 
                line=dict(color="#ba9c00"),
                mode=line_mode))
        
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                y=hotMembers_median[start_index:end_index+1], 
                line=dict(color="red"),
                mode=line_mode))

    fig.add_hline(y=8, line=dict(color='red'), name = 'cold-stun threshold') # cold stunning threshold

    fig.add_vline(x=onset_start, line=dict(color='blue', dash="dot")) 
    fig.add_vline(x=onset_end, line=dict(color='blue', dash="dot")) 
    fig.add_vline(x=offset_start, line=dict(color='blue', dash="dot")) 
    fig.add_vline(x=offset_end, line=dict(color='blue', dash="dot")) 

    fig.update_layout(
                title=f'{line_count}',
                xaxis_title='date_time',
                yaxis_title='water temperature (C)'
            )
    
    fig.show()

''' Ensemble time series grapher functions ends'''



''' Ensemble time series grapher functions begins'''


def calcThreshold_of_CS_event(path, datetime_split):
    '''Calculates the Threshold of the beginning and ending of the cold stunning event
        Author: Hector M. Marrero-Colominas'''

    import pandas as pd
    # load in the dataset (csv) to a dataframe
    ensemble_df = pd.read_csv(path, index_col='dateAndTime', parse_dates=True)
    if(datetime_split == "dec"):
        ensemble_df = ensemble_df.iloc[7660:] # for december  
    
    if(datetime_split == "jan-mar"):
        ensemble_df = ensemble_df.iloc[:2070] # for jan-march

    # Columns to remove
    columns_to_remove = ['Target']#, '3.5', '-3.5']  # List of column names to remove
    ensemble_df = ensemble_df.drop(columns=columns_to_remove)


    startIndex = {}
    endingIndex = {}

    # print(ensemble_df.head(15))

    # Iterate over each column and find the start values
    for column in ensemble_df.columns:
        if column == "dateAndTime": continue
        # print("columns:", column)
        col_values = ensemble_df[column].tolist()  # Convert column to a list
        # print(col_values)

        start_index = None
        for i, value in enumerate(col_values):
            if float(value) < 8:
                if start_index is None:
                    start_index = i
                    startIndex[str(column)] = i
        # start_index = ensemble_df[column].apply(lambda x: x < 8).idxmax()

        end_index = None
        for i, value in enumerate(col_values[startIndex[column]:]):
            if float(value) > 8 and float(value) < 8.5:
                if end_index is None and i > 24:
                    end_index = i
                    endingIndex[str(column)] = i + startIndex[column]
        
                


    dateAndTime = ensemble_df.index


    # complicated line to transform the dateAndTime string to something nicer to read
    startMin = pd.to_datetime(dateAndTime[min(list(startIndex.values()))]).strftime("%m-%d-%Y %H%M")
    startMax = pd.to_datetime(dateAndTime[max(list(startIndex.values()))]).strftime("%m-%d-%Y %H%M")
    endMin = pd.to_datetime(dateAndTime[min(list(endingIndex.values()))]).strftime("%m-%d-%Y %H%M")
    endMax = pd.to_datetime(dateAndTime[max(list(endingIndex.values()))]).strftime("%m-%d-%Y %H%M")
    
    start_threshold = startMin, startMax
    end_threshold = endMin, endMax
    print("start ", start_threshold, max(list(startIndex.values())) - min(list(startIndex.values())), "hours range")
    print("end ", end_threshold, max(list(endingIndex.values())) - min(list(endingIndex.values())), "hours range")

    
    #with open('ensemble_threshold', 'w') as file:
    #    file.write("start " + str(start_threshold) +' '+ str(max(list(startIndex.values())) - min(list(startIndex.values()))) + " hours range")
    #    file.write('\n')
    #    file.write("end " + str(end_threshold) +' '+ str(max(list(endingIndex.values())) - min(list(endingIndex.values()))) + " hours range")

def findBounds(df): 
    
    import numpy as np

    # returns 3 series of data

    lower_bound = df.apply(min, axis=1)
    upper_bound = df.apply(max, axis=1)
    median = df.apply(np.median, axis=1)

    return lower_bound, upper_bound, median

def section_part_of_year(dates_series, start_date, end_date):

    import pandas as pd
    # Convert the start and end dates from strings to datetime objects.
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date not in dates_series.values:
        raise NameError("check the year, date is missing")

    # Find and save the index corresponding to the start and end dates
    start_index = dates_series.loc[dates_series == start_date].index[0]
    end_index = dates_series.loc[dates_series == end_date].index[0]

    return start_index, end_index

def drawRegion(start_index, end_index, dates_series, df, color, alpha=0.2, zorder=0, fan = True):
    import matplotlib.pyplot as plt
    lower_bound, upper_bound, median = findBounds(df)

    # draw the shaded polygon around our median
    if fan: 
        plt.fill_between(dates_series[start_index:end_index + 1], lower_bound[start_index:end_index + 1], upper_bound[start_index:end_index + 1], alpha=alpha, color=color, zorder=zorder)

    # draw median line
    plt.plot(dates_series[start_index:end_index + 1], median[start_index:end_index + 1], color=color, linewidth=3)#, alpha=0.75)

import numpy as np
def fanGraph(df_allMembers, saver, start_date, end_date, onset, offset, line_count="Total Model Median", y_min=2, y_max=20, members=list(np.arange(-3.5, 4.0, 0.5)), cycle="0", leadtime="12", fan = True):  
    
    import matplotlib.dates as mdates
    import pandas as pd
    import matplotlib.pyplot as plt

    colorList = ['mediumblue', 'teal', 'cadetblue', 'cornflowerblue', 'deepskyblue', 'skyblue', 'lightsteelblue', 'gold', 'pink', 'lightsalmon', 'lightcoral', 'palevioletred', 'orangered', 'indianred', 'maroon']

    save = '_cycle_' + cycle + '_' + leadtime + 'h_' # variable used to create the name while saving files
    if fan == False: 
        save += '_spaghetti'
    elif fan == True:
        save += "_fans"

    dates_series = pd.to_datetime(df_allMembers['dateAndTime'], format="%Y-%m-%d %H:%M:%S") # seperate series just for datetime
    start_index, end_index = section_part_of_year(dates_series, start_date, end_date)
    # print(start_index,'starttjjl')
    # print(end_index, 'endslkfdjl')

    columns_to_remove = ['dateAndTime']  # List of column names to remove
    df_allMembers = df_allMembers.drop(columns=columns_to_remove)

    ## --------------------------------------------
    ## Make the graph fancy and nice looking
    axis_fontsize = 36
    plt.rcParams.update({'font.size': axis_fontsize})

    fig = plt.figure(figsize=(40,10))
    ax = fig.add_subplot(1, 1, 1)  # Adjust the parameters as needed

    plt.tight_layout()

    # plt.xticks(rotation=35)
    plt.xticks(rotation=15)
    
    plt.rcParams.update({'font.size': 28})  # Change the value as desired, to change the default font size for all text

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ##
    start_date = pd.to_datetime(start_date)
    plt.title('Measurements vs NN Predictions,  Year ' + str((start_date).year))
    plt.ylabel('Water Temperature (C)', fontsize=axis_fontsize)
    # plt.ylim(2, 20)
    plt.ylim(y_min, y_max)
    plt.margins(x=0)
    plt.xlabel('Date (MM-DD)', fontsize=axis_fontsize)

    turtleThreshold=plt.axhline(8, color='red', ls='--', linewidth=3)
    turtleThreshold.set_label('Sea Turtle Threshold')

    fisheriesThreshold=plt.axhline(4.5, color='blue', ls='--', linewidth=3)
    fisheriesThreshold.set_label('Sea Turtle Threshold')

    plt.grid()
    
    plt.plot(dates_series[start_index:end_index + 1], df_allMembers['Target'][start_index:end_index + 1], color='black', linewidth=3, ls='--', zorder= 1000, label='Target')


    ##
    # plt.savefig('fantest'+save)
    ## --------------------------------------------
    onset_start = dates_series.iloc[onset["Earliest"][0]]
    onset_end = dates_series.iloc[onset["Latest"][0]]
    offset_start = dates_series.iloc[offset["Earliest"][0]]
    offset_end = dates_series.iloc[offset["Latest"][0]]

    
    if line_count == "Total Model Median":
        lower_bound, upper_bound, median = findBounds(df_allMembers)
        
        if fan: 
            plt.fill_between(dates_series[start_index:end_index + 1], lower_bound[start_index:end_index + 1], upper_bound[start_index:end_index + 1], alpha=0.4, color='yellow', zorder=3)
        
        plt.plot(dates_series[start_index:end_index + 1], median[start_index:end_index + 1], color='#ba9c00', linewidth=3, label='Ensemble Median')# , alpha=0.75)
        

    elif line_count == "Hot/Cold/Perfect Medians":
        # split allMembers into cold, hot, and perfprog
        coldMembers = df_allMembers.loc[:, df_allMembers.columns.str.startswith('-')]             # Extract negative columns
        perfprogMembers = df_allMembers.loc[:, df_allMembers.columns.str.startswith('0.0')]       # Extract perfectProg column
        hotMembers = df_allMembers.loc[:, ~df_allMembers.columns.str.startswith(('-', '0.0'))]    # Extract positive columns

        # Plot the shaded regions
        drawRegion(start_index, end_index, dates_series, coldMembers, 'blue', fan=fan)
        drawRegion(start_index, end_index, dates_series, hotMembers, 'red', fan=fan)
        drawRegion(start_index, end_index, dates_series, perfprogMembers, 'yellow', fan=fan)


    elif line_count == "15 Member Medians":
        for i in range(len(members)):
            df = df_allMembers.loc[:, df_allMembers.columns.str.startswith(str(members[i]))]
            drawRegion(start_index, end_index, dates_series, df, colorList[i], fan=fan)


    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), fancybox=True, ncol=1)
    # plt.plot(dates_series[start_index:end_index + 1], df_allMembers['Target'][start_index:end_index + 1], color='black', linewidth=3)
    
    plt.axvline(x=onset_start, color='b', label=f'Vertical Line at x={onset_start}', linestyle='--')  
    plt.axvline(x=onset_end, color='b', label=f'Vertical Line at x={onset_end}', linestyle='--')
    plt.axvline(x=offset_start, color='b', label=f'Vertical Line at x={offset_start}', linestyle='--')
    plt.axvline(x=offset_end, color='b', label=f'Vertical Line at x={offset_end}', linestyle='--')

    if saver == True:
        plt.savefig(line_count + save, bbox_inches='tight')
    elif saver == False:
        plt.show()

    plt.close()

def onset_offset_calculations(df, threshold, date_range):
    """
    Author(s): Christian Duff


    - Finding the onset and offset of the ensemble model with respect to a given threshold
    - Also will be calculating the time interval between onset and offset
    """


    """ 
        - find the first instance each member onsets the threshold
        subtract the first member and the last member to onset the threshold

        - find the last instance each member offsets the threshold
        subtract the first member and the last member that offsets the threshold

        - subtract the very first instance (first member to onset) with the last instance (last member to offset) 
    """

    if(date_range == "dec"):
        df = df.iloc[7660:] # for december  
    
    if(date_range == "jan-march"):
        df = df.iloc[:2070] # for jan-march

    def onset_offset_blocks(df, threshold):
        '''finding the contiguous blocks of values per column that have values less than the threshold given, 
            returns list containing blocks per column'''

        # dropping the first three columns (indexer, datetime, target) as to not influence the calculations
        df = df.drop(columns=df.columns[:3], axis=0) 
        num_columns = df.shape[1]
        column_data = []

        for i in range(num_columns):
            data = df.iloc[:, [i]]
            
            # setting the values less than the threshold to true and the rest to false to be used for calculations later
            mask = (data < threshold).all(axis=1)

            # creating different groups based on blocks of contiguous true values 
            # (assigning an integer value to represent a group based on each change in contiguous blocks)
            mask_diff = mask.ne(mask.shift()).cumsum()

            # df containing only the True values based on the threshold given
            filtered_mask = mask[mask]

            # list to hold the information per cold stun event
            cs_event_info = []

            # iterating through the filtered mask and grabbing each group 
            for cs_id, group in filtered_mask.groupby(mask_diff):
                
                # skipping the groups of contiguous False blocks
                if group.empty:
                    continue  
                
                # beginning of the block
                start_index = group.index[0]

                # end of the block
                end_index = group.index[-1]
                
                if((end_index - start_index) < 48):
                    # if crossing of the threshold only lasts for a short amount of time
                    # (i.e doesn't really "count" as an event)
                    break
                else:
                    # storing the information of the block into a list to later be accessed for calculating the onset and offset
                    cs_event_info.append([start_index, end_index])

            
            # Output information about each block for each member
            #print(f"Column {i}\n")
            #for j, (start, end, num_hours) in enumerate(column_data[i], 1):

            #    print(f"Block {j}: Starts at row {start}, ends at row {end}, with {num_hours} rows.")

            column_data.append(cs_event_info)
        
        return column_data 

    # getting column data
    data = onset_offset_blocks(df, threshold)
    
    # getting maximum number of threshold crossings based on a column basis
    max_num_blocks = max(len(column) for column in data)

    # gathering information on all onsets
    onset_account = {"Earliest": [],
                    "Latest": [],
                    "Hours": []}
    
    # gathering information on all offsets
    offset_account = {"Earliest": [],
                    "Latest": [],
                    "Hours": []}
    
    # Iterate over each cold stun event position (across all Columns)
    for cs_event in range(max_num_blocks):
        #print(f"Analyzing Cold-Stun Event {cs_event + 1} across all columns:")
        
        # Collect the corresponding grandchild from each child, if it exists
        corresponding_block = [column[cs_event] if len(column) > cs_event else [None, None] for column in data]


        # Separate values by their positions (first or second)
        onsets = [b[0] for b in corresponding_block if b[0] is not None]
        offsets = [b[1] for b in corresponding_block if b[1] is not None]
        
        # Determine the smallest and largest values
        if onsets:  # If there are any valid first values

            earliest_onset = min(onsets)
            latest_onset = max(onsets)

            #print(f"The earliest onset of Cold-Stun Event {cs_event + 1} is at index {earliest_onset}.\n")
            #print(f"The latest onset of Cold-Stun Event {cs_event + 1} is at index {latest_onset}.\n")
            #print(f"Number of hours: {latest_onset - earliest_onset}\n")

            onset_account["Earliest"].append(earliest_onset)
            onset_account["Latest"].append(latest_onset)
            onset_account["Hours"].append((latest_onset - earliest_onset))

        else:
            print(f"No valid integers in the first index for Cold-Stun Event {cs_event + 1}.")

        if offsets:  # If there are any valid second values

            earliest_offset = min(offsets)
            latest_offset = max(offsets)

            #print(f"The earliest offset at Cold-Stun Event {cs_event + 1} is at index {earliest_offset}.\n")
            #print(f"The latest offset at Cold-Stun Event {cs_event + 1} is at index {latest_offset}.\n")
            #print(f"Number of hours: {latest_offset - earliest_offset}\n")

            offset_account["Earliest"].append(earliest_offset)
            offset_account["Latest"].append(latest_offset)
            offset_account["Hours"].append((latest_offset - earliest_offset))
        else:
            print(f"No valid integers in the second index for Cold-Stun Event {cs_event + 1}.")

    return onset_account, offset_account

def time_series_graphing(df, start_date, end_date, onset, offset, cycle, leadtime, line_count, fan, save):
    '''
        - df of data to be graphed
        - starting date to be graphed
        - ending date to be graphed
        - cycle of data
        - leadtime of data
        - line_count can be "15 Member Medians", "Total Model Median" , "Hot/Cold/Perfect Medians", "ALL 450"
        - fan of colored in space lines graphed

        Author(s): Christian , Hector
    '''

    import pandas as pd
    import plotly.graph_objects as go 

    colorList = ['mediumblue', 'teal', 'cadetblue', 'cornflowerblue', 'deepskyblue', 'skyblue', 'lightsteelblue', 'gold', 'pink', 'lightsalmon', 'lightcoral', 'palevioletred', 'orangered', 'indianred', 'maroon']
    members = list(np.arange(-3.5, 4.0, 0.5))

    # Convert the start and end dates from strings to datetime objects.
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    save_path = '_cycle_' + cycle + '_' + leadtime + 'h_'

    if fan == True:
        save_path += '_fans'
    if fan == False:
        save_path += '_spaghetti'

    df['dateAndTime'] = pd.to_datetime(df['dateAndTime'], format="%Y-%m-%d %H:%M:%S")

    if start_date not in df['dateAndTime'].values:
        raise NameError("check the year, date is missing")
    
    if end_date not in df['dateAndTime'].values:
        raise NameError("check the year, date is missing")

    # Find and save the index corresponding to the start and end dates
    start_index = df['dateAndTime'].loc[df['dateAndTime'] == start_date].index[0]
    end_index = df['dateAndTime'].loc[df['dateAndTime'] == end_date].index[0]

    sub_df = df.drop(df.columns[[0,1]], axis=1)

    onset_start = df['dateAndTime'].iloc[onset["Earliest"][0]]
    onset_end = df['dateAndTime'].iloc[onset["Latest"][0]]
    offset_start = df['dateAndTime'].iloc[offset["Earliest"][0]]
    offset_end = df['dateAndTime'].iloc[offset["Latest"][0]]

    fig = go.Figure()
    line_mode = "lines" # "lines+markers" "lines"
    
    if line_count == "Total Model Median":
        lower_bound = sub_df.apply(min, axis=1)
        upper_bound = sub_df.apply(max, axis=1)
        median = sub_df.apply(np.median, axis=1)

        if fan == True:
            # graphing the lower region of the total model
            fig.add_trace( go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=lower_bound[start_index:end_index+1], 
                    line=dict(color='yellow'),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=upper_bound[start_index:end_index+1], 
                    line=dict(color='yellow'), 
                    fill='tonexty',
                    mode=line_mode))
        
        # graphing the total model median (1 line: median of all 450 models)
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                   y=median[start_index:end_index+1], 
                   line=dict(color='#ba9c00'), 
                   mode=line_mode, 
                   name='Total Model Median'))
    
    if line_count == "15 Member Median":
        for i in range(len(members)):
            sub_df = sub_df.loc[:, sub_df.columns.str.startswith(str(members[i]))]
            lower_bound = sub_df.apply(min, axis=1)
            upper_bound = sub_df.apply(max, axis=1)
            median = sub_df.apply(np.median, axis=1)

            if fan == True:
                # graphing the lower region of the total model
                fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                        y=lower_bound[start_index:end_index+1], 
                        line=dict(color=colorList[i]),
                        mode=line_mode))
                
                # graphing the upper region of the total model 
                fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                        y=upper_bound[start_index:end_index+1], 
                        line=dict(color=colorList[i]), 
                        fill='tonexty',
                        mode=line_mode))
                
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                        y=median[start_index:end_index+1], 
                        line=dict(color=colorList[i]),
                        mode=line_mode))
        
    if line_count == "Hot/Cold/Perfect Medians":
        # split allMembers into cold, hot, and perfprog
        coldMembers = sub_df.loc[:, sub_df.columns.str.startswith('-')]             # Extract negative columns
        perfprogMembers = sub_df.loc[:, sub_df.columns.str.startswith('0.0')]       # Extract perfectProg column
        hotMembers = sub_df.loc[:, ~sub_df.columns.str.startswith(('-', '0.0'))]    # Extract positive columns

        cold_lower_bound = coldMembers.apply(min, axis=1)
        cold_upper_bound = coldMembers.apply(max, axis=1)
        cold_median = coldMembers.apply(np.median, axis=1)

        perfprogMembers_lower_bound = perfprogMembers.apply(min, axis=1)
        perfprogMembers_upper_bound = perfprogMembers.apply(max, axis=1)
        perfprogMembers_median = perfprogMembers.apply(np.median, axis=1)

        hotMembers_lower_bound = hotMembers.apply(min, axis=1)
        hotMembers_upper_bound = hotMembers.apply(max, axis=1)
        hotMembers_median = hotMembers.apply(np.median, axis=1)

        if fan == True:
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=cold_lower_bound[start_index:end_index+1], 
                    line=dict(color="blue"),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=cold_upper_bound[start_index:end_index+1], 
                    line=dict(color="blue"), 
                    fill='tonexty',
                    mode=line_mode))
            
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=perfprogMembers_lower_bound[start_index:end_index+1], 
                    line=dict(color="yellow"),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=perfprogMembers_upper_bound[start_index:end_index+1], 
                    line=dict(color="yellow"), 
                    fill='tonexty',
                    mode=line_mode))
            
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=hotMembers_lower_bound[start_index:end_index+1], 
                    line=dict(color="red"),
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                    y=hotMembers_upper_bound[start_index:end_index+1], 
                    line=dict(color="red"), 
                    fill='tonexty',
                    mode=line_mode))
            
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                y=cold_median[start_index:end_index+1], 
                line=dict(color="blue"),
                mode=line_mode))
        
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                y=perfprogMembers_median[start_index:end_index+1], 
                line=dict(color="#ba9c00"),
                mode=line_mode))
        
        fig.add_trace(go.Scatter(x=df['dateAndTime'][start_index:end_index+1].astype(dtype=str), 
                y=hotMembers_median[start_index:end_index+1], 
                line=dict(color="red"),
                mode=line_mode))

    fig.add_hline(y=8, line=dict(color='red'), name = 'cold-stun threshold') # cold stunning threshold

    fig.add_vline(x=onset_start, line=dict(color='blue', dash="dot")) 
    fig.add_vline(x=onset_end, line=dict(color='blue', dash="dot")) 
    fig.add_vline(x=offset_start, line=dict(color='blue', dash="dot")) 
    fig.add_vline(x=offset_end, line=dict(color='blue', dash="dot")) 

    fig.update_layout(
                title=f'{line_count}',
                xaxis_title='date_time',
                yaxis_title='water temperature (C)'
            )
    
    fig.show()

''' Ensemble time series grapher functions ends'''



''' Ensemble time series grapher functions begins'''


def calcThreshold_of_CS_event(path, datetime_split):
    '''Calculates the Threshold of the beginning and ending of the cold stunning event
        Author: Hector M. Marrero-Colominas'''

    import pandas as pd
    # load in the dataset (csv) to a dataframe
    ensemble_df = pd.read_csv(path, index_col='dateAndTime', parse_dates=True)
    if(datetime_split == "dec"):
        ensemble_df = ensemble_df.iloc[7660:] # for december  
    
    if(datetime_split == "jan-mar"):
        ensemble_df = ensemble_df.iloc[:2070] # for jan-march

    # Columns to remove
    columns_to_remove = ['Target']#, '3.5', '-3.5']  # List of column names to remove
    ensemble_df = ensemble_df.drop(columns=columns_to_remove)


    startIndex = {}
    endingIndex = {}

    # print(ensemble_df.head(15))

    # Iterate over each column and find the start values
    for column in ensemble_df.columns:
        if column == "dateAndTime": continue
        # print("columns:", column)
        col_values = ensemble_df[column].tolist()  # Convert column to a list
        # print(col_values)

        start_index = None
        for i, value in enumerate(col_values):
            if float(value) < 8:
                if start_index is None:
                    start_index = i
                    startIndex[str(column)] = i
        # start_index = ensemble_df[column].apply(lambda x: x < 8).idxmax()

        end_index = None
        for i, value in enumerate(col_values[startIndex[column]:]):
            if float(value) > 8 and float(value) < 8.5:
                if end_index is None and i > 24:
                    end_index = i
                    endingIndex[str(column)] = i + startIndex[column]
        
                


    dateAndTime = ensemble_df.index


    # complicated line to transform the dateAndTime string to something nicer to read
    startMin = pd.to_datetime(dateAndTime[min(list(startIndex.values()))]).strftime("%m-%d-%Y %H%M")
    startMax = pd.to_datetime(dateAndTime[max(list(startIndex.values()))]).strftime("%m-%d-%Y %H%M")
    endMin = pd.to_datetime(dateAndTime[min(list(endingIndex.values()))]).strftime("%m-%d-%Y %H%M")
    endMax = pd.to_datetime(dateAndTime[max(list(endingIndex.values()))]).strftime("%m-%d-%Y %H%M")
    
    start_threshold = startMin, startMax
    end_threshold = endMin, endMax
    print("start ", start_threshold, max(list(startIndex.values())) - min(list(startIndex.values())), "hours range")
    print("end ", end_threshold, max(list(endingIndex.values())) - min(list(endingIndex.values())), "hours range")

    
    #with open('ensemble_threshold', 'w') as file:
    #    file.write("start " + str(start_threshold) +' '+ str(max(list(startIndex.values())) - min(list(startIndex.values()))) + " hours range")
    #    file.write('\n')
    #    file.write("end " + str(end_threshold) +' '+ str(max(list(endingIndex.values())) - min(list(endingIndex.values()))) + " hours range")

def findBounds(df): 
    
    import numpy as np

    # returns 3 series of data

    lower_bound = df.apply(min, axis=1)
    upper_bound = df.apply(max, axis=1)
    median = df.apply(np.median, axis=1)

    return lower_bound, upper_bound, median

def section_part_of_year(dates_series, start_date, end_date):

    import pandas as pd
    # Convert the start and end dates from strings to datetime objects.
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date not in dates_series.values:
        raise NameError("check the year, date is missing")

    # Find and save the index corresponding to the start and end dates
    start_index = dates_series.loc[dates_series == start_date].index[0]
    end_index = dates_series.loc[dates_series == end_date].index[0]

    return start_index, end_index

def drawRegion(start_index, end_index, dates_series, df, color, alpha=0.2, zorder=0, fan = True):
    import matplotlib.pyplot as plt
    lower_bound, upper_bound, median = findBounds(df)

    # draw the shaded polygon around our median
    if fan: 
        plt.fill_between(dates_series[start_index:end_index + 1], lower_bound[start_index:end_index + 1], upper_bound[start_index:end_index + 1], alpha=alpha, color=color, zorder=zorder)

    # draw median line
    plt.plot(dates_series[start_index:end_index + 1], median[start_index:end_index + 1], color=color, linewidth=3)#, alpha=0.75)

import numpy as np
def fanGraph(df_allMembers, saver, start_date, end_date, onset, offset, line_count="Total Model Median", y_min=2, y_max=20, members=list(np.arange(-3.5, 4.0, 0.5)), cycle="0", leadtime="12", fan = True):  
    
    import matplotlib.dates as mdates
    import pandas as pd
    import matplotlib.pyplot as plt

    colorList = ['mediumblue', 'teal', 'cadetblue', 'cornflowerblue', 'deepskyblue', 'skyblue', 'lightsteelblue', 'gold', 'pink', 'lightsalmon', 'lightcoral', 'palevioletred', 'orangered', 'indianred', 'maroon']

    save = '_cycle_' + cycle + '_' + leadtime + 'h_' # variable used to create the name while saving files
    if fan == False: 
        save += '_spaghetti'
    elif fan == True:
        save += "_fans"

    dates_series = pd.to_datetime(df_allMembers['dateAndTime'], format="%Y-%m-%d %H:%M:%S") # seperate series just for datetime
    start_index, end_index = section_part_of_year(dates_series, start_date, end_date)
    # print(start_index,'starttjjl')
    # print(end_index, 'endslkfdjl')

    columns_to_remove = ['dateAndTime']  # List of column names to remove
    df_allMembers = df_allMembers.drop(columns=columns_to_remove)

    ## --------------------------------------------
    ## Make the graph fancy and nice looking
    axis_fontsize = 36
    plt.rcParams.update({'font.size': axis_fontsize})

    fig = plt.figure(figsize=(40,10))
    ax = fig.add_subplot(1, 1, 1)  # Adjust the parameters as needed

    plt.tight_layout()

    # plt.xticks(rotation=35)
    plt.xticks(rotation=15)
    
    plt.rcParams.update({'font.size': 28})  # Change the value as desired, to change the default font size for all text

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ##
    start_date = pd.to_datetime(start_date)
    plt.title('Measurements vs NN Predictions,  Year ' + str((start_date).year))
    plt.ylabel('Water Temperature (C)', fontsize=axis_fontsize)
    # plt.ylim(2, 20)
    plt.ylim(y_min, y_max)
    plt.margins(x=0)
    plt.xlabel('Date (MM-DD)', fontsize=axis_fontsize)

    turtleThreshold=plt.axhline(8, color='red', ls='--', linewidth=3)
    turtleThreshold.set_label('Sea Turtle Threshold')

    fisheriesThreshold=plt.axhline(4.5, color='blue', ls='--', linewidth=3)
    fisheriesThreshold.set_label('Sea Turtle Threshold')

    plt.grid()
    
    plt.plot(dates_series[start_index:end_index + 1], df_allMembers['Target'][start_index:end_index + 1], color='black', linewidth=3, ls='--', zorder= 1000, label='Target')


    ##
    # plt.savefig('fantest'+save)
    ## --------------------------------------------
    onset_start = dates_series.iloc[onset["Earliest"][0]]
    onset_end = dates_series.iloc[onset["Latest"][0]]
    offset_start = dates_series.iloc[offset["Earliest"][0]]
    offset_end = dates_series.iloc[offset["Latest"][0]]

    
    if line_count == "Total Model Median":
        lower_bound, upper_bound, median = findBounds(df_allMembers)
        
        if fan: 
            plt.fill_between(dates_series[start_index:end_index + 1], lower_bound[start_index:end_index + 1], upper_bound[start_index:end_index + 1], alpha=0.4, color='yellow', zorder=3)
        
        plt.plot(dates_series[start_index:end_index + 1], median[start_index:end_index + 1], color='#ba9c00', linewidth=3, label='Ensemble Median')# , alpha=0.75)
        

    elif line_count == "Hot/Cold/Perfect Medians":
        # split allMembers into cold, hot, and perfprog
        coldMembers = df_allMembers.loc[:, df_allMembers.columns.str.startswith('-')]             # Extract negative columns
        perfprogMembers = df_allMembers.loc[:, df_allMembers.columns.str.startswith('0.0')]       # Extract perfectProg column
        hotMembers = df_allMembers.loc[:, ~df_allMembers.columns.str.startswith(('-', '0.0'))]    # Extract positive columns

        # Plot the shaded regions
        drawRegion(start_index, end_index, dates_series, coldMembers, 'blue', fan=fan)
        drawRegion(start_index, end_index, dates_series, hotMembers, 'red', fan=fan)
        drawRegion(start_index, end_index, dates_series, perfprogMembers, 'yellow', fan=fan)


    elif line_count == "15 Member Medians":
        for i in range(len(members)):
            df = df_allMembers.loc[:, df_allMembers.columns.str.startswith(str(members[i]))]
            drawRegion(start_index, end_index, dates_series, df, colorList[i], fan=fan)


    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2), fancybox=True, ncol=1)
    # plt.plot(dates_series[start_index:end_index + 1], df_allMembers['Target'][start_index:end_index + 1], color='black', linewidth=3)
    
    plt.axvline(x=onset_start, color='b', label=f'Vertical Line at x={onset_start}', linestyle='--')  
    plt.axvline(x=onset_end, color='b', label=f'Vertical Line at x={onset_end}', linestyle='--')
    plt.axvline(x=offset_start, color='b', label=f'Vertical Line at x={offset_start}', linestyle='--')
    plt.axvline(x=offset_end, color='b', label=f'Vertical Line at x={offset_end}', linestyle='--')

    if saver == True:
        plt.savefig(line_count + save, bbox_inches='tight')
    elif saver == False:
        plt.show()

    plt.close()

def onset_offset_calculations(df, threshold, on_off_percentile, date_range):
    """
    Author(s): Christian Duff


    - Finding the onset and offset of the ensemble model with respect to a given threshold
    - Also will be calculating the time interval between onset and offset
    """


    """ 
        - find the first instance each member onsets the threshold
        subtract the first member and the last member to onset the threshold

        - find the last instance each member offsets the threshold
        subtract the first member and the last member that offsets the threshold

        - subtract the very first instance (first member to onset) with the last instance (last member to offset) 
    """

    if(date_range == "dec"):
        df = df.iloc[7660:] # for december  
    
    if(date_range == "jan-march"):
        df = df.iloc[:2070] # for jan-march
    
    def onset_offset_blocks(df, threshold, percentile):
        '''finding the contiguous blocks of values per column that have values less than the threshold given, 
            returns list containing blocks per column'''

        # dropping the first three columns (indexer, datetime, target) as to not influence the calculations
        df = df.drop(columns=df.columns[:3], axis=0) 
        num_columns = df.shape[1]
        column_data = []

        if percentile != 100:
            perc_remove = ((100 - percentile) / 2) / 100
            columns_to_remove = int(num_columns * perc_remove)
            df = df.iloc[:, columns_to_remove:-columns_to_remove]
            num_columns = df.shape[1]


        for i in range(num_columns):
            data = df.iloc[:, [i]]
            
            # setting the values less than the threshold to true and the rest to false to be used for calculations later
            mask = (data < threshold).all(axis=1)

            # creating different groups based on blocks of contiguous true values 
            # (assigning an integer value to represent a group based on each change in contiguous blocks)
            mask_diff = mask.ne(mask.shift()).cumsum()

            # df containing only the True values based on the threshold given
            filtered_mask = mask[mask]

            # list to hold the information per cold stun event
            cs_event_info = []

            # iterating through the filtered mask and grabbing each group 
            for cs_id, group in filtered_mask.groupby(mask_diff):
                
                # skipping the groups of contiguous False blocks
                if group.empty:
                    continue  
                
                # beginning of the block
                start_index = group.index[0]

                # end of the block
                end_index = group.index[-1]
                
                if((end_index - start_index) < 48):
                    # if crossing of the threshold only lasts for a short amount of time
                    # (i.e doesn't really "count" as an event)
                    break
                else:
                    # storing the information of the block into a list to later be accessed for calculating the onset and offset
                    cs_event_info.append([start_index, end_index])

            
            # Output information about each block for each member
            #print(f"Column {i}\n")
            #for j, (start, end, num_hours) in enumerate(column_data[i], 1):

            #    print(f"Block {j}: Starts at row {start}, ends at row {end}, with {num_hours} rows.")

            column_data.append(cs_event_info)
        
        return column_data 

    percentile = int(on_off_percentile)
    # getting column data
    data = onset_offset_blocks(df, threshold, percentile)
    
    # getting maximum number of threshold crossings based on a column basis
    max_num_blocks = max(len(column) for column in data)

    # gathering information on all onsets
    onset_account = {"Earliest": [],
                    "Latest": [],
                    "Hours": []}
    
    # gathering information on all offsets
    offset_account = {"Earliest": [],
                    "Latest": [],
                    "Hours": []}
    
    # Iterate over each cold stun event position (across all Columns)
    for cs_event in range(max_num_blocks):
        #print(f"Analyzing Cold-Stun Event {cs_event + 1} across all columns:")
        
        # Collect the corresponding grandchild from each child, if it exists
        corresponding_block = [column[cs_event] if len(column) > cs_event else [None, None] for column in data]


        # Separate values by their positions (first or second)
        onsets = [b[0] for b in corresponding_block if b[0] is not None]
        offsets = [b[1] for b in corresponding_block if b[1] is not None]
        
        # Determine the smallest and largest values
        if onsets:  # If there are any valid first values

            earliest_onset = min(onsets) - 1
            latest_onset = max(onsets) - 1


            #print(f"The earliest onset of Cold-Stun Event {cs_event + 1} is at index {earliest_onset}.\n")
            #print(f"The latest onset of Cold-Stun Event {cs_event + 1} is at index {latest_onset}.\n")
            #print(f"Number of hours: {latest_onset - earliest_onset}\n")

            onset_account["Earliest"].append(earliest_onset)
            onset_account["Latest"].append(latest_onset)
            onset_account["Hours"].append((latest_onset - earliest_onset))

        else:
            print(f"No valid integers in the first index for Cold-Stun Event {cs_event + 1}.")

        if offsets:  # If there are any valid second values

            earliest_offset = min(offsets) 
            latest_offset = max(offsets) 

            #print(f"The earliest offset at Cold-Stun Event {cs_event + 1} is at index {earliest_offset}.\n")
            #print(f"The latest offset at Cold-Stun Event {cs_event + 1} is at index {latest_offset}.\n")
            #print(f"Number of hours: {latest_offset - earliest_offset}\n")

            offset_account["Earliest"].append(earliest_offset)
            offset_account["Latest"].append(latest_offset)
            offset_account["Hours"].append((latest_offset - earliest_offset))
        else:
            print(f"No valid integers in the second index for Cold-Stun Event {cs_event + 1}.")

    return onset_account, offset_account

def time_series_grapher(df, start_date, end_date, onset, offset, cycle, leadtime, line_count, fan, fan_percentile, line_mode, save):
    '''
        - df of data to be graphed
        - starting date to be graphed
        - ending date to be graphed
        - cycle of data
        - leadtime of data
        - line_count can be "15 Member Medians", "Total Model Median" , "Hot/Cold/Perfect Medians", "ALL 450"
        - fan of colored in space lines graphed

        Author(s): Christian , Hector
    '''

    import pandas as pd
    import plotly.graph_objects as go 

    colorList = ['mediumblue', 'teal', 'cadetblue', 'cornflowerblue', 'deepskyblue', 'skyblue', 'lightsteelblue', 'gold', 'pink', 'lightsalmon', 'lightcoral', 'palevioletred', 'orangered', 'indianred', 'maroon']
    members = list(np.arange(-3.5, 4.0, 0.5))


    # Convert the start and end dates from strings to datetime objects.
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    save_path = '_cycle_' + cycle + '_' + leadtime + 'h'

    if fan == True:
        save_path += '_fans'
    if fan == False:
        save_path += '_spaghetti'


    df['dateAndTime'] = pd.to_datetime(df['dateAndTime'], format="%Y-%m-%d %H:%M:%S")
    date_time = df['dateAndTime']

    if start_date not in date_time.values:
        raise NameError("check the year, date is missing")
    
    if end_date not in date_time.values:
        raise NameError("check the year, date is missing")

    # Find and save the index corresponding to the start and end dates
    start_index = date_time.loc[date_time == start_date].index[0]
    end_index = date_time.loc[date_time == end_date].index[0]

    sub_df = df.drop(df.columns[[0,1]], axis=1)
    df = df.drop(df.columns[[0,1]], axis=1)

    # dropping the first three columns (indexer, datetime, target) as to not influence the calculations
    num_columns = df.shape[1]

    if fan_percentile != 100:
        target = sub_df.iloc[:, 0]
        sub_df = sub_df.drop(sub_df.columns[[0]], axis=1)
        perc_remove = ((100 - int(fan_percentile)) / 2) / 100
        columns_to_remove = int(num_columns * perc_remove)
        sub_df = sub_df.iloc[:, columns_to_remove:-columns_to_remove]
        sub_df = pd.concat([target, sub_df], axis=1)

    if bool(onset["Earliest"]) == True:
        onset_start = date_time.iloc[onset["Earliest"][0]]
        onset_end = date_time.iloc[onset["Latest"][0]]
        offset_start = date_time.iloc[offset["Earliest"][0]]
        offset_end = date_time.iloc[offset["Latest"][0]]

    fig = go.Figure()
    
    if line_count == "Total_Model_Median":
        lower_bound = sub_df.apply(min, axis=1)
        upper_bound = sub_df.apply(max, axis=1)
        median = sub_df.apply(np.median, axis=1)

        if fan == True:
            # graphing the lower region of the total model
            fig.add_trace( go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=lower_bound[start_index:end_index+1], 
                    line=dict(color='yellow'),
                    name='Lower Bound',
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=upper_bound[start_index:end_index+1], 
                    line=dict(color='yellow'), 
                    fill='tonexty',
                    name='Upper Bound',
                    mode=line_mode))
        
        # graphing the total model median (1 line: median of all 450 models)
        fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                   y=median[start_index:end_index+1], 
                   line=dict(color='#ba9c00'), 
                   mode=line_mode, 
                   name='Total Model Median'))
    
    if line_count == "15_Member_Medians":

        for i in range(len(members)):
            sub_df = df.loc[:, df.columns.str.startswith(str(members[i]))]

            if fan_percentile != 100:
                target = sub_df.iloc[:, 0]
                sub_df = sub_df.drop(sub_df.columns[[0]], axis=1)
                perc_remove = ((100 - int(fan_percentile)) / 2) / 100
                columns_to_remove = int(num_columns * perc_remove)
                sub_df = sub_df.iloc[:, columns_to_remove:-columns_to_remove]
                sub_df = pd.concat([target, sub_df], axis=1)

            lower_bound = sub_df.apply(min, axis=1)
            upper_bound = sub_df.apply(max, axis=1)
            median = sub_df.apply(np.median, axis=1)

            if fan == True:
                # graphing the lower region of the total model
                fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                        y=lower_bound[start_index:end_index+1], 
                        line=dict(color=colorList[i]),
                        name=f'Member {members[i]} Lower Bound',
                        mode=line_mode))
                
                # graphing the upper region of the total model 
                fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                        y=upper_bound[start_index:end_index+1], 
                        line=dict(color=colorList[i]), 
                        fill='tonexty',
                        name=f'Member {members[i]} Upper Bound',
                        mode=line_mode))
                
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                        y=median[start_index:end_index+1], 
                        line=dict(color=colorList[i]),
                        name=f'Member {members[i]} Median',
                        mode=line_mode))
        
    if line_count == "Hot_Cold_Perfect_Medians":
        # split allMembers into cold, hot, and perfprog
        coldMembers = sub_df.loc[:, sub_df.columns.str.startswith('-')]             # Extract negative columns
        perfprogMembers = sub_df.loc[:, sub_df.columns.str.startswith('0.0')]       # Extract perfectProg column
        hotMembers = sub_df.loc[:, ~sub_df.columns.str.startswith(('-', '0.0'))]    # Extract positive columns

        cold_lower_bound = coldMembers.apply(min, axis=1)
        cold_upper_bound = coldMembers.apply(max, axis=1)
        cold_median = coldMembers.apply(np.median, axis=1)

        perfprogMembers_lower_bound = perfprogMembers.apply(min, axis=1)
        perfprogMembers_upper_bound = perfprogMembers.apply(max, axis=1)
        perfprogMembers_median = perfprogMembers.apply(np.median, axis=1)

        hotMembers_lower_bound = hotMembers.apply(min, axis=1)
        hotMembers_upper_bound = hotMembers.apply(max, axis=1)
        hotMembers_median = hotMembers.apply(np.median, axis=1)

        if fan == True:
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=cold_lower_bound[start_index:end_index+1], 
                    line=dict(color="blue"),
                    name=f'Cold Lower Bound',
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=cold_upper_bound[start_index:end_index+1], 
                    line=dict(color="blue"), 
                    fill='tonexty',
                    name=f'Cold Upper Bound',
                    mode=line_mode))
            
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=perfprogMembers_lower_bound[start_index:end_index+1], 
                    line=dict(color="yellow"),
                    name=f'Perfect Prog Lower Bound',
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=perfprogMembers_upper_bound[start_index:end_index+1], 
                    line=dict(color="yellow"), 
                    fill='tonexty',
                    name=f'Perfect Prog Upper Bound',
                    mode=line_mode))
            
            # graphing the lower region of the total model
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=hotMembers_lower_bound[start_index:end_index+1], 
                    line=dict(color="red"),
                    name=f'Hot Lower Bound',
                    mode=line_mode))
            
            # graphing the upper region of the total model 
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                    y=hotMembers_upper_bound[start_index:end_index+1], 
                    line=dict(color="red"), 
                    fill='tonexty',
                    name=f'Hot Upper Bound',
                    mode=line_mode))
            
        fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                y=cold_median[start_index:end_index+1], 
                line=dict(color="blue"),
                name=f'Cold Median',
                mode=line_mode))
        
        fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                y=perfprogMembers_median[start_index:end_index+1], 
                line=dict(color="#ba9c00"),
                name=f'Perfect Prog Median',
                mode=line_mode))
        
        fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                y=hotMembers_median[start_index:end_index+1], 
                line=dict(color="red"),
                name=f'Hot Median',
                mode=line_mode))

    if line_count == "All 450":
        for i in range(sub_df.shape[1]):
            fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                   y=sub_df.iloc[start_index:end_index+1, i],  
                   mode=line_mode,
                   name=df.columns[i+3]))
            

    fig.add_hline(y=8, line=dict(color='red'), name = 'cold-stun threshold') # cold stunning threshold

    if bool(onset["Earliest"]) == True:
        fig.add_vline(x=onset_start, line=dict(color='blue', dash="dot")) 
        fig.add_vline(x=onset_end, line=dict(color='blue', dash="dot")) 
        fig.add_vline(x=offset_start, line=dict(color='blue', dash="dot")) 
        fig.add_vline(x=offset_end, line=dict(color='blue', dash="dot")) 

    fig.add_trace(go.Scatter(x=date_time[start_index:end_index+1].astype(dtype=str), 
                y=df["Target"][start_index:end_index+1], 
                line=dict(color="black", dash="dot"),
                mode=line_mode,
                name="Target"))
    
    fig.update_layout(
                title=f'{line_count} {fan_percentile} %',
                xaxis_title='date_time',
                yaxis_title='water temperature (C)'
            )
    
    if save == True:
        import plotly
        plotly.offline.plot(fig, filename=f"{line_count}{save_path}.html")
    else:
        fig.show()

''' Ensemble time series grapher functions ends'''

''' Evaluation Metrics Begin


Metrics for Model Runs  *add other metrics (evaluation.py)

Author(s): Christian Duff 
'''


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

# Monoticity Fraction 
def mf(y_true, y_pred):
    import numpy as np
    
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()

    y_mean = np.mean(y_pred, axis=-1)
    y_std = np.std(y_pred, axis=-1)

    bins = None
    if bins is None:
        nbins = 10
        bins = np.linspace(0., 0.9, nbins)
    else:
        nbins = len(bins)

    ytrue1d = y_true.reshape(-1)
    ymean1d = y_mean.reshape(-1)
    ystd1d = y_std.reshape(-1)
    nsamples = ystd1d.shape[0]

    yrefs = np.argsort(ystd1d)
    ytrueSorted = ytrue1d[yrefs]
    ymeanSorted = ymean1d[yrefs]

    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    rmseOut = np.empty((nbins))
    for i in range(nbins):
        iCutoff = nsamples - int(nsamples * bins[i])
        ytrueH = ytrueSorted[:iCutoff]
        ymeanH = ymeanSorted[:iCutoff]
        rmseOut[i] = rmse(ytrueH, ymeanH)

    dtiA = np.empty((nbins - 1))
    for i in range(nbins - 1):
        dtiA[i] = (rmseOut[i] - rmseOut[i + 1])

    dtmf = 0.
    for i in range(nbins - 1):
        if rmseOut[i] >= rmseOut[i + 1]:
            indicator = 1.
        else:
            indicator = 0.
        dtmf += indicator
    dtmf *= 1 / (nbins - 1)
    
    mf_score = dtmf

    return mf_score 
  
# Discard Test Improvement 

def di(y_true, y_pred):
    import numpy as np
    
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()

    y_mean = np.mean(y_pred, axis=-1)
    y_std = np.std(y_pred, axis=-1)

    bins = None
    if bins is None:
        nbins = 10
        bins = np.linspace(0., 0.9, nbins)
    else:
        nbins = len(bins)

    ytrue1d = y_true.reshape(-1)
    ymean1d = y_mean.reshape(-1)
    ystd1d = y_std.reshape(-1)
    nsamples = ystd1d.shape[0]

    yrefs = np.argsort(ystd1d)
    ytrueSorted = ytrue1d[yrefs]
    ymeanSorted = ymean1d[yrefs]
    
    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    rmseOut = np.empty((nbins))
    for i in range(nbins):
        iCutoff = nsamples - int(nsamples * bins[i])
        ytrueH = ytrueSorted[:iCutoff]
        ymeanH = ymeanSorted[:iCutoff]
        rmseOut[i] = rmse(ytrueH, ymeanH)

    dtiA = np.empty((nbins - 1))
    for i in range(nbins - 1):
        dtiA[i] = (rmseOut[i] - rmseOut[i + 1])

    dtmf = 0.
    for i in range(nbins - 1):
        if rmseOut[i] >= rmseOut[i + 1]:
            indicator = 1.
        else:
            indicator = 0.
        dtmf += indicator
    dtmf *= 1 / (nbins - 1)
        
    di_score = np.mean(dtiA[:-1]) 
    
    return di_score 
  

# Ignorance Score 
def ign(y_true, y_pred): 
    import numpy as np
    import scipy

    def find_nearest(array, value, smaller=False, larger=False):
        if larger:
            # The implementation below is used so that when a value is halfway
            # between two points, the larger index is returned (like infind)
            return len(array) - np.abs((array - value))[::-1].argmin() - 1
            
        if smaller:
            return np.where(array <= value)[0][-1]
        
        return (np.abs(array - value)).argmin()

    probMin=0.0001
    nBins = 10
    nPts = y_true.shape[0]
    y_mean = np.mean(y_pred, axis=-1)
    y_std = np.std(y_pred, axis=-1)

    if len(y_true.shape) > 1:
        y_true = y_true.reshape(-1)
    if len(y_mean.shape) > 1:
        y_mean = y_mean.reshape(-1)
    if len(y_std.shape) > 1:
        y_std = y_std.reshape(-1)

    ymin = y_mean - 3. * y_std
    ymax = y_mean + 3. * y_std

    ign_score = 0.
    for i in range(nPts):
        minH = ymin[i]
        maxH = ymax[i]
        trueH = y_true[i]
        if trueH >= minH and trueH <= maxH:
            bin_edges = np.array([value for value in np.linspace(
                minH, maxH, num=nBins + 1)])
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            indx = find_nearest(bin_centers, trueH)

            dist = scipy.stats.norm(y_mean[i], y_std[i])
            probability = dist.pdf(bin_centers[indx])
            ign_score += np.log2(probability)

        else:
            ign_score += np.log2(probMin)

    ign_score = -1. * ign_score / nPts
    
    return ign_score 

# This is still in an investigation of accuracy stage 

# Spread Skill Reliability 
def ssrel(y_true, y_pred): 
    import tensorflow as tf 
    import numpy as np 

    if not isinstance(y_true, np.ndarray):
        y_true = y_true.numpy()
    
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()
    
    def create_contours(minVal, maxVal, nContours, match=False): 
        if match: 
            xVal = np.max([np.abs(minVal), np.abs(maxVal)]) 
            interval = 2 * xVal / (nContours - 1) 
        else: 
            interval = (maxVal - minVal) / (nContours - 1) 
        contours = np.empty((nContours)) 
        for i in range(nContours): 
            contours[i] = minVal + i * interval 
        return contours 
    
    nPts = y_true.shape[0] 
    y_pred_mean = tf.math.reduce_mean(y_pred, axis=-1) 
    y_std = np.std(y_pred, axis=-1) 
    minBin = np.min([0., y_std.min()]) 
    maxBin = np.ceil(np.max([rmse(y_true, y_pred), y_std.max()])) 
    
    nBins = 10 
    ssRel = 0. 
    error = np.zeros((nBins)) - 999. 
    spread = np.zeros((nBins)) - 999. 
    y_on_error = np.zeros((y_pred.shape)) - 999. 
    
    bins = create_contours(minBin, maxBin, nBins+1) 
    
    for i in range(nBins): 
        refs = np.logical_and(y_std >= bins[i], y_std < bins[i + 1]) 
        nPtsBin = np.count_nonzero(refs) 
        if nPtsBin > 0: 
            ytrueBin = y_true[refs] 
            ymeanBin = y_pred[refs] 
            error[i] = rmse(ytrueBin, ymeanBin) 
            spread[i] = np.mean(y_std[refs]) 
            y_on_error[refs] = np.abs(y_true[refs] - y_pred[refs]) 
            ssRel += (nPtsBin/nPts) * np.abs(error[i] - spread[i])

    score = ssRel

    return score



""" BREAK HERE"""


# function to get the stats using each member results (~ 20 min to run ?)
def member_metric(df):
    import numpy as np

    import pandas as pd

    print("Running Each Member Stats...\n")

    main_df = df # csv of all model run results

    num_rows = main_df.shape[0] # number of rows within dataframe (date_times)

    members = list(np.arange(-3.5, 4.0, 0.5)) # list of different members

    dfs = [] # hold a list of dfs, one df per member (will concat at the end)  
    
    for m in members: # iterating through each member (each member has 30 predictions per date_time)

        # dataframe to hold member metrics 
        df = pd.DataFrame(index=np.arange(num_rows), columns=['date_time', 'y_true', 'mean', 'std'])

        for i in range(num_rows): # iterating through each row (aka looping through each date_time water temp)
        

            df['date_time'].iloc[i] = main_df.iloc[i,1] # date_time
            df['y_true'].iloc[i] = main_df.iloc[i, 2] # true water temperature value
            df['mean'].iloc[i] = main_df.iloc[i, main_df.columns.str.startswith(str(m))].mean() # mean of predictions (each members mean)
            df['std'].iloc[i] = main_df.iloc[i, main_df.columns.str.startswith(str(m))].std()  # standard deviation of predictions (each members standard deviation)
            

        dfs.append(df) # appending the dataframe to the list of dataframes (dataframe per member)

    #metrics_df = pd.concat(dfs) # concatonating each dataframe that contains member stats into one dataframe

    return dfs

# function to get the stats using the 15 median member results (~ 10-15 seconds to run)
def med_metric(df):

    import pandas as pd
    print("Running Median Member Stats...\n")

    main_df = df 
    num_rows = main_df.shape[0]

    count = 1
    # dataframe to hold member metrics 
    df = pd.DataFrame(index=np.arange(num_rows), columns=['date_time', 'y_true', 'mean', 'std'])
    for i in range(num_rows):
        df['date_time'].iloc[i] = main_df.iloc[i,1] # date_time
        df['y_true'].iloc[i] = main_df.iloc[i, 2] # true water temperature value
        df['mean'].iloc[i] = main_df.iloc[i, 3:].mean() # mean of predictions (each members mean)
        df['std'].iloc[i] = main_df.iloc[i, 3:].std()  # standard deviation of predictions (each members standard deviation)
        count += 1

    return df

# function to get the stats using ALL 450 results (~ 10-20 seconds to run)
def all_metric(df):

    import pandas as pd
    #print("Running ALL Member Stats...\n")

    main_df = df # csv of all model run results

    num_rows = main_df.shape[0] # number of rows within dataframe (date_times)

    # dataframe to hold member metrics 
    df = pd.DataFrame(index=np.arange(num_rows), columns=['date_time', 'y_true', 'mean', 'std'])

    for i in range(num_rows): # iterating through each row (aka looping through each date_time water temp)
    

        df['date_time'].iloc[i] = main_df.iloc[i,1] # date_time
        df['y_true'].iloc[i] = main_df.iloc[i, 2] # true water temperature value
        df['mean'].iloc[i] = main_df.iloc[i, 2:].mean() # mean of predictions (all members mean)
        df['std'].iloc[i] = main_df.iloc[i, 2:].std()  # standard deviation of predictions (all members standard deviation)
        
    return df

# get the confidence interval of model runs (Miranda Approach)
def confidence_interval(confidence_percentage, mean, std_dev, sample_size):
    from math import sqrt

    from math import sqrt
    from math import sqrt
    z_scores = {"80": 1.28,
                "90": 1.645,
                "95": 1.96,
                "98": 2.33,
                "99": 2.575}
    
    score = z_scores[confidence_percentage]

    plus = mean + score * (std_dev/sqrt(sample_size))
    minus = mean - score * (std_dev/sqrt(sample_size))

    return plus, minus



''' Evaluation Metrics End'''

def unconventional_confidence_interval(confidence_percentage, sample_size, df):
    """Dr. Tissot Approach"""
    import pandas as pd
    import numpy as np

    percent = float(confidence_percentage) / 100
    
    num = int(sample_size * percent)
    num = int(num / 2)

    main_df = df # df of all model run results
    
    main_df = main_df.drop([main_df.columns[0], main_df.columns[1], main_df.columns[2]], axis=1)

    num_rows = main_df.shape[0] # number of rows within dataframe (date_times)
    values = pd.DataFrame(index=np.arange(num_rows), columns=['lower', 'upper'])

    for i in range(num_rows): # iterating through each row (aka looping through each date_time water temp)

        main_df.iloc[i].sort_values()

        values['lower'].iloc[i] = (main_df.iloc[i, num] + main_df.iloc[i, num+1]) / 2 # lower bound value

        values['upper'].iloc[i] = (main_df.iloc[i, -num] + main_df.iloc[i, -num-1]) / 2 # upper bound value

    return values


'''Weather Company Begin'''


'''
Weather Company Data

Author(s): Christian Duff 
'''

def weather_company_df_converter(path_to_year):
    ''' 
    - Each json file represents a datetime 
    - Within each file is 10 days of forecasts
    - Within each forecast is 100 members, each of which has 240 prototype forecasts
    '''

    import glob 
    import json
    from datetime import datetime
    import pandas as pd
    from warnings import simplefilter
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    print("Creating df of air temp predictions from the weather company...\n")

    start = datetime.now()
    list_of_jsons = glob.glob(f'{path_to_year}/*.json')
    dfs = []

    for k in range(len(list_of_jsons)):
        file = open(list_of_jsons[k])
        #issued_date_time = list_of_jsons[k][-18:-5]
        #year = path_to_year[-4:]

        data = json.load(file)

        #issued_date_time = datetime.fromtimestamp(data['metadata']['initTime']) # date time at which forecast was issued

        # forecast date times
        forecast_date_times = []

        for i in range(241): # 241 (10 days)
          
            forecast_date_times.append(utcDateTimeConverter(data['forecasts1Hour']['fcstValid'][i]))


        # print(data['forecasts1Hour']['prototypes'][0]['forecast'][99][240]) # [99] - 100 members [240] - 241 prototypes (predictions)

        df = pd.DataFrame()
        
        print(forecast_date_times)
        
        #Modified to account for function that converts from UTC time correctly
        df['forecast_date_time'] = forecast_date_times
        
        print(len(forecast_date_times[0]))
        
        #Unneeded with new function creation
        #df['forecast_date_time'] = df['forecast_date_time'].dt.strftime('%m-%d-%Y %H%M')

        for j in range(100):
            df[f'member_{j}'] = data['forecasts1Hour']['prototypes'][0]['forecast'][j]

        file.close()
        dfs.append(df)
    
    
    main_df = pd.concat(dfs)
    
    end = datetime.now()

    print(f'Total time to run: {end-start}\n')
    return main_df
    


'''Weather Company End'''