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
# Importing libraries 
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from tensorflow import keras
import keras.backend as K


#from weather_company_functions import utcDateTimeConverter



# Creating a class where the AI architecture is defined
class mlp_model():
    
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

    print('inside readingData in utils. path_to_data:',path_to_data)
    csvs = glob.glob(f"{path_to_data}/*csv")
    
    # Reading years with extra rows (5 days back and 5 days after)
    #print(csvs)
    data_year1 = pd.read_csv(csvs[11])
    data_year2 = pd.read_csv(csvs[1])
    data_year3 = pd.read_csv(csvs[2])
    data_year4 = pd.read_csv(csvs[3])
    data_year5 = pd.read_csv(csvs[4])
    data_year6 = pd.read_csv(csvs[5])
    data_year7 = pd.read_csv(csvs[6])
    data_year8 = pd.read_csv(csvs[7])
    data_year9 = pd.read_csv(csvs[8])
    data_year10 = pd.read_csv(csvs[10])
    
    return data_year1, data_year2, data_year3, data_year4, data_year5, data_year6, data_year7, data_year8, data_year9, data_year10

# I added a default value for pred_atp_interval of 1 -hector (12/21/2024)
def creatingAdditionalColumns(df, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval=1, IPPOffset=0.0):
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
    
    
    #df.to_csv('datasetTest.csv', encoding='utf-8', index=False)
    
    return df

def splittingData(year1, year2, year3, year4, year5, year6, year7, year8, year9, year10, cycle):
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
                testing = pd.concat([testing, yearList[index]])

                
        if j == cycle:
            return training, testing, validation
        
        ### reset lists to empty
        training = pd.DataFrame()
        testing = pd.DataFrame()
        validation = pd.DataFrame()
    
def countingMissingValues(df):
    '''countingMissingValues() counting the rows that contains a missing value in at least one of the columns'''


    missingValues = df[df.isin([-999]).any(axis=1)]
    numMissValues = len(missingValues)
    percMissValues = (numMissValues/len(df))*100

    return numMissValues, percMissValues

def deletingMissingValues(df):
    '''deletingMissingValues() deleting the rows that at least one of the columns contain a missing value'''

    valueRemove = [-999]   
    df = df[df.isin(valueRemove) == False]
    df = df.dropna()
    
    return df

def reshaping(training, testing, validation, model):
    '''reshaping() reshaping the training, testing, and validation datasets to be 
    able to use them as an input for the AI model'''
    import numpy as np

    rangeValues = len(training.columns) - 1 # for marina, why is this variable here? -hector
    
    #print(testing['dateAndTime', 'packeryATP_lighthouse', 'npsbiWTP_lighthouse'].head(125))
    # Dividing the datasets between the inputs and the target
    trainingData = training.iloc[:,3:-1].values.astype(float) 
    trainingTarget = training.iloc[:,-1].values.astype(float)
    
    testingData = testing.iloc[:,3:-1].values.astype(float) 
    testingTarget = testing.iloc[:,-1].values.astype(float) 
    
    validationData = validation.iloc[:,3:-1].values.astype(float) 
    validationTarget = validation.iloc[:,-1].values.astype(float) 
    
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

def DateTimeRetriever(df, input_hours_forecast):
    '''This function is designed to grab the date times from the testing data set for computations.'''

    import pandas as pd

    Dates = [] # holds date times

    # Loop to add date times for future calculations
    df['dateAndTime'] = pd.to_datetime(df['dateAndTime'], format='%m-%d-%Y %H%M', yearfirst=False) + pd.DateOffset(hours=input_hours_forecast)

    Dates = df['dateAndTime'].tolist() # Converts series into a list

    return Dates


'''if there is no issue known, delete the deprecated functions (2 of them) below, and use the 1 above -hector Feb 5'''
# def testingDateTimeRetriever(testing, input_hours_forecast):
#     '''This function is designed to grab the date times from the testing data set for computations.'''

#     import pandas as pd

#     testingDates = [] # holds date times

#     #print(testing.head())import pandas as pd
#     #print(testing.head())import pandas as pd
    
#     # Loop to add date times for future calculations
#     testing['dateAndTime'] = pd.to_datetime(testing['dateAndTime'], format='%m-%d-%Y %H%M', yearfirst=False) + pd.DateOffset(hours=input_hours_forecast)
    
#     #print(testing['dateAndTime'].head())
    
#     # Ask if an offset is needed for the air temps
    
#     #Converts series into a list
#     testingDates  = testing['dateAndTime'].tolist()    
            
#     return testingDates

# def validationDateTimeRetriever(validation, input_hours_forecast):
#     import pandas as pd
#     '''This function is designed to grab the date times from the testing data set for computations.'''

#     import pandas as pd
#     import pandas as pd
#     # holds date times
#     validationDates = []
    
#     #print(testing.head())import pandas as pd
#     #print(testing.head())import pandas as pd
    
#     # Loop to add date times for future calculations
#     validation['dateAndTime'] = pd.to_datetime(validation['dateAndTime'], format='%m-%d-%Y %H%M', yearfirst=False) + pd.DateOffset(hours=input_hours_forecast)
    
#     #print(testing['dateAndTime'].head())
    
#     # Ask if an offset is needed for the air temps
    
#     #Converts series into a list
#     validationDates  = validation['dateAndTime'].tolist()    
            
#     return validationDates

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


"""
load2021testing() 
for PNN
author: hector
"""
def load_for_testing(input_hours_forecast, atp_hours_back, wtp_hours_back, dataset="cold", year=None): 
    import pandas as pd

    if year == None: raise NameError("year is None, need to receive year param in load_for_testing func")

    if dataset == "cold":
        var1 = "Cold"
        var2 = "coldseason"
    else:
        var1 = "Full"
        var2 = "year"

    # Reading years with extra rows (5 days back and 5 days after)
    match year:
        case 'simulated-1':
            data_year = pd.read_csv('./simulatedDataset/simulatedCSE_1314_-1.csv')

        case 'simulated-1':
            data_year = pd.read_csv('./simulatedDataset/simulatedCSE_1314_-2.csv')

        case 'simulated-1':
            data_year = pd.read_csv('./simulatedDataset/simulatedCSE_1314_-3.csv')

        case '2021':
            data_year = pd.read_csv(f"data\June_May_Datasets\june_atp_and_wtp_2020_2021_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv")
            
        case '2024':
            data_year = pd.read_csv(f"data\June_May_Datasets\june_atp_and_wtp_2023_2024_withExtraRows_INDEPENDENTTESTINGYEAR_MW.csv")
            
        case _:
            data_year = pd.read_csv('./' + var1 + '_Season_Input_Dataset/' + var2 + '_' + year + '_withExtraRows.csv')


    
        
    # Function call to create additional columns
    year_ = creatingAdditionalColumns(data_year, input_hours_forecast, atp_hours_back, wtp_hours_back)
    
    year__afterDelete = deletingMissingValues(year_)

    # year__afterDelete = year__afterDelete.reset_index(drop=True) # this resets the index of the df, so that the 120 lines that got erased (could be more) dont affect the index for later graphing
    
    # Dividing the datasets between the inputs and the target
    year__data = year__afterDelete.iloc[:,3:-1].values.astype(float) 
    year__target = year__afterDelete.iloc[:,-1].values.astype(float)

    # _datetime = pd.to_datetime(year__afterDelete['dateAndTime'], format='%m-%d-%Y %H%M') + pd.DateOffset(hours=input_hours_forecast)

    '''i dont think you need the offset here, when you load the df in dataPreparation.py, it is before you have done the predictions.
    Only after you have done the predictions do you then *need* to do the offset'''
    # _datetime = pd.to_datetime(year__afterDelete['dateAndTime'], format='%m-%d-%Y %H%M')     
    _datetime = DateTimeRetriever(year__afterDelete, input_hours_forecast)

    return year__data, year__target, _datetime
    


# I added a default value for pred_atp_interval of 1 -hector (12/21/2024)
def preparingData(path_to_data, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval=1, IPPOffset = 0.0, cycle = 0, model="MLP", date_time=False, verbose=0, val_date_time=False):
    '''preparingData() is the driver function'''
    # Importing libraries
    from datetime import datetime

    # Function call to read the data
    data_year1, data_year2, data_year3, data_year4, data_year5, data_year6, data_year7, data_year8, data_year9, data_year10 = readingData(path_to_data)


    # with open('time_for_offsetcreator', 'w') as file:
    #     totaltime = end_time - start_time
    #     file.write(str(totaltime))

    # Function call to create additional columns
    start_time = datetime.now()
    year1 = creatingAdditionalColumns(data_year1, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    
    
    #for col in year1.columns: 
    #    print(col)
    year2 = creatingAdditionalColumns(data_year2, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year3 = creatingAdditionalColumns(data_year3, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year4 = creatingAdditionalColumns(data_year4, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year5 = creatingAdditionalColumns(data_year5, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year6 = creatingAdditionalColumns(data_year6, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year7 = creatingAdditionalColumns(data_year7, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year8 = creatingAdditionalColumns(data_year8, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year9 = creatingAdditionalColumns(data_year9, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    year10 = creatingAdditionalColumns(data_year10, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset)
    if verbose > 0:
        print('finished input construction')
    end_time = datetime.now()
    
    #with open('time_for_inputconstruction', 'w') as file:
    #    totaltime = end_time - start_time
    #    file.write(str(totaltime) + ',\t\t\tlt,' + str(input_hours_forecast) + ',temp,' + str(IPPOffset) + ',cycle,' + str(cycle))

    # training_data, testing_data, validation_data = splittingData(IPPYear1, IPPYear2, IPPYear3, IPPYear4, IPPYear5, IPPYear6, IPPYear7, IPPYear8, IPPYear9, IPPYear10, cycle)
    training_data, testing_data, validation_data = splittingData(year1,   year2,   year3,   year4,   year5,   year6,   year7,   year8,   year9,   year10, cycle)
    if verbose > 0:
        print('finished splitting the data')
    #print(training_data)

    # Function call to count the number of missing values
    training_numMissingValues,      training_percMissVal    = countingMissingValues(training_data)
    testing_numMissingValues,       testing_percMissVal     = countingMissingValues(testing_data)
    validation_numMissingValues,    validation_percMissVal  = countingMissingValues(validation_data)
    
    #  Printing the number of missing values
    if verbose > 0:
        print()
        print(f"Testing Missing Values: {training_numMissingValues}, Percentatge: {round(training_percMissVal,4)} %")
        print(f"Testing Missing Values: {validation_numMissingValues}, Percentatge: {round(validation_percMissVal,4)} %")
        print(f"Testing Missing Values: {testing_numMissingValues}, Percentatge: {round(testing_percMissVal,4)} %")
        print()
        
        
    dataframe_checker(-999, [training_data, testing_data, validation_data]) # checking for any rogue number less than -999

    # Function call to delete the rows that at least one of the columns contain a missing value (-999)
    training = deletingMissingValues(training_data)
    testing = deletingMissingValues(testing_data)
    validation = deletingMissingValues(validation_data)

    dataframe_checker(-100, [training, testing, validation]) # checking for any rogue number less than -100
    

    # For new calculations created in the Summer of 2023
    testingDates = DateTimeRetriever(testing, input_hours_forecast)
    
    # To grab air temperatures
    testingAirTemps = testing['packeryATP_lighthouse'].tolist()
    
    if verbose > 0:
        print()
    
    # Function call to reshpe the dataset and prepare it to be used as in input for the neural network
    x_train, y_train, x_val, y_val, x_test, y_test = reshaping(training, testing, validation, model) 

    if val_date_time==True:
        validationDates = DateTimeRetriever(validation, input_hours_forecast)
        return x_train, y_train, x_val, y_val, x_test, y_test, testingDates, testingAirTemps, validationDates    

    return x_train, y_train, x_val, y_val, x_test, y_test, testingDates, testingAirTemps


''' Data Preparation End'''




'''Evaluation Begins

Author: Marina Vicens Miquel

Modified by: Hector Marrero-Colominas, Jarett T. Woodall, Christian Duff

'''


'''Series of metrics imported from the file 'evaluation.py' whose original author was Marina Vicens Miquel & modified by Jarett Woodall'''

#
#
#----HELPER FUNCTIONS FOR SOME CALCS-------
#
#
"""
This function serves as a helper to the centralFreq and centralFreqBelow12 functions.

subLength - parameter that takes in the length or "count" of the list of values within a 
certain frequency

totalLength - total length of the target list

Created By: Jarett T. Woodall
"""
def simplePercentCalc(subLength, totalLength):
    import numpy as np
    
    # Calculates percent of values within a certain frequency
    percent = np.round((subLength / totalLength) * 100, 4)
    
    return percent

#END: def simplePercentCalc

"""
divideCheckMAECalc() is a helper function for the centralFreq function helping to modularize
and make the function easier to read.

Created By: Jarett T. Woodall
"""
def divideCheckMAECalc(cfLstPred, cfLstAct):
    import numpy as np
    
    # Checks to see if a list is empty to avoid divide by zero error
    if len(cfLstPred) == 0:
        
        # Returns this to avoid error
        return 0, 0
    
    else:
        # Calls maeFunction to calculate mae on lists being passed
        cfMae = np.round(maeFunc(cfLstPred, cfLstAct)[1], 4) #CfMAEFirst
        #Should remove brackets from calculations
        #cfMae = cfMaeFirst[0]
        cfCount = len(cfLstPred)
        
    return cfMae, cfCount

#END: def divideCheckMAECalc()

"""
residualLstCreator() takes care of making a list from the error values,
also returns top 10 percent from dates and prediction values.

Created By: Jarett T. Woodall
"""
def residualLstCreator(dictionary):
    
    # Sotring data split from dictionary
    residualList = []
    dateTimeList = []
    predList = []
    
    # Holds top 10 percent pf dates and predictions
    pred10PercList = []
    dates10PercList = []
    
    # Iterator to grab residuals from inside the dictionary
    for key, item in dictionary.items():
        
        # Appends all residuals to a list
        residualList.append(item[1])
        dateTimeList.append(key)
        predList.append(item[0])
        
    # Boundary for iteration
    num10PerVal = int(len(residualList) * 0.1)
    
    #Iterates through and appends top 10 perc vals from COLD dates and predictions to lists
    for i in range(num10PerVal):
        pred10PercList.append(predList[i])
        dates10PercList.append(dateTimeList[i])
        
    return residualList, dates10PercList, pred10PercList

#END: def residualLstCreator()

"""
residualSplitterCalc() takes care of splitting the the residual dictionary
into a "cold" and "warm" dictionaries based on a midpoint.

Created By: Jarett Woodall

1) calculates midpoint for dictionary split
2) creates new sorted dictionary for use
3) sorts dictionary by second value in the value field
4) calls residualListCreator to split lists and find 10 percent of predictions and dates
5) calls centralTendencyFinder to find values for prediciton values 
5) calls 10percError function to calculate

returns: 10percError for warm and cold, and returns centralFreq data for cold and warm pred
"""
def residualSplitterCalc(residualsDict, trials):
    import numpy as np

    # List to store prediction values
    pred = []
    
    # Empty Dictionaries to hold new data split by midPoint
    coldDict = {}
    warmDict = {}
    
    # Grabs prediction data from residualsDict
    for key, value in residualsDict.items():
        pred.append(value[0])
        
    #This code will determine the midpoint of the datasets in order to split for seperate calcs
    midPoint = np.round(centralTendencyFinder(pred, trials)["Mean"])
    
    # Iterator structure that creates new dictionaries and assigns values
    for key, item in residualsDict.items():
        
        # Determines if a key will go into the cold or warm list
        if item[0] < midPoint:
            coldDict[key] = item
    
        elif item[0] >= midPoint:
            warmDict[key] = item                                                    

    # Lambda function to sort by the residual value which is at dict[x][1]
    newCold = dict(sorted(coldDict.items(), key=lambda item: item[1][1], reverse=True))
    newWarm = dict(sorted(warmDict.items(), key=lambda item: item[1][1], reverse=True))
    
    # Conversion of cold & warm dicts to residual lists
    coldResiduals, coldDates, coldPreds = residualLstCreator(newCold)
    warmResiduals, warmDates, warmPreds = residualLstCreator(newWarm)  
    
    # Calculates central tendency metrics for top 10Percent predicitons     
    coldPredResults10 = centralTendencyFinder(coldPreds, trials)
    warmPredResults10 = centralTendencyFinder(warmPreds, trials)
    
    # Calls max10PercError func and calculates result
    coldResult10PerError = max10PercError(coldResiduals)[0]
    warmResult10PerError = max10PercError(warmResiduals)[0]
        
    return coldResult10PerError, coldPredResults10, warmResult10PerError, warmPredResults10 

#END: def residualSplitterCalc()

#
#
# ------EVALUATION METRIC FUNCTIONS---------
#
#
"""
centralFreq(updated)() is a function that determines data distribution

Created By: Jarett T. Woodall

# User needs to pass either "all" for all the predictions (already initializes) or
"below12" to get the distributions for all the predictions below 12 degrees C

1) Creates lists for all the values (predicted or actual) to hold redistributed data
2) Small conditional structure to check control sequence
3) Checks difference to determine where the values will be added
4) Calls helper function for mae calculation and count
5) Returns a dictionary that was created to hold the values calculated by the function
"""
def centralFreq(actual, pred, controlSequence = "all"):
    
    # These values hold split lists holding values based on specs
    cf1LstPred = []
    cf1LstAct = []
    cf2LstPred = []
    cf2LstAct = []
    cf3LstPred = []
    cf3LstAct = []
    cf4LstPred = []
    cf4LstAct = []
    
    # These lists will hold the rest of the data that is only useful for calculating percentages
    restPred = []
    restAct = []
    
    # This value holds the length of the actual list
    actualLength = len(actual)
    
    # This structure checks control sequence value, it is responsible for changing function computations
    if controlSequence == "all":
        #If there are values even close to this range, we are doing something wrong
        temperature = 1000
        processedActual = actual
        processedPred = pred
        
    elif controlSequence == "below12":
        temperature = 12
        #Lists to hold processed data
        processedActual = []
        processedPred = []
        
        # Creates a new list for the values below 12 for correct calcs
        for i in range(actualLength):
            if actual[i] <= 12:
                #Appends values to modified lists
                processedActual.append(actual[i])
                processedPred.append(pred[i])
        actualLength = len(processedActual)
    
    # Iteration structure for going through data
    for i in range(actualLength):
        
        # Calculation for how far away a prediction value is from the target
        difference = abs(processedPred[i] - processedActual[i])

        # Conditional Structure for determining the placement of values based on Specs
        # Appends values if difference is less then or equal to 1
        if difference <= 1 and processedActual[i] < temperature:
            cf1LstPred.append(processedPred[i])
            cf1LstAct.append(processedActual[i])
            
        # Appends values if difference is less then or equal to 2
        if difference <= 2 and processedActual[i] < temperature:
            cf2LstPred.append(processedPred[i])
            cf2LstAct.append(processedActual[i])
        
        # Appends values if difference is less then or equal to 3
        if difference <= 3 and processedActual[i] < temperature:
            cf3LstPred.append(processedPred[i])
            cf3LstAct.append(processedActual[i])
        
        # Appends values if difference is less then or equal to 4
        if difference <= 4 and processedActual[i] < temperature:
            cf4LstPred.append(processedPred[i])
            cf4LstAct.append(processedActual[i])
            
        # For the rest of the data
        restPred.append(processedPred[i])
        restAct.append(processedActual[i])
    
    # Assigns and computes mae values for each 
    cf1mae, cf1Count = divideCheckMAECalc(cf1LstPred, cf1LstAct)
    cf2mae, cf2Count = divideCheckMAECalc(cf2LstPred, cf2LstAct)
    cf3mae, cf3Count = divideCheckMAECalc(cf3LstPred, cf3LstAct)
    cf4mae, cf4Count = divideCheckMAECalc(cf4LstPred, cf4LstAct)
    restMae, restCount = divideCheckMAECalc(restPred, restAct)
    
    # Function calls to calculate percentage of values in each frequency and the rest
    percent1 = simplePercentCalc(cf1Count, actualLength)
    percent2 = simplePercentCalc(cf2Count, actualLength)
    percent3 = simplePercentCalc(cf3Count, actualLength)
    percent4 = simplePercentCalc(cf4Count, actualLength)
    percentRest = simplePercentCalc(restCount, actualLength)
    
    # Dictionary Creation to hold results
    resultDict = {"cf1mae": cf1mae, "cf1Count": cf1Count, "cf1Percent": percent1,
                    "cf2mae": cf2mae, "cf2Count": cf2Count, "cf2Percent": percent2,
                    "cf3mae": cf3mae, "cf3Count": cf3Count, "cf3Percent": percent3,
                    "cf4mae": cf4mae, "cf4Count": cf4Count, "cf4Percent": percent4,
                    "restMae": restMae, "restCount": restCount, "percentRest": percentRest}
    
    return resultDict

#END: def centralFreqUpdated()


"""
errorBelow12c() computes the mean error and mae below 12 celsius. It also returns a list of all
the errors for both metrics. These lists are necessary to later compute the std and std error
"""
def errorBelow12c(y_true, y_pred):
    import numpy as np
    
    meanErrBelow12List = []
    maeBelow12List = []
    
    for i in range(len(y_true)):
        
        # Creating lists that contains the mean error and mae below 12 celsius
        if (y_true[i] < 12):
            residualBelow12 = y_pred[i] - y_true[i]
            meanErrBelow12List.append(residualBelow12)
            
            absResidual = abs(residualBelow12)
            maeBelow12List.append(absResidual)
    
    # Computing the mean error and mae for the predictions
    meanErrorBelow12 = np.mean(meanErrBelow12List)
    maeBelow12 = np.mean(maeBelow12List)
    
    return meanErrorBelow12,  maeBelow12, maeBelow12List

'''
me12() computes only Mean Error below 12 degrees celsuis.
Referenced Jaretts earlier version 'errorBelow12c' funciton. 
Author: Hector M. Marrero-Colominas
'''
def me12(y_true, y_pred):
    import numpy as np
    meanErrBelow12List = []
    
    for i in range(len(y_true)):
        # Creating lists that contains the mean error and mae below 12 celsius
        if (y_true[i] < 12): meanErrBelow12List.append(y_pred[i] - y_true[i])

    # Computing the mean error for the predictions
    return np.mean(meanErrBelow12List)


"""
max10PercError() computes the mean error for the 10% worst predictions. It also retunrs a list
with all the mean absolute errors for the top 10% worst predictions
"""
# this func might be obsolete now -hector
def max10PercError(residuals, trials):
    import numpy as np
    
    absResidualsList = []
    
    # Creating a list for the absolute residuals
    for j in range(len(residuals)):
        
        # Why is it temp = abs(residuals[j][0]) The [0] throws everything out of wack
        # Float is non-subscriptable
        temp = abs(residuals[j])
        absResidualsList.append(temp)
    
    # Sorting the list of the absolute residuals reversely
    sortedReverseResiduals = absResidualsList.copy()
    sortedReverseResiduals.sort(reverse=True)
  
    # Creating a list that contains the top 10% worst predictions
    max10PerErrList = []
    num10PerVal = int(len(sortedReverseResiduals) * 0.1)
    
    for i in range(num10PerVal):
        max10PerErrList.append(sortedReverseResiduals[i])
        
    # Computes various metric calculations
    results = centralTendencyFinder(max10PerErrList, trials)
    
    max10PercError = results['Mean']
    
    return max10PercError, results #, max10PerErrList

"""
max10PercError() computes the mean error for the 10% worst predictions. It also retunrs a list
with all the mean absolute errors for the top 10% worst predictions
Referenced Jaretts earlier version 'max10PercError' funciton. 
Author: Hector M. Marrero-Colominas 
"""
def max10PercentError(y_true, y_pred):
    """
    Calculate the mean of the top 10% largest absolute residual errors from a list of residuals.
    
    Args:
        residuals (list or array): List of residual errors (could be negative or positive).
        trials (int): The number of trials (not used in this calculation but could be used for further processing).
        
    Returns:
        float: The mean of the top 10% largest absolute residual errors, rounded to 4 decimal places.
    """

    import numpy as np

    # Step 1: Compute the residuals of each prediction in the input list
    residuals = [y_pred[i] - y_true[i] for i in range(len(y_pred))]

    # Step 2: Compute the absolute value of each residual in the input list
    absResiduals = [abs(residual) for residual in residuals] 

    # Step 3: Sort the absolute residuals in descending order and get the top 10% worst residuals
    top10Percent = sorted(absResiduals, reverse=True)[:max(1, int(len(absResiduals) * 0.1))]

    # Step 4: Calculate and return the mean of the top 10% of residuals, rounded to 4 decimal places
    return np.round(np.mean(top10Percent), 4)

#END: def max10PercErrorFunc()

"""
iqrFunc() takes a list and finds the IQR.

Created By: Jarett Woodall
"""
def iqrFunc(lst):
    import numpy as np
    
    # Assigns relevant values to calculate iqr
    q3, q1 = np.percentile(lst, [75 ,25])
    
    # IQR calculation
    iqr = q3 - q1
    
    return iqr

# END: def iqrFunc()


"""
centralTendencyFinder(), takes a list or dictionary and outputs basic values

Created By: Jarett Woodall
"""
def centralTendencyFinder(lst, trials):
    import numpy as np
    from math import sqrt
    import statistics as stats
    
    # Stats for prediction list
    median = np.round(stats.median(lst), 4)
    mean = np.round(np.mean(lst), 4)
    calcIQR = np.round(iqrFunc(lst), 4)
    rangeCalc = np.round((max(lst) - min(lst)), 4)
    stDev = np.round(np.std(lst), 4)
    standard_error = stDev/sqrt(trials)
    count = len(lst)
        
        
    return {"Median": median, 
            "Mean": mean, 
            "IQR": calcIQR, 
            "Range": rangeCalc, 
            "StandardDev": stDev, 
            "2 Standard Error": 2*standard_error,
            "Count": count, 
            "Max": max(lst),
            "Min": min(lst)
            }
    
#END: def centralTendencyFinder()

def percCorrect_1_deg_below12Func(y_test, prediction):
    
    correct = 0
    incorrect = 0
    
    for i in range(len(y_test)):
        if y_test[i] < 12:    
            error = abs(y_test[i] - prediction[i])
            
            if error <= 1.0:
                correct = correct + 1
                
            else:
                incorrect = incorrect + 1
    
    perc = (correct / (correct + incorrect)) * 100
    
    return perc

def percCorrect_0_5_deg_below12Func(y_test, prediction):
    
    correct = 0
    incorrect = 0
    
    for i in range(len(y_test)):
        if y_test[i] < 12:    
            error = abs(y_test[i] - prediction[i])
            
            if error <= 0.5:
                correct = correct + 1
                
            else:
                incorrect = incorrect + 1
    
    perc = (correct / (correct + incorrect)) * 100
    
    return perc   

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

def calcThreshold_of_CS_event(path, datetime_split, save=False):
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
    
    if save == True:
        with open('ensemble_threshold', 'w') as file:
            file.write("start " + str(start_threshold) +' '+ str(max(list(startIndex.values())) - min(list(startIndex.values()))) + " hours range")
            file.write('\n')
            file.write("end " + str(end_threshold) +' '+ str(max(list(endingIndex.values())) - min(list(endingIndex.values()))) + " hours range")
    
    return start_threshold, end_threshold

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

from numpy import arange
def fanGraph(df_allMembers, saver, start_date, end_date, onset, offset, line_count="Total Model Median", y_min=2, y_max=20, members=list(arange(-3.5, 4.0, 0.5)), cycle="0", leadtime="12", fan = True):  
    
    import matplotlib.dates as mdates
    import pandas as pd
    import matplotlib.pyplot as plt

    colorList = ['mediumblue', 'teal', 'cadetblue', 'cornflowerblue', 'deepskyblue', 'skyblue', 'lightsteelblue', 'gold', 'pink', 'lightsalmon', 'lightcoral', 'palevioletred', 'orangered', 'indianred', 'maroon']

    save = '_cycle_' + cycle + '_' + leadtime + 'h_' # variable used to create the name while saving files
    if fan == False:    save += '_spaghetti'
    elif fan == True:   save += "_fans"

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

    if saver == True:       plt.savefig(line_count + save, bbox_inches='tight')
    elif saver == False:    plt.show()

    plt.close()

# christian, i think having the same name is problamatic, either refactor so that 1 func works and is backwards compatible, or keep keep but rename the second, wdyt? -hector
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

# Continuous Rank Probability Score Metric
def crps(y_true, y_pred):
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

# CRPS normal distribution function and code - Miranda sent in an email, I think she got this code from Ryan L
def crps_gaussian_tf(mu, sigma, y_true):
    """
    Compute the CRPS for a Gaussian predictive distribution in TensorFlow.
   
    mu : tf.Tensor
        Predicted mean, shape [batch_size, ...]
    sigma : tf.Tensor
        Predicted std dev, shape [batch_size, ...]
    y_true : tf.Tensor
        True values, same shape as mu
    """

    import tensorflow as tf
    import numpy as np

    # Clip sigma to avoid division by zero
    sigma = tf.clip_by_value(sigma, 1e-8, 1e8)

    # z = (y - mu) / sigma
    z = (y_true - mu) / sigma

    # Standard normal PDF and CDF
    # tfp.distributions.Normal supports both pdf and cdf
    import tensorflow_probability as tfp
    normal = tfp.distributions.Normal(loc=0.0, scale=1.0)

    pdf_z = normal.prob(z)  # varphi(z)
    cdf_z = normal.cdf(z)   # Phi(z)

    # CRPS = sigma * [ z(2Phi(z)-1) + 2 pdf(z) - 1/sqrt(pi) ]
    two_cdf_z_minus_1 = 2.0 * cdf_z - 1.0
    term1 = z * two_cdf_z_minus_1
    term2 = 2.0 * pdf_z
    term3 = 1.0 / tf.sqrt(tf.constant(np.pi, dtype=z.dtype))

    crps_per_sample = sigma * (term1 + term2 - term3)

    # Return mean CRPS across the batch
    return tf.reduce_mean(crps_per_sample)


"""
Andrew's MAE (below) 12 (degress Celsuis) function'
"""
def mae12(y_true, y_pred):
    import tensorflow as tf

    # Ensure consistent data types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
        
    # Create mask for values <= 12
    mask = tf.less_equal(y_true, 12)
        
    try: 
        # Apply mask
        filtered_true = tf.boolean_mask(y_true, mask)
        filtered_pred = tf.boolean_mask(y_pred, mask)
        
        # Compute MAE
        return tf.reduce_mean(tf.abs(filtered_pred - filtered_true)).numpy()
        
    except ValueError:
        return -999

def mae(y_true, y_pred):
    import tensorflow as tf

    return tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)), axis=-1)

def mse(y_true, y_pred):
    import tensorflow as tf

    return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=-1)

def me(y_true, y_pred):
    import tensorflow as tf

    return tf.reduce_mean(tf.subtract(y_true, y_pred), axis=-1)

def rmse(y_true, y_pred):
    import tensorflow as tf

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=-1))

def rmse_avg(y_true, y_pred): 
    import tensorflow as tf 


    mean_pred = y_pred 
    
    root_mean_square = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred)), axis=-1)) 
    
    
    score = tf.reduce_mean(root_mean_square) 
    
    return score.numpy() 

# Spread Skill Ratio; SSRAT
def ssrat(y_true, y_pred):
    import tensorflow as tf

    y_pred_std = tf.math.reduce_std(y_pred, axis=-1)

    ssrat_score = tf.math.reduce_mean(y_pred_std)/rmse(y_true, y_pred)
    
    return ssrat_score

def ssrat_avg(y_true, y_pred, y_std):
    import tensorflow as tf 
    ssrat_score = tf.math.reduce_mean(y_std)/rmse_avg(y_true, y_pred)

    return ssrat_score.numpy()

# Spread Skill Reliability
def ssrel(y_true, y_pred):
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    
    y_pred_std = tf.math.reduce_std(y_pred, axis=-1)
    
    num_bins = 10
    min_edge = tf.reduce_min(y_pred_std)
    max_edge = tf.reduce_max(y_pred_std)
    bin_boundaries = tf.linspace(min_edge, max_edge, num_bins + 1)  # Generate bin edges as a Python list
    
    bins = len(bin_boundaries) - 1  # Number of bins based on the boundary count
    bin_boundaries_list = [bound.numpy() for bound in bin_boundaries]  # Convert to a list

    # Ensure inputs are tensors with the correct type
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_pred_std = tf.convert_to_tensor(y_pred_std, dtype=tf.float32)
    
    # print(f"Shape of y_true: {y_true.shape}")
    # print(f"Shape of y_pred: {y_pred.shape}")
    # print(f"Shape of y_std: {y_pred_std.shape}")
    
    # Use tf.raw_ops.Bucketize with the converted list
    bin_indices = tf.raw_ops.Bucketize(input=y_pred_std, boundaries=bin_boundaries_list)
    
    # Debug: Print bin indices
    # print(f"Bin Indices: {bin_indices}")
    
    # Calculate mean predictions (y_pred) for each observation
    y_pred_mean = tf.reduce_mean(y_pred, axis=-1)
    
    # Initialize variables to accumulate SSREL
    ssrel_score = 0.0
    total_observations = tf.cast(tf.shape(y_true)[0], tf.float32)
    
    # Loop through each bin index to compute SSREL for each bin
    for bin_idx in range(bins):
        # Get indices for current bin
        indices_in_bin = tf.where(tf.equal(bin_indices, bin_idx))
        
        # Check if there are any indices in the bin
        num_indices = tf.shape(indices_in_bin)[0]
        
        # print(f"Bin {bin_idx}: Number of indices in bin = {num_indices}")
        
        if num_indices == 0:
            # If there are no indices in the bin, skip to the next bin
            continue
        
        # Extract the first column for the actual indices (only if we have indices)
        indices_in_bin = tf.squeeze(indices_in_bin)
        
        # Extract observations and predictions in the current bin
        obs_in_bin = tf.gather(y_true, indices_in_bin)
        preds_in_bin = tf.gather(y_pred, indices_in_bin)
        mean_preds_in_bin = tf.gather(y_pred_mean, indices_in_bin)
        
        # Calculate the number of observations in the bin
        Nk = tf.cast(tf.size(obs_in_bin), tf.float32)
        
        # Calculate RMSE_k
        rmse_k = tf.sqrt(tf.reduce_mean((obs_in_bin - mean_preds_in_bin) ** 2))
        
        # Calculate SD_k (spread of the ensemble predictions)
        mean_preds_in_bin_expanded = tf.expand_dims(mean_preds_in_bin, axis=-1)
        sd_k = tf.sqrt(tf.reduce_mean(tf.reduce_mean((preds_in_bin - mean_preds_in_bin_expanded) ** 2, axis=1)))
        
        # Calculate |RMSE_k - SD_k|
        abs_diff = tf.abs(rmse_k - sd_k)
        
        # Calculate the weighted contribution to SSREL Nk / N
        weight = Nk / tf.cast(total_observations, tf.float32)
        ssrel_score += weight * abs_diff
    
    # print(f"Final SSREL Score: {ssrel_score}")
    return ssrel_score

def ryan_ssrel(y_true, y_pred, y_std=None): 
    import tensorflow as tf 
    import numpy as np 

    y_true = tf.expand_dims(y_true, axis=-1) 

    if not isinstance(y_true, np.ndarray):
        # y_true = y_true.numpy()
        y_true = np.array(y_true)
    
    if not isinstance(y_pred, np.ndarray):
        # y_pred = y_pred.numpy()
        y_pred = np.array(y_pred)
    
    if not isinstance(y_std, np.ndarray):
        # y_pred = y_pred.numpy()
        y_std = np.array(y_std)
    
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
    # if y_std == None:
    if y_std is None:
        y_std = np.std(y_pred, axis=-1) 
    minBin = np.min([0., y_std.min()]) 
    
    print()
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

# PITD: Probability Integral Transgform Distance
def pitd(y_true, y_pred):
    import tensorflow as tf
    
    bins = 10

    y_pred_mean = tf.reduce_mean(y_pred, axis=-1)
    
    Nk = tf.histogram_fixed_width(y_pred_mean, value_range=[0.0, 38.0], nbins=bins)
    N = tf.reduce_sum(Nk)

    Nk = tf.cast(Nk, dtype=tf.float32)
    N = tf.cast(N, dtype=tf.float32)

    proportion = Nk / N

    ideal_proportion = 1.0 / bins

    pitd_score = tf.sqrt(tf.reduce_mean((proportion - ideal_proportion) ** 2))

    return pitd_score


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


'''Jaretts weather_company_GUI -- src/cross_validation_visuals.py
modified by hector for PNN'''
###----GRAPHING FUNCTIONS-----###
"""
Creates boxplot visuals for data
"""
def box(df, index, y_true, leadTime, architecture, cycle, save):
    import plotly.graph_objects as go 
    
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    
    count = 0
    
    fig = go.Figure()
    
    opacityValue = 1.0
    

    
    color = colors[count]
    
    # if architecture == "CRPS":
    #     customda = df[['target', 'crps', 'ssrat', 'mae', 'mae<12','central_mae','central_mae<12']]
    #     hovertemp = "<br>".join([
    #                 "date_time: %{x}",
    #                 "Combination: " + key,
    #                 "Mean Predicted Temperature (" + chr(176)+ "C): %{y}",
    #                 "Actual temperature (" + chr(176)+ "C): %{customdata[0]}",
    #                 "CRPS (" + chr(176) + "C): %{customdata[1]}",
    #                 "SSRAT (" + chr(176) + "C): %{customdata[2]}",
    #                 "Indiv_MAE (" + chr(176) + "C): %{customdata[3]}",
    #                 "Indiv_MAE<12 (" + chr(176) + "C): %{customdata[4]}",
    #                 "Central_MAE (" + chr(176) + "C): %{customdata[5]}",
    #                 "Central_MAE<12 (" + chr(176) + "C): %{customdata[6]}"
                    
    #                 ])
        
    # else:
    #     customda = df[['target', 'rmse', 'mae', 'mae<12', 'central_mae','central_mae<12']]
    #     hovertemp = "<br>".join([
    #                 "date_time: %{x}",
    #                 "Combination: " + "PNN",
    #                 "Mean Predicted Temperature (" + chr(176)+ "C): %{y}",
    #                 "Actual temperature (" + chr(176)+ "C): %{customdata[0]}",
    #                 "Combination: " + "PNN",
    #                 "Indiv_RMSE (" + chr(176) + "C): %{customdata[1]}",
    #                 "Indiv_MAE (" + chr(176) + "C): %{customdata[2]}",
    #                 "Indiv_MAE<12 (" + chr(176) + "C): %{customdata[3]}",
    #                 "Central_MAE (" + chr(176) + "C): %{customdata[4]}",
    #                 "Central_MAE<12 (" + chr(176) + "C): %{customdata[5]}"

    #                 ])
        
    #Median
    fig.add_trace(go.Scatter(
        x=index,
        y = df.iloc[:, 2].round(3),
        name="PNN" + " Mean",
        # customdata = customda,
        # hovertemplate = hovertemp,        
        
        marker=dict(
                    color=color, line_width=100
                ),
                mode='lines',
                showlegend=True, opacity=opacityValue
            ))
    
    #Adds traces and specified quartiles according to user preference
    fig.add_trace(go.Box(
        x=index,
        q1=df['q1'],
        q3=df['q3'],
        median=df['Median'],
        upperfence=df['max'],
        lowerfence=df['min'],
        name = "PNN" + " Box",
        showlegend=True,
        boxpoints=False,
        marker_color=color,opacity=opacityValue
    )) 
    
    #Decrements opacity value
    opacityValue -= 0.1
    
    # Resets opacity level if 
    if opacityValue == 0:
        opacityValue = 1
    count += 1
    






    # Adds Actual Line
    fig.add_trace(go.Scatter(
        x=index,
        y=y_true.round(3),
        name="Actual Water Temperature",
        
        marker=dict(
                    color='black', line_width=100
                ),
                mode='lines',
                showlegend=True
                ))
    
    # Add threshold ;ine
    fig.add_hline(y=8, line_dash="dot", line_color="red",annotation_text="Turtle Threshold", 
                    annotation_position="top left",
                    annotation_font_size=13,
                    annotation_font_color="red"
                    )
    
    fig.update_layout(title= 'BoxPlot_'+ str(leadTime) + "h_Cycle_" + str(cycle),
                        xaxis_title='DateTime',
                        yaxis_title='Temperature (' + chr(176)+'C)'
                        )
    
    save_path =  architecture +"_" +str(leadTime) + "h_Cycle_" + str(cycle)     
    if save == True:
        from pathlib import Path
        p = Path(architecture + "_BoxPlot/")
        # p = pathlib.Path(architecture + "_BoxPlot/")
        p.mkdir(parents=True, exist_ok=True)
        fig.write_html( p / f"{save_path}.html")
        
    else:
        fig.show()

#END: def box()