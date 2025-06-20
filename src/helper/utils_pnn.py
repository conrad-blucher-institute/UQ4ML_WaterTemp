
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
