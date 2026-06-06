'''
Evaluation Begins

Author: Marina Vicens Miquel
Modified by: Hector Marrero-Colominas, Jarett T. Woodall, Christian Duff
'''
'''
Series of metrics imported from the file 'evaluation.py' whose original 
author was Marina Vicens Miquel & modified by Jarett Woodall
'''

#
#----HELPER FUNCTIONS FOR SOME CALCS-------
#

def simplePercentCalc(subLength, totalLength):
    """
    This function serves as a helper to the centralFreq and centralFreqBelow12 functions.

    subLength - parameter that takes in the length or "count" of the list of values within a 
    certain frequency

    totalLength - total length of the target list

    Created By: Jarett T. Woodall
    """

    import numpy as np
    
    # Calculates percent of values within a certain frequency
    percent = np.round((subLength / totalLength) * 100, 4)
    
    return percent
#END: def simplePercentCalc

def divideCheckMAECalc(cfLstPred, cfLstAct):
    """
    divideCheckMAECalc() is a helper function for the centralFreq function helping to modularize
    and make the function easier to read.

    Created By: Jarett T. Woodall
    """

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

def residualLstCreator(dictionary):
    """
    residualLstCreator() takes care of making a list from the error values,
    also returns top 10 percent from dates and prediction values.

    Created By: Jarett T. Woodall
    """

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

def centralFreq(actual, pred, controlSequence = "all"):
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

def errorBelow12c(y_true, y_pred):
    """
    errorBelow12c() computes the mean error and mae below 12 celsius. 
    It also returns a list of all the errors for both metrics. 
    These lists are necessary to later compute the std and std error
    
    Author: Jarett T. Woodall
    """

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
#END: def errorBelow12c()

def me12(y_true, y_pred):
    '''
    me12() computes only Mean Error below 12 degrees celsuis.
    Referenced Jaretts earlier version 'errorBelow12c' funciton. 
    Author: Hector M. Marrero-Colominas
    '''

    import numpy as np
    meanErrBelow12List = []
    
    for i in range(len(y_true)):
        # Creating lists that contains the mean error and mae below 12 celsius
        if (y_true[i] < 12): meanErrBelow12List.append(y_pred[i] - y_true[i])

    # Computing the mean error for the predictions
    return np.mean(meanErrBelow12List)
#END: def me12()

def max10PercentError(y_true, y_pred):
    """
    Compute the mean of the top 10% largest absolute residual errors.

    Self-contained replacement for the older max10PercError(): takes raw
    y_true/y_pred (computes residuals internally) and needs no `trials`.
    Author: Hector M. Marrero-Colominas

    Args:
        y_true (list or array): True values.
        y_pred (list or array): Predicted values.

    Returns:
        float: Mean of the top 10% largest absolute residual errors,
        rounded to 4 decimal places.
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

def iqrFunc(lst):
    """
    iqrFunc() takes a list and finds the IQR.

    Created By: Jarett Woodall
    """
    import numpy as np
    
    # Assigns relevant values to calculate iqr
    q3, q1 = np.percentile(lst, [75 ,25])
    
    # IQR calculation
    iqr = q3 - q1
    
    return iqr
# END: def iqrFunc()

def centralTendencyFinder(lst, trials):    
    """
    centralTendencyFinder(), takes a list or dictionary and outputs basic values

    Created By: Jarett Woodall
    """

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
#END: def percCorrect_1_deg_below12Func()

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
#END: def percCorrect_0_5_deg_below12Func()

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
#END: def crps()

def crps_gaussian_tf(mu, sigma, y_true):
    """
    Compute the CRPS for a Gaussian predictive distribution in TensorFlow.
   
    mu : tf.Tensor
        Predicted mean, shape [batch_size, ...]
    sigma : tf.Tensor
        Predicted std dev, shape [batch_size, ...]
    y_true : tf.Tensor
        True values, same shape as mu

    Note: CRPS normal distribution function and code - 
    Miranda sent in an email, I think she got this code from Ryan L
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
#END: def crps_gaussian_tf()

def mae12(y_true, y_pred):
    """
    Andrew's MAE (below) 12 (degress Celsuis) function'
    """

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
#END: def mae12()

def mae(y_true, y_pred):
    import tensorflow as tf

    return tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)), axis=-1)
#END: def mae()

def mse(y_true, y_pred):
    import tensorflow as tf

    return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=-1)
#END: def mse()

def me(y_true, y_pred):
    import tensorflow as tf

    return tf.reduce_mean(tf.subtract(y_true, y_pred), axis=-1)
#END: def me()

def rmse(y_true, y_pred):
    import tensorflow as tf

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)), axis=-1))
#END: def rmse()

def rmse_avg(y_true, y_pred): 
    import tensorflow as tf 

    mean_pred = y_pred 
    
    root_mean_square = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred)), axis=-1)) 
    
    score = tf.reduce_mean(root_mean_square) 
    
    return score.numpy() 
#END: def rmse_avg()

def ssrat(y_true, y_pred):
    """ 
    Spread Skill Ratio; SSRAT
    """
    import tensorflow as tf

    y_pred_std = tf.math.reduce_std(y_pred, axis=-1)

    ssrat_score = tf.math.reduce_mean(y_pred_std)/rmse(y_true, y_pred)
    
    return ssrat_score
#END: def ssrat()

def ssrat_avg(y_true, y_pred, y_std):
    import tensorflow as tf 
    ssrat_score = tf.math.reduce_mean(y_std)/rmse_avg(y_true, y_pred)

    return ssrat_score.numpy()
#END: def ssrat_avg()

def ssrel(y_true, y_pred):
    """
    Spread Skill Reliability
    """

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
#END: def ssrel()

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
#END: def ryan_ssrel()

def pitd(y_true, y_pred):
    """
    PITD: Probability Integral Transgform Distance
    """
    
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
#END: def pitd()

def member_metric(df):
    """
    function to get the stats using each member results (~ 20 min to run ?)
    """

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
#END: def member_metric()

def med_metric(df):
    """
    function to get the stats using the 15 median member results (~ 10-15 seconds to run)
    """

    import pandas as pd
    import numpy as np
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
#END: def med_metric()

def all_metric(df):
    """
    function to get the stats using ALL 450 results (~ 10-20 seconds to run)
    """
    
    import pandas as pd
    import numpy as np

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
#END: def all_metric()

def confidence_interval(confidence_percentage, mean, std_dev, sample_size):   
    """
    get the confidence interval of model runs (Miranda Approach)
    """    

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
#END: def confidence_interval()

