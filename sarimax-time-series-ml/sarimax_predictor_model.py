##################################################################
#                                                                #
#      SARIMAX Predictor Model (sarimax_predictor_model.py)      #
#                                                                #
#      A SARIMAX Time Series model for predicting the value      #
#      of future values of a univariate dataset.                 #
#                                                                #
#      By Louis Heery                                            #
#                                                                #
##################################################################

import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults

from sarimax_graph_plotter import find_anomaly_values, round_up, plotSARIMAXGraph

# define seasonality to by weekly, can replace with "day" for day long seasonality time series analysis
week_or_day_seasonality = "week"

def lstmPredictorModel(usernumber):
    print("******")
    print("User Number = " + str(usernumber))
    print("******")

    # import user time series data -> to be used as training&testing time series
    filename = "/data-outputs/lanl_dataset_user" + str(usernumber) + ".csv"
    user = np.genfromtxt(filename, delimiter=",")
    df = pd.DataFrame(user)

    # only use time series events of type '4634', i.e. "4634 - An account was logged on" on https://csr.lanl.gov/data/2017.html
    df_login = df[[0,1]].loc[df[1] == 4634]
    df_login.head()

    # bin dataset into hour long intervals
    minimum, maximum = df_login[0].min(), df_login[0].max()
    intervalLength = 60 * 60 # can be changed to specify interval length (units: seconds)
    intervalBins = np.linspace(minimum, maximum, (maximum - minimum)/intervalLength)
    binned_temp = pd.cut(df_login[0], intervalBins)
    binnedValues = binned_temp.value_counts(sort=False).as_matrix()

    time = np.delete(bins, -1)
    binnedDataset = np.vstack((time,binnedValues))
    binned_df = (pd.DataFrame({"time": binnedDataset[0], "actual": binnedDataset[1]}))
    binned_df = binned_df.replace(0.0, 0.00000000001) # replace any 0.0 frequency bins in dataset with 0.00000000001 as 0.0 is not compatiable with log10 distortion
    actualValues = binned_df.actual.values

    # define seasonality duration of time series of dataset as 1 day or 1 week; and size of training&testing dataset
    if week_or_day_seasonality == "day":
        number_of_days = 7
        number_of_datapoints = number_of_days * 24
        train, test = actualValues[0:number_of_datapoints], actualValues[number_of_datapoints:number_of_datapoints*2]
        trainLog, testLog = np.log10(train), np.log10(test)
        modelOrder = (1, 1, 1)
        modelSeasonalOrder = (0, 1, 1, 24)

    if week_or_day_seasonality == "week":
        number_of_weeks = 2
        number_of_datapoints = number_of_weeks * 7 * 24
        train, test = actualValues[0:number_of_datapoints], actualValues[number_of_datapoints:number_of_datapoints*2]
        trainLog, testLog = np.log10(train), np.log10(test)

        # SARIMAX hyperparameters
        modelSeasonalOrder = (1, 0, 0, 24*7)
        # modelOrder = (1, 1, 1) # SLOWER (2X slower), better convergence
        modelOrder = (1, 1, 0) # COMPREMISE
        # modelOrder = (0, 0, 0) # FASTER, poorer convergence

    # initiate array to store predicted dataset
    historicDataset = [x for x in trainLog]
    predictionsArray = list()

    # iterate SARIMAX prediction model through entire testing data time series range (0, len(testLog))
    t = 0
    while t < len(testLog):
        if t % 10 == 0 or t == 1:
            print("Testing on " + str(t) + "/" + str(testLog - 1))
        model = sm.tsa.SARIMAX(historicDataset, order = modelOrder, seasonal_order = modelSeasonalOrder,enforce_stationarity = False,enforce_invertibility = False)
        modelFit = model.fit(disp=0)
        output = modelFit.forecast()
        predictedValue = 10**output[0]
        predictionsArray.append(predictedValue)
        newPrediction = testLog[t]
        historicDataset.append(newPrediction)
        t = t + 1

    # save model as .pkl file
    exec("modelFit.save('user" + str(usernumber) + "_model.pkl')")

    # save data to individual arrays
    timeData = np.asarray(np.linspace(0, number_of_datapoints - 1, number_of_datapoints))
    trainData = np.asarray(dataset[1, 0:number_of_datapoints])
    testData = np.asarray(dataset[1, number_of_datapoints:number_of_datapoints*2])
    predictedData = np.asarray(predictionsArray)

    # save train, test and predicted data to CSV file
    datasaver = np.vstack((timeData,trainData,testData,predictedData))
    filename = "/data-outputs/sarimax_results_user" + str(usernumber) + ".csv"
    np.savetxt(filename, datasaver, delimiter=',')

    # plot graph & save graph as PNG
    plotSARIMAXGraph(usernumber, timeData, trainData, testData, predictedData)

# iterate code through array of user numbers
for usernumber in (1,2,2255,6763,7449,8248,8710,19779):
    lstmPredictorModel(usernumber)

print("DONE")
