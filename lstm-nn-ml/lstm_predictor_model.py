##################################################################
#                                                                #
#      LSTM Predictor Model (lstm_predictor_model.py)            #
#                                                                #
#      A LSTM Time Series model for predicting the value         #
#      of future values of a univariate dataset.                 #
#                                                                #
#      By Louis Heery                                            #
#                                                                #
##################################################################

import math
import os
import warnings
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from pandas import DataFrame, Series, concat, datetime, read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from lstm_graph_plotter import find_anomaly_values, round_up, plotLSTMGraph

# define seasonality to by weekly, can replace with "day" for day long seasonality time series analysis
week_or_day_seasonality = "week"

# scale training&testing dataset to -1 to +1 scale
def scaleTrainTestDataset(train, test):
    scalerRange = (MinMaxScaler(feature_range=(-1, 1))).fit(train)
    trainScaled = scalerRange.transform(train.reshape(train.shape[0], train.shape[1]))
    testScaled = scalerRange.transform(test.reshape(test.shape[0], test.shape[1]))
    return scalerRange, trainScaled, testScaled

# Train LSTM model
def lstmTrain(train, batch_size, epochNumber, neuronNumber):
    print("TRAINING")
    X, Y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neuronNumber, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in np.linspace(0, epochNumber, epochNumber+1).astype(int):
        if (i % 50 == 0) or (i == 1):
            print("Training on Epoch " + str(i) + "/" + str(epochNumber))
        model.fit(X, Y, epochs = 1, batch_size = batch_size, verbose = 0, shuffle = False)
        model.reset_states()
    print("TRAINING FINISHED")
    return model

# LSTM Model to make 1 time unit forecast
def lstmForecast(model, batchSize, X):
    X = X.reshape(1, 1, len(X))
    predictedValue = model.predict(X, batch_size=batch_size)
    return predictedValue[0,0]

# LSTM Model Implementation
def lstmPredictorModel(usernumber):
    print("******")
    print("User Number = " + str(usernumber))
    print("******")

    filename = "lanl_dataset_user" + str(usernumber) + ".csv"
    user = np.genfromtxt(filename, delimiter=",")
    df = pd.DataFrame(user)

    # only use time series events of type '4634', i.e. "4634 - An account was logged on" on https://csr.lanl.gov/data/2017.html
    df_login = df[[0,1]].loc[df[1] == 4634]
    df_login.head()

    # bin dataset into hour long intervals
    minimum, maximum = df_login[0].min(), df_login[0].max()
    intervalLength = 60 * 60 # can be changed to specify interval length (units: seconds)
    intervalBins = np.linspace(minimum, maximum, (max - maximum)/intervalLength)
    binned_temp = pd.cut(df_login[0], intervalBins)
    binnedValues = (binned_temp.value_counts(sort=False)).as_matrix()
    time = np.delete(bins, -1)
    binnedDataset = np.vstack((time,binnedValues))
    binned_df = (pd.DataFrame({"time": binnedDataset[0], "actuals": binnedDataset[1]}))
    binned_df = binned_df.replace(0.0, 0.00000000001) # replace any 0.0 frequency bins in dataset with 0.00000000001 as 0.0 is not compatiable with log10 distortion
    actualValues = binned_df.actuals.values
    actualValuesLog = np.log10(actualValues)

    # prepare training values for training
    model_lag = 1
    temporary_df = DataFrame(actualValuesLog)
    columns = [temporary_df.shift(i) for i in np.linspace(0, model_lag+1, model_lag+2).astype(int)]
    columns.append(temporary_df)
    temporary_df = concat(columns, axis=1)
    temporary_df.fillna(0, inplace=True)
    supervised = temporary_df
    supervisedValues = supervised.values

    # define training&testing dataset size dependeing on seasonality setting
    if week_or_day_seasonality == "day":
        number_of_days = 7
        number_of_datapoints = number_of_days * 24
        trainData, testData = supervisedValues[0:number_of_datapoints], supervisedValues[number_of_datapoints:number_of_datapoints*2]

    if week_or_day_seasonality == "week":
        number_of_weeks = 2
        number_of_datapoints = number_of_weeks * 7 * 24
        trainData, testData = supervisedValues[0:number_of_datapoints], supervisedValues[number_of_datapoints:number_of_datapoints*2]

    # transform the scale of the data
    scalerRange, trainScaledLSTM, testScaledLSTM = scaleTrainTestDataset(train, testData)

    # train LSTM model using BatchSize = 1; EpochNumber = 850; NeuronNumber = 3
    lstm_model = lstmTrain(trainScaledLSTM, 1, 850, 3)
    # forecast the entire training dataset to build up state for forecasting
    trainForecasting = trainScaledLSTM[:, 0].reshape(len(trainScaledLSTM), 1, 1)

    # Implementation of LSTM Model to predict values over the Test dataset (testScaledLSTM)
    print("TESTING")
    predictionsArray = list()

    # cycle through entire test dataset
    for i in np.linspace(0, len(testScaledLSTM)-1, len(testScaledLSTM)).astype(int):

        if (i % 100 == 0) or (i == 1):
            print("Testing on " + str(i) + "/" + str(len(testScaledLSTM) - 1))

        # make forecast of the particular time step = i
        X, Y = testScaledLSTM[i, 0:-1], testScaledLSTM[i, -1]
        predictedValue = lstmForecast(lstm_model, 1, X)

        # apply invert scaling to the predicted values
        row = [x for x in X] + [predictedValue]
        temp_array = np.array(row)
        temp_array = temp_array.reshape(1, len(temp_array))
        inverted = scalerRange.inverse_transform(temp_array)
        predictedValue = inverted[0, -1]

        # save predicted values to array
        predictionsArray.append(10**predictedValue)
        expected = actualValuesLog[len(trainData) + i]
    print("TESTING FINISHED")

    # save data to individual arrays
    timeData = np.asarray(np.linspace(0, number_of_datapoints - 1, number_of_datapoints))
    trainData = np.asarray(dataset[1, 0:number_of_datapoints])
    testData =  np.asarray(dataset[1, number_of_datapoints:number_of_datapoints*2])
    predictedData = np.asarray(predictionsArray)

    # save train, test and predicted data to CSV file
    dataSaver = np.vstack((timeData,trainData,testData,predictedData))
    filename = 'lstm_results_user" + str(k) + ".csv'
    np.savetxt(filename, dataSaver, delimiter=',')

    # plot graph & save graph as PNG
    plotLSTMGraph(usernumber, timeData, trainData, testData, predictedData)

# iterate code through array of user numbers
for usernumber in (1,2,2255,6763,7449,8248,8710,19779):
    lstmPredictorModel(usernumber)

print("DONE")
