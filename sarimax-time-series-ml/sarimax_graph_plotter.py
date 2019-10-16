##################################################################
#                                                                #
#      SARIMAX Graph Plotter (sarimax_graph_plotter.py)          #
#                                                                #
#      For plotting Graph of results obtained from the           #
#      sarimax_predictor_model.py script.                        #
#                                                                #
#      By Louis Heery                                            #
#                                                                #
##################################################################

def findAnomalies(df,movingAverageRange):
    import numpy as np
    import pandas as pd

    df.replace([np.inf, -np.inf], np.NaN, inplace = True)
    df.fillna(0,inplace = True)
    df['error'] = df['actual'] - df['predicted']
    df['stdev'] = df['error'].rolling(window = movingAverageRange).std()
    df['mean'] = df['error'].rolling(window = movingAverageRange).mean()
    df['lowerbound'] = df['mean'] - (2 * df['stdev'])
    df['upperbound'] = df['mean'] + (2 * df['stdev'])
    df['anomaly_values'] = np.where(df['error'] < df['lowerbound'], df['error'], np.where(df['error'] > df['upperbound'], df['error'], np.nan))
    df = df.sort_values(by = 'time', ascending = False)
    return df

def round_up(n, decimals = 0):
    multiplier = 20 ** decimals
    return math.ceil(n * multiplier) / multiplier

week_or_day_seasonality = "week"

def plotSARIMAXGraph(usernumber, timeData, trainData, testData, predictedData):
    import math
    import os
    import warnings
    from math import sqrt

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    from matplotlib import pyplot as plt
    plt.switch_backend('agg')
    from pandas import DataFrame, Series, concat, datetime, read_csv
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler

    # dataframe to store predicted&testing values for detecting anomaly values
    predicted_df = pd.DataFrame()
    predicted_df['time'] = timeData
    predicted_df['actual'] = testData
    predicted_df['predicted'] = predictedData
    predicted_df.reset_index(inplace = True)
    del predicted_df['index']
    predicted_df.head()
    anomaly_df = findAnomalies(predicted_df,7)
    anomaly_df.reset_index(inplace=True)
    del anomaly_df['index']
    anomaly_df.head()

    # Set up subplot grid
    fig=plt.figure()
    gridspec.GridSpec(15,10)
    ax1 = plt.subplot2grid((17,10), (0,0), colspan = 10, rowspan = 8)
    ax2 = plt.subplot2grid((17,10), (9,0), colspan = 10, rowspan = 6)

    ##### Subplot 1 #####
    ax1.set_title("User " + str(k) + " - SARIMAX Time Series Predictor")
    ax1.plot(timeData/24, trainData, label='Training Data') # scale time scale by 24, i.e. from units of Hours to units of Days
    ax1.plot(timeData/24, testData, label='Test Data')
    ax1.plot(timeData/24, predictedData, color = "red",label='Predicted Data\n(of the Test Data)')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0,14)
    ax1.set_xticks(np.arange(0, 15, 1.0))
    ax1.set_ylabel("Host User Account Activity\n (Frequency per Bin)")
    ax1.grid()

    ##### Subplot 1 #####
    ax2.plot(dataarray[0]/24, test - dataarray[2], label='Residual (Test - Predicted)', color='green')
    ax2.plot(anomaly_df['time']/24, anomaly_df['anomaly_values'], 'o', label='Anomaly', color='red')
    ax2.fill_between(anomaly_df['time']/24, anomaly_df['lowerbound'], anomaly_df['upperbound'], alpha=0.1, label="3 STDEV Confidence Interval")
    ax2.set_xlabel("Time (Day)")
    ax2.set_xlim(0,14)
    ax2.legend(loc='best')
    ax2.set_xticks(np.arange(0, 15, 1.0))
    ax2.set_ylabel("Residual Activity\n (Frequency per Bin)")
    ax2.grid(True)

    ##### Combined Subplot setup #####
    h,l=ax2.get_legend_handles_labels() # get labels and handles from ax2
    ax2.get_shared_x_axes().join(ax2, ax1)
    ax1.set_xticklabels([])
    ax2.grid(True)
    ax1.grid(True)
    ax2.grid(which='minor', linestyle='-', linewidth='0.02', color='black')
    ax1.grid(which='minor', linestyle='-', linewidth='0.02', color='black')

    ##### Save Figure #####
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    filename = "SARIMAX_user" + str(k) + ".pdf"
    plt.savefig(filename, bbox_inches='tight',dpi=300)
