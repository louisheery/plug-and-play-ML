# XGBoost Boosted Decision Tree Classifier
# Author: Louis Heery

import pandas
import numpy
import sys
sys.path.append("../")
sys.path.append("../dataset-and-plotting")
import pickle

import matplotlib.cm as cm
from sklearn.preprocessing import scale

from ../bdtPlotting import *
from ../sensitivity import *
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading

def totalSensitivity(A,B,errorA,errorB):
    totalSensitivity = np.sqrt(A**2 + B**2)
    totalError = np.sqrt(((A*errorA)/np.sqrt(A**2 + B**2))**2 + ((B*errorB)/np.sqrt(A**2 + B**2))**2)

    return (totalSensitivity,totalError)

start = time.time()

for nJets in [2,3]:

print("************")
print("STARTED XGBoost Classifier")

if nJets == 2:

    variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'MV1cB1_cont', 'MV1cB2_cont', 'nTrackJetsOR',]

    n_estimators = 200
    max_depth = 4
    learning_rate = 0.15
    subsample = 0.5

else:

    variables = ['mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB', 'dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cB1_cont', 'MV1cB2_cont', 'MV1cJ3_cont','nTrackJetsOR',]

    n_estimators = 200
    max_depth = 4
    learning_rate = 0.15
    subsample = 0.5

# Reading Data
if nJets == 2:
    dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_even.csv')
    dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_2jet_odd.csv')

else:
    dfEven = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_even.csv')
    dfOdd = pd.read_csv('../dataset-and-plotting/CSV/VHbb_data_3jet_odd.csv')


xgbEven = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample)
xgbOdd = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample)



print("Training the " + str(nJets) + " Jet Dataset")
xgbEven.fit(dfEven[variables], dfEven['Class'], sample_weight=dfEven['training_weight'])

xgbOdd.fit(dfOdd[variables], dfOdd['Class'], sample_weight=dfOdd['training_weight'])

# Calculate Score of Trained BDT
scoresEven = xgbOdd.predict_proba(dfEven[variables])[:,1]
scoresOdd = xgbEven.predict_proba(dfOdd[variables])[:,1]

dfEven['decision_value'] = ((scoresEven-0.5)*2)
dfOdd['decision_value'] = ((scoresOdd-0.5)*2)
df = pd.concat([dfEven,dfOdd])
figureName = "XGBoost_" + str(nJets) + "Jets_" + str(n_estimators) + "estimators_" + str(max_depth) + "depth_" + str(learning_rate) + "learnrate.pdf"

h1, ax = final_decision_plot(df, figureName)

sensitivity2Jet = calc_sensitivity_with_error(df)
print(str(nJets) + " Jet using the Standard BDT: "+ str(sensitivity2Jet[0]) + " ± "+ str(sensitivity2Jet[1]))

print("Time Taken", time.time() - start)
print("FINISHED")
print("************")
