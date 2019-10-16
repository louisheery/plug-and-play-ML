# XGBoost Boosted Decision Tree Classifier
# Author: Louis Heery

import pandas
import numpy
import sys
import pickle

import matplotlib.cm as cm
from sklearn.preprocessing import scale
from xgboost import XGBClassifier
from IPython.display import display
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading

start = time.time()

print("************")
print("STARTED XGBoost Classifier")

# Model Input Variables
variables = ['variable1', 'variable2', 'variable3']  # ENTER INPUT VARIABLES HERE = Column Names of CSV File

# Model Hyperparameters
n_estimators = 200
max_depth = 4
learning_rate = 0.15
subsample = 0.5

# Reading Data
dfEven = pd.read_csv('/training_dataset.csv') # ENTER DATASET 1 HERE
dfOdd = pd.read_csv('/testing_dataset.csv') # ENTER DATASET 2 HERE

# Initialise XGBoost Even and Odd Classifier
xgbEven = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample)
xgbOdd = XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample)

# Train XGBoost Even and Odd Classifier
xgbEven.fit(dfEven[variables], dfEven['Class'])
xgbOdd.fit(dfOdd[variables], dfOdd['Class'])

# Calculate Score of Trained BDT
scoresEven = xgbOdd.predict_proba(dfEven[variables])[:,1]
scoresOdd = xgbEven.predict_proba(dfOdd[variables])[:,1]

# Determine Predicted Decision Value of Trained BDT
dfEven['decision_value'] = ((scoresEven-0.5)*2)
dfOdd['decision_value'] = ((scoresOdd-0.5)*2)

# Combine Dataset of Odd and Even Model
df = pd.concat([dfEven,dfOdd])

# Save Predicted & Training Data to CSV File
df.to_csv("XGBoost_Predicted_Dataset.csv")


print("Time Taken", time.time() - start)
print("FINISHED")
print("************")
