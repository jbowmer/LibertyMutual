# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:07:00 2015

@author: Jake
"""

#Liberty Mutual Kaggle Competition:

#Random Forest Feature Importance:

import os
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from __future__ import division
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder




os.chdir('/Users/Jake/Kaggle/LibertyMutual/Data')


data = pd.read_csv('train.csv')

Y_train = data['Hazard']
X_train = data.drop(['Hazard', 'Id'], axis = 1)

#encode all the categorical variables with LabelEncoder:

# Encode categorical variables.
for i in range(X_train.shape[1]):
    if type(X_train[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(X_train[:,i]))
        X_train[:,i] = lbl.transform(X_train[:,i])
        



rf_mod_gini = RandomForestRegressor(n_estimators = 100, n_jobs = 2, criterion = 'gini')
rf_mod_gini.fit(X_train, Y_train)

names = list(data)

print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf_mod_gini.feature_importances_), names), 
             reverse=True)

