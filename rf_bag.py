# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:02:22 2015

@author: Jake
"""

import os
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from __future__ import division
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

os.chdir('/Users/Jake/Kaggle/LibertyMutual/Data')

train = pd.read_csv('train.csv', index_col = 0)
test = pd.read_csv('test.csv', index_col = 0)
test_id = test.index


Y = train['Hazard']
X = train.drop(['Hazard'], axis = 1)

X = np.array(X)

# Encode categorical variables.
for i in range(X.shape[1]):
    if type(X[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(X[:,i]))
        X[:,i] = lbl.transform(X[:,i])

test = np.array(test)
for i in range(test.shape[1]):
    if type(test[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(test[:,i]))
        test[:,i] = lbl.transform(test[:,i])



std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)
test = std_scale.transform(test)

#Tranform Y variable:
Y_log = np.log(Y)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
rf_log_mod = RandomForestRegressor(n_estimators = 500)
rf_log_mod.fit(X, Y_log)

rf_log_mod_predicted = rf_log_mod.predict(test)

rf_log_mod_predicted = np.exp(rf_log_mod_predicted)

submission = pd.DataFrame({'Id': test_id, 'Hazard': rf_log_mod_predicted})
submission = submission.set_index('Id')
submission.to_csv('rf_log.csv')


#a simple blender:

blended_prediction = (gbm_bag_predicted + rf_log_mod_predicted)/2

submission = pd.DataFrame({'Id': test_id, 'Hazard': blended_prediction})
submission = submission.set_index('Id')
submission.to_csv('blender.csv')