# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:31:18 2015

@author: Jake
"""

#Blended GBM and RF Bag results:

#Recreate gbm benchmark:

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




#Base GBM model:
from sklearn.ensemble import GradientBoostingRegressor
gbm_mod = GradientBoostingRegressor(n_estimators = 100)
gbm_bag = BaggingRegressor(gbm_mod, n_estimators = 50, oob_score = True)
gbm_bag.fit(X, Y)
gbm_bag_predicted = gbm_bag.predict(test)


#Base RF model:
from sklearn.ensemble import RandomForestRegressor
rf_mod = RandomForestRegressor(n_estimators = 100)
rf_mod.fit(X, Y)
rf_mod_predicted = rf_mod.predict(test)

blended_prediction = (gbm_bag_predicted + rf_mod_predicted)/2

submission = pd.DataFrame({'Id': test_id, 'Hazard': blended_prediction})
submission = submission.set_index('Id')
submission.to_csv('blender.csv')