# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:03:49 2015

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

Y = train['Hazard']
X = train.drop(['Hazard'], axis = 1)

X = np.array(X)

# Encode categorical variables.
for i in range(X.shape[1]):
    if type(X[1,i]) is str:
        lbl = LabelEncoder()
        lbl.fit(list(X[:,i]))
        X[:,i] = lbl.transform(X[:,i])


#Create a training and validation set, 20% of the data will be reserved as the test set:

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

#We then split the test set again in order to have different data to train the base models and
#metamodels. Split the data equally.:

X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X_train, Y_train, test_size = 0.5) 

#This leaves 40% of the total data for the base models, 40% for training the meta models and 20% for testing the meta models.

#Fit a standard scaler:

std_scale = preprocessing.StandardScaler().fit(X_train1)


X_train1 = std_scale.transform(X_train1)
X_train2 = std_scale.transform(X_train2)
X_test = std_scale.transform(X_test)


#Base RF model:
from sklearn.ensemble import RandomForestRegressor
rf_mod = RandomForestRegressor(n_estimators = 100)
rf_mod.fit(X_train1, Y_train1)
rf_mod_predicted = rf_mod.predict(X_train2)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_train2, rf_predicted) #14.9

#Base SVM model
from sklearn.svm import SVR
svm_mod = SVR()

svm_mod.fit(X_train1, Y_train1)
svm_mod_predicted = svm_mod.predict(X_train2)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_train2, svm_predicted) #15.9

#Base extra trees model:
from sklearn.ensemble import ExtraTreesRegressor
et_mod = ExtraTreesRegressor()
et_mod.fit(X_train1, Y_train1)
et_mod_predicted = et_mod.predict(X_train2)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_train2, et_predicted) #16.62

#GBM Mod
from sklearn.ensemble import GradientBoostingRegressor
gbm_mod = GradientBoostingRegressor()
gbm_mod.fit(X_train1, Y_train1)
gbm_mod_predicted = gbm_mod.predict(X_train2)

from sklearn.metrics import mean_squared_error
mean_squared_error(Y_train2, gbm_predicted) #14.20


#Predictions for second training set combined into new dataframe:
combined_pred = pd.DataFrame({'rf_mod': rf_mod_predicted, 'svm_mod': svm_mod_predicted,
                              'et_mod': et_mod_predicted, 'gbm_mod': gbm_mod_predicted})


#Train a new regressor on the predictions. Target is Y_train2
from sklearn.ensemble import BaggingRegressor

rf_meta = RandomForestRegressor(n_estimators = 100)
rf_bag = BaggingRegressor(rf_meta, n_estimators = 50, oob_score = True)
rf_bag.fit(combined_pred, Y_train2)

#Test oob score:
#rf_bag.oob_score_


test_preds_rf_mod = rf_mod.predict(X_test)
test_preds_svm_mod = svm_mod.predict(X_test)
test_preds_et_mod = et_mod.predict(X_test)
test_preds_gbm_mod = gbm_mod.predict(X_test)

test_pred_df = pd.DataFrame({'rf_mod': test_preds_rf_mod, 'svm_mod': test_preds_svm_mod,
                              'et_mod': test_preds_et_mod, 'gbm_mod': test_preds_gbm_mod})

#Now predict and test accuracy.

rf_bag_predictions = rf_bag.predict(test_pred_df)
mean_squared_error(Y_test, rf_bag_predictions) #14.86




