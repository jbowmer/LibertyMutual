# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:57:50 2015

@author: Jake
"""

#Liberty Mutual Base models:

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

os.chdir('/Users/Jake/Kaggle/LibertyMutual/Data')

####Read in Data


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


#Create a training and validation set, 20% of the data will be reserved as the test set:

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

#We then split the test set again in order to have different data to train the base models and
#metamodels. Split the data equally.:

X_ensemble, X_blend, Y_ensemble, Y_blend = train_test_split(X_train, Y_train, test_size = 0.5) 

#This leaves 40% of the total data for the base models, 40% for training the meta models and 20% for testing the meta models.

#Fit a standard scaler:

std_scale = preprocessing.StandardScaler().fit(X_ensemble)


X_ensemble = std_scale.transform(X_ensemble)
X_blend = std_scale.transform(X_blend)
X_test = std_scale.transform(X_test)

test = std_scale.transform(test)


#####Construct Base Models:

####Gradient Boosting Bag on Log(Y)
gbm_log_mod = GradientBoostingRegressor(n_estimators = 500)
gbm_log_bag = BaggingRegressor(gbm_log_mod, n_estimators = 50, oob_score = True)
gbm_log_bag.fit(X_ensemble, np.log(Y_ensemble))



####Random Forest - Log(Y)
rf_log_mod = RandomForestRegressor(n_estimators = 500)
rf_log_mod.fit(X_ensemble, np.log(Y_ensemble))


#### SVM - Log(Y)
svm_log_mod = SVR()
svm_log_mod.fit(X_ensemble, np.log(Y_ensemble))

#### ET - Log(Y)
et_log_mod = ExtraTreesRegressor()
et_log_mod.fit(X_ensemble, np.log(Y_ensemble))

###Elastic net - Log(Y)
enet_mod = ElasticNet(alpha = 0.1, l1_ratio = 0.7)
enet_mod.fit(X_ensemble, np.log(Y_ensemble))

###Lasso - Log(Y)

lasso_mod = Lasso(alpha = 0.1)
lasso_mod.fit(X_ensemble, np.log(Y_ensemble))

#### GBM model:
gbm_mod = GradientBoostingRegressor(n_estimators = 100)
gbm_bag = BaggingRegressor(gbm_mod, n_estimators = 50, oob_score = True)
gbm_bag.fit(X_ensemble, Y_ensemble)

#### RF model:
rf_mod = RandomForestRegressor(n_estimators = 100)
rf_mod.fit(X_ensemble, Y_ensemble)

#### SVM Model:
svm_mod = SVR()
svm_mod.fit(X_ensemble,Y_ensemble)

#### ET Mod
et_mod = ExtraTreesRegressor()
et_mod.fit(X_ensemble, Y_ensemble)

