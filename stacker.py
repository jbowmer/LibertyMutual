# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:38:14 2015

@author: Jake
"""

#Stacker:

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

#Create a training and validation set, 20% of the data will be reserved as the test set:

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

#We then split the test set again in order to have different data to train the base models and
#metamodels. Split the data equally.:

X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X_train, Y_train, test_size = 0.5) 


#Random Forest Model:
from sklearn.ensemble import RandomForestRegressor
rf_mod = RandomForestRegressor(n_estimators = 500)
rf_mod.fit(X_train1, Y_train1)


#Extra Trees model:
from sklearn.ensemble import ExtraTreesRegressor
et_mod = ExtraTreesRegressor(n_estimators = 100)
et_mod.fit(X_train1, Y_train1)

#Gradient Boosting Model:
from sklearn.ensemble import GradientBoostingRegressor
gbm_mod = GradientBoostingRegressor(n_estimators = 500)
gbm_mod.fit(X_train1, Y_train1)


#Dataframe to train stacked model:

rf_mod_predicted = rf_mod.predict(X_train2)
et_mod_predicted = et_mod.predict(X_train2)
gbm_mod_predicted = gbm_mod.predict(X_train2)

base_predictions = pd.DataFrame({'rf_mod':rf_mod_predicted,
                                 'et_mod':et_mod_predicted,
                                 'gbm_mod':gbm_mod_predicted})

                                 
#Train stacked model:
from sklearn.svm import SVR

svm_mod = SVR(kernel = 'linear')
svm_bag = BaggingRegressor(svm_mod, n_estimators = 50)
svm_bag.fit(base_predictions, Y_train2)


#Examine performance on test set:
rf_mod_test_pred = rf_mod.predict(X_test)
et_mod_test_pred = et_mod.predict(X_test)
gbm_mod_test_pred = gbm_mod.predict(X_test)

test_predictions = pd.DataFrame({'rf_mod':rf_mod_test_pred,
                                 'et_mod':et_mod_test_pred,
                                 'gbm_mod':gbm_mod_test_pred})
                                 
                                 
#Use stacked model to predict:
svm_bag_pred = svm_bag.predict(test_predictions)

#Test with gini scoring script:
Gini(Y_test, rf_mod_test_pred) #30.2
Gini(Y_test, et_mod_test_pred) #30.1
Gini(Y_test, gbm_mod_test_pred) #36.46
Gini(Y_test, svm_bag_pred) #36.25
                                 
#Create submission:

rf_sub = rf_mod.predict(test)                                 
et_sub = et_mod.predict(test)
gbm_sub = gbm_mod.predict(test)                                 
                  
test_df = pd.DataFrame({'rf_mod':rf_sub,'et_mod':et_sub,'gbm_mod':gbm_sub})                  


svm_sub = svm_bag.predict(test_df)

#Submission:

submission = pd.DataFrame({'Id': test_id, 'Hazard': svm_sub})
submission = submission.set_index('Id')
submission.to_csv('stacker.csv')