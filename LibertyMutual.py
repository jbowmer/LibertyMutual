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

os.chdir('/Users/Jake/Kaggle/LibertyMutual/Data')


train = pd.read_csv('train.csv', index_col = 0)

Y = train['Hazard']
X = train.drop(['Hazard'])


#Create a training and validation set, 20% of the data will be reserved as the test set:

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

#We then split the test set again in order to have different data to train the base models and
#metamodels. Split the data equally.:

X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X_train, Y_train, test_size = 0.5) 