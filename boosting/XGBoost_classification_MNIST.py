# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:07:16 2020

@author: Andrew
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

data = loadmat('mnistFull.mat')
X_train = np.transpose(data['X_train'])
X_test = np.transpose(data['X_test'])
y_train = data['y_train'][:,0]
y_train = y_train.reshape(X_train.shape[0],1)
y_test = data['y_test'][:,0]
y_test = y_test.reshape(X_test.shape[0],1)

D_train = xgb.DMatrix(X_train,label=y_train)
D_test = xgb.DMatrix(X_test,label=y_test)
parameters = {"max_depth":4,"objective":"multi:softprob","num_class":10}
steps = 100
model = xgb.train(parameters,D_train,steps)
predictions_train = np.argmax(model.predict(D_train),axis=1).reshape(y_train.shape[0],1)
predictions_test = np.argmax(model.predict(D_test),axis=1).reshape(y_test.shape[0],1)
check_train = np.equal(predictions_train,y_train)*1
check_test = np.equal(predictions_test,y_test)*1
correct_train = np.sum(check_train)
correct_test = np.sum(check_test)
accuracy_train = 100*(correct_train/y_train.shape[0])
accuracy_test = 100*(correct_test/y_test.shape[0])
error_train = 100-accuracy_train
error_test = 100-accuracy_test