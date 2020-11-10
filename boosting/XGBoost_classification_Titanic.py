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

training = pd.read_csv("Titanic Training.csv")
test = pd.read_csv("Titanic Test.csv")

X_train = training.drop(columns="Survived").to_numpy()
y_train = training["Survived"].to_numpy().reshape(y_train.shape[0],1)
m = y_train.shape[0]
D = np.ones((m,1))/m
X_test = test.drop(columns="PassengerId").to_numpy()
test_ID = test["PassengerId"]

D_train = xgb.DMatrix(X_train,label=y_train)
D_test = xgb.DMatrix(X_test,label=None)
parameters = {"max_depth":10,"num_class":2}
steps = 100
model = xgb.train(parameters,D_train,steps)
predictions_train = model.predict(D_train).reshape(X_train.shape[0],1)
predictions_test = model.predict(D_test).reshape(X_test.shape[0],1)
check_train = np.equal(predictions_train,y_train)*1
correct_train = np.sum(check_train)
accuracy_train = 100*(correct_train/y_train.shape[0])
error_train = 100-accuracy_train
results_df = pd.concat([test_ID,pd.DataFrame(predictions_test)],axis=1)
results_df = results_df.rename(columns = {0:"Survived"})
