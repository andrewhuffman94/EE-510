# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:43 2020

@author: Andrew
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.linear_model import RidgeClassifier

data = loadmat('mnistFull.mat')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train'][:,0]
y_test = data['y_test'][:,0]

## YOUR CODE BELOW
x_train = np.transpose(X_train) # transpose training examples to match classifier expected format of #sample x #features
x_test = np.transpose(X_test) # transpose test examples to match classifier expected format of #sample x #features
classifier = RidgeClassifier(alpha=0.0001,solver="lsqr") # Define ridge classifier to use regularization paramemter = 10^-4 and regularized least squares routine as solver
classifier.fit(x_train,y_train) # Fit the ridge classification model to the training data 
accuracy_training = classifier.score(x_train,y_train) # Calculate classification accuracy on training data as a decimal
error_training = 100*(1-accuracy_training) # Calculate classification error on training data as a percentage
accuracy_test = classifier.score(x_test,y_test) #Calculate classification accraucy on test data as a decimal
error_test = 100*(1-accuracy_test) # Calculate classification error on test data as a percentage