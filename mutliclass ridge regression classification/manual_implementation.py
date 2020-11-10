# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:57:23 2020

@author: andr4
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('mnistFull.mat')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train'][:,0]
y_test = data['y_test'][:,0]

## YOUR CODE BELOW
# convert to one-hot encoding
k = 10 #number of classes
m_test = X_test.shape[1] # number of testing examples
m_train = X_train.shape[1] # number of training examples
Y_test = np.zeros((k,m_test))
Y_train = np.zeros((k,m_train))
for i in range(0,Y_test.shape[1]):
    Y_test[y_test[i],i] = 1
for i in range(0,Y_train.shape[1]):
    Y_train[y_train[i],i] = 1


# solve for W
X = np.transpose(X_train)
Y = np.transpose(Y_train)
X_T = np.transpose(X)
I = np.identity(X.shape[1])
l = 0.0001
inv = np.linalg.inv(X_T.dot(X)+np.multiply(l,I))
W = (inv.dot(X_T)).dot(Y)


# report training and test error 
Y_train = np.transpose(W).dot(X_train)
train_pred = np.argmax(Y_train,axis=0)
check = (train_pred==y_train)*1
correct_pred_train = check.sum()
incorrect_pred_train = check.shape[0]-correct_pred_train
accuracy_training = 100*(correct_pred_train/check.shape[0])
error_training = 100*(incorrect_pred_train/check.shape[0]) 


Y_test = np.transpose(W).dot(X_test)
test_pred = np.argmax(Y_test,axis=0)
check = (test_pred==y_test)*1
correct_pred_test = check.sum()
incorrect_pred_test = check.shape[0]-correct_pred_test
accuracy_test = 100*(correct_pred_test/check.shape[0])
error_test = 100*(incorrect_pred_test/check.shape[0])