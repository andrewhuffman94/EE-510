# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:57:23 2020

@author: andr4
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

data = loadmat('mnistFull.mat')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train'][:,0]
y_test = data['y_test'][:,0]
X_all = np.concatenate([X_train,X_test],axis=1)
## YOUR CODE BELOW
# convert to one-hot encoding
k = 10 #number of classes
m_test = X_test.shape[1] # number of testing examples
m_train = X_train.shape[1] # number of training examples
Y_test = np.zeros((k,m_test))
Y_train = np.zeros((k,48000))
Y_val = np.zeros((k,12000))
    
X_train, X_val, y_train, y_val = train_test_split(X_train.T, y_train, test_size=0.2, random_state=42)
for i in range(0,Y_test.shape[1]):
    Y_test[y_test[i],i] = 1
for i in range(0,Y_train.shape[1]):
    Y_train[y_train[i],i] = 1
for i in range(0,Y_val.shape[1]):
    Y_val[y_val[i],i] = 1    
Train_Acc = []
Val_Acc = []
Test_Acc = []
Train_Error = []
Val_Error = []
Test_Error = []

for n in range(1,121):
    # Random Fourier transformation
    mu = 0
    var = 0.1
    sigma = math.sqrt(var)
    d = X_all.shape[0]
    m = X_all.shape[1]
    p = 50*n
    print(p)
    G = np.random.normal(mu,sigma,(p,d))
    b = np.random.uniform(0,2*math.pi,(p,1))
    X_ft_train = np.cos((G.dot(X_train.T)+b))
    X_ft_val = np.cos((G.dot(X_val.T)+b))
    X_ft_test = np.cos((G.dot(X_test)+b))
    
    # solve for W
    X = X_ft_train.T
    X_T = X.T
    I = np.identity(X_ft_train.shape[0])
    l = 0.0001
    inv = np.linalg.inv(X_T.dot(X)+np.multiply(l,I))
    W = (inv.dot(X_T)).dot(Y_train.T)
    
    # Make validation predictions and compute error
    pred_val = np.transpose(W).dot(X_ft_val)
    val_pred = np.argmax(pred_val,axis=0)
    check = (val_pred==y_val)*1
    correct_pred_val = check.sum()
    incorrect_pred_val = check.shape[0]-correct_pred_val
    accuracy_val = 100*(correct_pred_val/check.shape[0])
    error_val = 100*(incorrect_pred_val/check.shape[0]) 
    
    # report training and test error 
    pred_train = np.transpose(W).dot(X_ft_train)
    train_pred = np.argmax(pred_train,axis=0)
    check = (train_pred==y_train)*1
    correct_pred_train = check.sum()
    incorrect_pred_train = check.shape[0]-correct_pred_train
    accuracy_training = 100*(correct_pred_train/check.shape[0])
    error_training = 100*(incorrect_pred_train/check.shape[0]) 
    
    
    pred_test = np.transpose(W).dot(X_ft_test)
    test_pred = np.argmax(pred_test,axis=0)
    check = (test_pred==y_test)*1
    correct_pred_test = check.sum()
    incorrect_pred_test = check.shape[0]-correct_pred_test
    accuracy_test = 100*(correct_pred_test/check.shape[0])
    error_test = 100*(incorrect_pred_test/check.shape[0])
    
    Train_Acc.append(accuracy_training)
    Val_Acc.append(accuracy_val)
    Test_Acc.append(accuracy_test)
    Train_Error.append(error_training)
    Val_Error.append(error_val)
    Test_Error.append(error_test)