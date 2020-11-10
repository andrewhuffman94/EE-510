# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10
from statistics import mode
from sklearn.metrics import confusion_matrix
import re
import math



# Define ridge regression and logistic regression functions

def ridge_reg_validate(X,y,l,split_index,classes):
    global ridge_confusion_matrix
    X_train = X[:split_index,:]
    y_train = y[:split_index]
    X_val = X[(split_index+1):,:]
    y_val = y[(split_index+1):]
    
    # convert to one-hot encoding
    k = classes #number of classes
    m_val = X_val.shape[0] # number of testing examples
    m_train = X_train.shape[0] # number of training examples
    Y_val = np.zeros((k,m_val))
    Y_train = np.zeros((k,m_train))
    for i in range(0,Y_val.shape[1]):
        Y_val[y_val[i],i] = 1
    for i in range(0,Y_train.shape[1]):
        Y_train[y_train[i],i] = 1
    
    # solve for W
    X_T = X_train.T
    I = np.identity(X_train.shape[1])
    inv = np.linalg.inv(X_T.dot(X_train)+np.multiply(l,I))
    W = (inv.dot(X_T)).dot(Y_train.T)
    
    
    # report training and test error 
    y_pred_train = X_train.dot(W)
    train_pred = np.argmax(y_pred_train,axis=1)
    check = (train_pred==y_train)*1
    correct_pred_train = check.sum()
    incorrect_pred_train = check.shape[0]-correct_pred_train
    accuracy_training = 100*(correct_pred_train/check.shape[0])
    error_training = 100*(incorrect_pred_train/check.shape[0]) 
    
    y_pred_val = X_val.dot(W)
    val_pred_ridge_reg = np.argmax(y_pred_val,axis=1)
    check = (val_pred_ridge_reg==y_val)*1
    correct_pred_val = check.sum()
    incorrect_pred_val = check.shape[0]-correct_pred_val
    accuracy_val = 100*(correct_pred_val/check.shape[0])
    error_val = 100*(incorrect_pred_val/check.shape[0])
    ridge_confusion_matrix = confusion_matrix(y_val,val_pred_ridge_reg)
    
    results_training_ridge_reg = [accuracy_training,error_training]
    results_validation_ridge_reg = [accuracy_val,error_val]
    
    return results_validation_ridge_reg

def ridge_reg_train(X,y,l,classes):
#    offset = np.ones((X.shape[0],1))
    X_train = X
    # convert to one-hot encoding
    k = classes #number of classes
    m_train = X_train.shape[0] # number of training examples
    Y_train = np.zeros((k,m_train))
    for i in range(0,Y_train.shape[1]):
        Y_train[y[i],i] = 1
    
    # solve for W
    X_T = X_train.T
    I = np.identity(X_train.shape[1])
    inv = np.linalg.inv(X_T.dot(X_train)+np.multiply(l,I))
    W = (inv.dot(X_T)).dot(Y_train.T)

    return W



def log_reg_validate(X,y,split_index,classes,mu,iterations):
    global K 
    global d
    global log_confusion_matrix
    K = classes
    X_train = X[:split_index,:]
    y_train = y[:split_index]
    X_val = X[(split_index+1):,:]
    y_val = y[(split_index+1):]
    d = X_val.shape[1]
    W = np.random.randn(d, K)
    Ntrain = len(y_train)
    global ACCURACY 
    global ERROR
    
    # Define softmax calculation
    def softmax(W,x):
        num = np.exp(W.T@x)
        denom = sum(num)
        smax = num/denom
        return smax
    
    # Define gradient calculation
    def gradient(W,x,y):
        Y = np.zeros((1,K))
        Y[0,y] = 1      
        grad = np.kron((softmax(W,x).reshape(1,K)-Y),x)
        return grad
    
    # SGD iterations
    w = np.random.randn(d,K)
    ERROR = np.zeros((iterations,1))
    ACCURACY = np.zeros((iterations,1))
    ## YOUR SGD LOOP GOES HERE
    for i in range(iterations):
        rows = np.random.permutation(Ntrain)
        for r in range(Ntrain):
            row = rows[r]
            x = X_train[row,:].reshape(d,1)
            grad = gradient(w.reshape((d,K)),x,y_train[row])
            w = w-(mu*grad)
            ## reshape to get final weight matrix
            W = w.reshape((d, K), order='F')
    
        # report training and test error
        P = X_val.dot(W)
        val_predictions_log_reg = np.argmax(P,axis=1)
        check = val_predictions_log_reg-y_val.T
        incorrect_predictions = np.count_nonzero(check,axis=0)
        error = 100*(incorrect_predictions/y_val.shape[0])
        accuracy = 100-error
        ERROR[i] = error
        ACCURACY[i] = accuracy
    results_validation_log_reg = np.concatenate([ACCURACY,ERROR],axis=1)
    log_confusion_matrix = confusion_matrix(y_val,val_predictions_log_reg)
    # Generate accuracy and error plots
    n = np.linspace(1,iterations,iterations)
    ticks = np.arange(min(n)-1,max(n)+1,5)
    plt.figure()
    plt.plot(n,ACCURACY,"b*")
    plt.xticks(ticks)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy (%)")
    
    plt.figure()
    plt.plot(n,ERROR,"b*")
    plt.xticks(ticks)
    plt.xlabel("Iterations")
    plt.ylabel("Error (%)")
    
    return val_predictions_log_reg,results_validation_log_reg,W

def log_reg_train(X,y,classes,mu,iterations):
    K = classes
#    offset = np.ones((X.shape[0],1))
    X_train = X
    d = X_train.shape[1]
    W = np.random.randn(d, K)
    Ntrain = len(y)
    
    # Define softmax calculation
    def softmax(W,x):
        num = np.exp(W.T@x)
        denom = sum(num)
        smax = num/denom
        return smax
    
    # Define gradient calculation
    def gradient(W,x,y):
        Y = np.zeros((1,K))
        Y[0,y] = 1      
        grad = np.kron((softmax(W,x).reshape(1,K)-Y),x)
        return grad
    
    # SGD iterations
    w = np.random.randn(d,K)
    
    ## YOUR SGD LOOP GOES HERE
    for i in range(iterations):
        rows = np.random.permutation(Ntrain)
        for r in range(Ntrain):
            row = rows[r]
            x = X_train[row,:].reshape(d,1)
            grad = gradient(w.reshape((d,K)),x,y[row])
            w = w-(mu*grad)
            ## reshape to get final weight matrix
            W = w.reshape((d, K), order='F')
    
    return W

def predict(X,W):
#    offset = np.ones((X.shape[0],1))
    X_predict = X
    y = X_predict.dot(W)
    predictions = np.argmax(y,axis=1)
    return predictions


# Load training and test data
training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
missing_features = training_data.isnull().any() #check which features have missing values
missing_feature_count = training_data.isnull().sum()
training = training_data.dropna(axis=0,how="any") # training set with missing values removed

# Data exploration and wrangling
summary_trainingdata = training_data.describe()
trainingdata_sex = training_data.groupby("Sex").Survived.value_counts()
trainingdata_Pclass = training_data.groupby("Pclass").Survived.value_counts()
trainingdata_Age = training_data.groupby("Age").Survived.value_counts()
trainingdata_SibSp = training_data.groupby("SibSp").Survived.value_counts()
trainingdata_Parch = training_data.groupby("Parch").Survived.value_counts()
trainingdata_Fare = training_data.groupby("Fare").Survived.value_counts()
trainingdata_Embarked = training_data.groupby("Embarked").Survived.value_counts()
age_group = training_data.groupby("Pclass")["Age"]
test_age_group = test_data.groupby("Pclass")["Age"]
training_data.loc[training_data.Age.isnull(), 'Age'] = training_data.groupby("Pclass").Age.transform('median')
test_data.loc[test_data.Age.isnull(), "Age"] = test_data.groupby("Pclass").Age.transform('median')
Age_bins = pd.cut(training_data["Age"],7,labels=["1","2","3","4","5","6","7"],retbins=False)
test_Age_bins = pd.cut(test_data["Age"],7,labels=["1","2","3","4","5","6","7"],retbins=False)
Age_binned = pd.Series.to_frame(Age_bins).rename(columns={"Age":"AgeBin"})
test_Age_binned = pd.Series.to_frame(test_Age_bins).rename(columns={"Age":"AgeBin"})


#plt.figure()
#sns.countplot(x = 'Survived', hue = 'Sex', data = training_data)
#plt.figure()
#sns.countplot(x = 'Survived', hue = 'Pclass', data = training_data)
#plt.figure()
#sns.countplot(x = 'Survived', hue = 'Age', data = training_data)
#plt.figure()
#sns.countplot(x = 'Survived', hue = 'SibSp', data = training_data)
#plt.figure()
#sns.countplot(x = 'Survived', hue = 'Parch', data = training_data)
#plt.figure()
#sns.countplot(x = 'Survived', hue = 'Fare', data = training_data)
#plt.figure()
#sns.countplot(x = 'Survived', hue = 'Embarked', data = training_data)
#plt.figure()
#sns.countplot(x="Survived",hue="AgeBin",data=training_edited)




#training_2 = training_edited.drop(columns="Cabin") # remove cabin feature because it dominates the number of rows with missing values
#training_2 = training_2.dropna(axis=0,how="any") # remove rows with missing values 
##plt.figure()
#correlation = pd.DataFrame.corr(training_data)
#sns.heatmap(correlation,annot=True)
#
#summary_training = training.describe()
#sex_training = training.groupby("Sex").Survived.value_counts()
#summary_training2 = training_2.describe()
#sex_training2 = training_2.groupby("Sex").Survived.value_counts()
#male = training[training["Sex"]=="male"]
#summmary_male = male.describe()
#female = training[training["Sex"]=="female"]
#summary_female = female.describe()
#male2 = training_2[training_2["Sex"]=="male"]
#summary_male2 = male2.describe()
#female2 = training_2[training_2["Sex"]=="female"]
#summary_female2 = female2.describe()

# Impute missing values and drop featrues and convert categorical features to integer classes for training and test data
training_data["Embarked"] = training_data["Embarked"].fillna(mode(training_data["Embarked"]))
test_data["Embarked"] = test_data["Embarked"].fillna(mode(test_data["Embarked"]))

training_data['Cabin'] = training_data['Cabin'].fillna('U')
test_data["Cabin"] = test_data["Cabin"].fillna("U")
#
training_edited = pd.concat([training_data,Age_binned],axis=1).drop(columns=["Fare","Ticket","Name","Age"])
training_edited['Cabin'] = training_edited['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
test_edited = pd.concat([test_data,test_Age_binned],axis=1).drop(columns=["Fare","Ticket","Name","Age"])
test_edited['Cabin'] = test_edited['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

training_edited["Sex"][training_edited["Sex"] == "male"] = 0
training_edited["Sex"][training_edited["Sex"] == "female"] = 1
test_edited["Sex"][test_edited["Sex"] == "male"] = 0
test_edited["Sex"][test_edited["Sex"] == "female"] = 1

training_edited["Embarked"][training_edited["Embarked"] == "S"] = 0
training_edited["Embarked"][training_edited["Embarked"] == "C"] = 1
training_edited["Embarked"][training_edited["Embarked"] == "Q"] = 2
test_edited["Embarked"][test_edited["Embarked"] == "S"] = 0
test_edited["Embarked"][test_edited["Embarked"] == "C"] = 1
test_edited["Embarked"][test_edited["Embarked"] == "Q"] = 2

C = sorted(training_edited["Cabin"].unique().tolist())
integer_label = 0
for c in range(len(C)):
    integer_label = integer_label+1
    training_edited["Cabin"][training_edited["Cabin"] == C[c]] = integer_label
test_C = sorted(test_edited["Cabin"].unique().tolist())
integer_label = 0
for c in range(len(test_C)):
    integer_label = integer_label+1
    test_edited["Cabin"][test_edited["Cabin"] == test_C[c]] = integer_label
training_edited = training_edited.astype(int)   
  
#plt.figure()
#correlation = pd.DataFrame.corr(training_edited)
#sns.heatmap(correlation,annot=True,fmt=".2f")
#
training_labels = training_edited["Survived"]
training_examples = training_edited.drop(columns=["PassengerId","Survived","Cabin","AgeBin"])
test_examples = test_edited.drop(columns=["PassengerId","Cabin","AgeBin"])
test_ID = test_edited["PassengerId"]
train_df = pd.concat([training_labels,training_examples],axis=1)
test_df = pd.concat([test_ID,test_examples],axis=1)
train_df.to_csv("Titanic Training.csv",index=False)
test_df.to_csv("Titanic Test.csv",index=False)
## Perform UMAP embedding and generate matplotlib static plot
##reducer = umap.UMAP(random_state=42)
##embedding = reducer.fit_transform(x)
##plt.figure()
##plt.scatter(embedding[:,0],embedding[:,1], c=y)
##plt.gca().set_aspect("equal", "datalim")
##plt.title("UMAP projection of the edited Titanic dataset")
#
## Visulaize effects of sex, Pclass, and Cabin
##plt.figure()
##sns.catplot("Sex", "Survived", hue="Pclass", kind="bar", data=training_edited);
##plt.figure()
##sns.catplot("Cabin", "Survived", hue="Sex", kind="bar", data=training_edited);
#
#
#X = training_examples.to_numpy().astype(int)
#y = training_labels.to_numpy().astype(int)
#X_test = test_examples.to_numpy().astype(int)
#split_index = math.ceil(X.shape[0]*0.8) 
#mu = 3e-3
#iterations = 100
#classes = 2
#l = 1e-3
#ridge_train = ridge_reg_train(X,y,l,classes)
#log_train = log_reg_train(X,y,classes,mu,iterations)
#
#ridge_predict = pd.DataFrame(predict(X_test,ridge_train))
#log_predict = pd.DataFrame(predict(X_test,log_train))
#
#ridge_df = pd.concat([test_ID,ridge_predict],axis=1)
#ridge_df = ridge_df.rename(columns = {0:"Survived"})
#log_df = pd.concat([test_ID,log_predict],axis=1)
#log_df = log_df.rename(columns = {0:"Survived"})
#
#ridge_accuracy = []
#ridge_error = []
#for a in range(1):
#    l = 1*(10**-a)
#    ridge = ridge_reg_validate(X,y,l,split_index,classes)
#    ridge_accuracy.append(ridge[0])
#    ridge_error.append(ridge[1])
#n = np.linspace(1,a+1,a+1)
#plt.figure()
#plt.plot(n,ridge_accuracy,"b*")
#plt.xticks(n)
#plt.xlabel("Iterations")
#plt.ylabel("Accuracy (%)")
#
#plt.figure()
#plt.plot(n,ridge_error,"b*")
#plt.xticks(n)
#plt.xlabel("Iterations")
#plt.ylabel("Error (%)")
#
#log = log_reg_validate(X,y,split_index,classes,mu,iterations) 
#
#
