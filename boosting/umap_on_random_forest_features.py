# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:38:34 2020

@author: Andrew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import umap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomTreesEmbedding
from statistics import mode
import re

# import our dataset
data = pd.read_csv("train.csv")
training_data = data
y = data["Survived"]

# Impute missing values and drop featrues and convert categorical features to integer classes for training and test data
training_data["Embarked"] = training_data["Embarked"].fillna(mode(training_data["Embarked"]))
training_data["Age"] = training_data["Age"].fillna(mode(training_data["Age"]))
training_data = training_data.drop(columns=["PassengerId","Survived","Name","Ticket"])
training_data['Cabin'] = training_data['Cabin'].fillna('U')
training_data['Cabin'] = training_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

training_data["Sex"][training_data["Sex"] == "male"] = 0
training_data["Sex"][training_data["Sex"] == "female"] = 1

training_data["Embarked"][training_data["Embarked"] == "S"] = 0
training_data["Embarked"][training_data["Embarked"] == "C"] = 1
training_data["Embarked"][training_data["Embarked"] == "Q"] = 2

C = sorted(training_data["Cabin"].unique().tolist())
integer_label = 0
for c in range(len(C)):
    integer_label = integer_label+1
    training_data["Cabin"][training_data["Cabin"] == C[c]] = integer_label

training_data = training_data.astype(float)   
X = training_data.to_numpy()


reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)
plt.figure()
plt.scatter(embedding[:,0],embedding[:,1], c=y, cmap="Spectral", s=8)
plt.gca().set_aspect("equal", "datalim")
cb = plt.colorbar()
loc = np.arange(0,max(y)+0.5,1)
cb.set_ticks(loc)
plt.title("UMAP projection of Titanic dataset")

# Use Extra Trees Classifier Embedding
model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10)
model.fit(X,y)
leaves = model.apply(X)
reducer = umap.UMAP(metric='hamming',random_state=42)
embedding = reducer.fit_transform(leaves)
# plotting the embedding
plt.figure()
plt.scatter(embedding[:,0],embedding[:,1], c=y, cmap="Spectral", s=8)
plt.gca().set_aspect("equal", "datalim")
cb = plt.colorbar()
loc = np.arange(0,max(y)+0.5,1)
cb.set_ticks(loc)
plt.title("UMAP Projection of Titanic Dataset\n Using Extra Trees Classifier Embedding")


# Use DecisionTreeClassifier Embedding
model = DecisionTreeClassifier(max_leaf_nodes=2)
tree = model.fit(X,y)
clusters = tree.apply(X)
# plotting the embedding
plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=clusters, cmap='Spectral', s=8)
plt.gca().set_aspect("equal", "datalim")
plt.title('UMAP Projection of Titanic Dataset\n Using Clustering with Decision Trees')


# Use RandomTreesEmbedding
model = RandomTreesEmbedding(n_estimators=100,max_leaf_nodes=2)
model.fit(X,y)
leaves = model.apply(X)
reducer = umap.UMAP(metric='hamming',random_state=42)
embedding = reducer.fit_transform(leaves)
# plotting the embedding
plt.figure()
plt.scatter(embedding[:,0],embedding[:,1], c=y, cmap="Spectral", s=8)
plt.gca().set_aspect("equal", "datalim")
cb = plt.colorbar()
loc = np.arange(0,max(y)+0.5,1)
cb.set_ticks(loc)
plt.title("UMAP Projection of Titanic Dataset\n Using Random Trees Embedding")
