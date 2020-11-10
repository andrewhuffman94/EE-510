# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:55:19 2020

@author: andr4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import xgboost as xgb
from sklearn import decomposition
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import random
import lightgbm as lgb
from sklearn.metrics import plot_confusion_matrix
import plotly.express as px
#mpl.rcParams['font.family'] = 'Arial'


# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = train.drop(["id","target"],axis=1)
targets = train["target"]

## Plot target distribution
#ax = sns.countplot(x = targets ,palette="Blues_d")
#sns.set(font_scale=1.5)
#ax.set_xlabel(' ')
#ax.set_ylabel(' ')
#fig = plt.gcf()
#fig.set_size_inches(10,5)
#ax.set_ylim(top=700000)
#for p in ax.patches:
#    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))
#
##plt.title('Distribution of 595212 Targets')
#plt.xlabel('Target Value')
#plt.ylabel('Frequency (%)')
#plt.savefig("Target_Distribution.eps")
#plt.savefig("Target_Distribution.png")
#plt.show()


## Select Integer-valued Features #
#train_int = train.select_dtypes(include=['int64'])
#corr_int = train_int.corr()
#sns.set(style="white")
#colormap = sns.diverging_palette(220, 10, as_cmap=True)
#plt.figure(figsize=(16,12))
#plt.title('Pearson correlation of integer-valued features', y=1.05, size=15)
#sns.heatmap(corr_int,linewidths=0.1,vmax=0.3, center=0, square=True, 
#            cmap=colormap, linecolor='white', annot=False)
#
##sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
##            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
## Select Continuous Features #
#train_float = train.select_dtypes(include=['float64'])
#train_float.insert(loc=0, column="Target",value=targets)
#corr_float = train_float.corr()
#sns.set(style="white")
#colormap = sns.diverging_palette(220, 10, as_cmap=True)
#plt.figure(figsize=(16,12))
#plt.title('Pearson correlation of continuous features', y=1.05, size=15)
#sns.heatmap(corr_float,linewidths=0.1,vmax=1.0, center=0, square=True, 
#            cmap=colormap, linecolor='white', annot=True)


# Plot correlation matrix
#f = train.drop(["id"],axis=1)
#corr = f.corr()
#sns.set(style="white")
## Generate a custom diverging colormap
#colormap = sns.diverging_palette(220, 10, as_cmap=True)
#plt.figure(figsize=(16,12))
#plt.title('Pearson correlation', y=1.05, size=15)
#sns.heatmap(corr,linewidths=0.1,vmax=0.4, center=0, square=True, cmap=colormap, linecolor='white', annot=False)
##plt.show()
#plt.savefig("Correlationn_Matrix.eps")
#plt.savefig("Correlationn_Matrix.png")


# Plot ones and zeros by feature




##################### Gini Score Calculation #################################
#def eval_gini(y_true, y_prob):
#    y_true = np.asarray(y_true)
#    y_true = y_true[np.argsort(y_prob)]
#    ntrue = 0
#    gini = 0
#    delta = 0
#    n = len(y_true)
#    for i in range(n-1, -1, -1):
#        y_i = y_true[i]
#        ntrue += y_i
#        gini += y_i * delta
#        delta += 1 - y_i
#    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
#    return gini
#
#def ginic(actual, pred):
#    actual = np.asarray(actual) #In case, someone passes Series or list
#    n = len(actual)
#    a_s = actual[np.argsort(pred)]
#    a_c = a_s.cumsum()
#    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
#    return giniSum / n
 
#def gini_normalizedc(a, p):
#    if p.ndim == 2:#Required for sklearn wrapper
#        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
#    return ginic(a, p) / ginic(a, a)
##############################################################################

############## Calculate Gini Score from Labels and XGBoost Predictions ############
#def gini_xgb(preds, dtrain):
#    labels = dtrain.get_label()
#    gini_score = eval_gini(labels, preds)
#    return [('gini', gini_score)]   
#####################################################################################





# Undersample the training data    
X = features.to_numpy()
y = targets.to_numpy()
##
#rus = RandomUnderSampler()
#X_rus, y_rus = rus.fit_sample(X, y)


########## UMAP ############
#n = len(X)
#rows = []
#for a in range(1,1000):
#    i = random.randint(0, n)
#    rows.append(i)
#f = X[rows,:]
#t = y[rows]

#n_rus = len(X_rus)
#rows_rus = []
#for a in range(1,1000):
#    i = random.randint(0, n_rus)
#    rows_rus.append(i)
#f_rus = X_rus[rows_rus,:]
#t_rus = y_rus[rows_rus]

#reducer = umap.UMAP(random_state=42)
#embedding = reducer.fit_transform(f)
#plt.figure()
#plt.scatter(embedding[:,0],embedding[:,1], c=t, cmap="Spectral", s=5)
#plt.gca().set_aspect("equal", "datalim")
#plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
#plt.title("UMAP projection of random subset of original dataset")
#plt.savefig("UMAP.eps")
#plt.savefig("UMAP.png")
###
#embedding = reducer.fit_transform(f_rus)
#plt.figure()
#plt.scatter(embedding[:,0],embedding[:,1], c=t_rus, cmap="Spectral", s=5)
#plt.gca().set_aspect("equal", "datalim")
#plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
#plt.title("UMAP projection of random subset of resampled dataset")
##plt.savefig("UMAP_rus.eps")
##plt.savefig("UMAP_rus.pdf")
#plt.savefig("UMAP_rus.png",dpi=600)
#
#
######### TSVD Visualizations of Data #######
n_comp = 2
svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd.fit(X)
print(svd.explained_variance_ratio_.sum())
X_svd = svd.transform(X)
print("Transformation Finished")
plt.figure()
plt.scatter(X_svd[:,0],X_svd[:,1], c=y, cmap="Spectral", s=2)
plt.gca().set_aspect("equal", "datalim")
plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
plt.title("Truncated SVD projection of Original dataset")
#plt.savefig("SVD_projection.eps")
#plt.savefig("SVD_projection.pdf")
plt.savefig("SVD_projection.png",dpi=1000)

#n_comp = 2
#svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack')
#svd.fit(X_rus)
#print(svd.explained_variance_ratio_.sum())
#X_svd_rus = svd.transform(X_rus)
#print("Transformation Finished")
#plt.figure()
#plt.scatter(X_svd_rus[:,0],X_svd_rus[:,1], c=y_rus, cmap="Spectral", s=2)
#plt.gca().set_aspect("equal", "datalim")
#plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
#plt.title("Truncated SVD projection of resampled dataset")
#plt.savefig("RUS_SVD_projection.eps")
#plt.savefig("RUS_SVD_projection.png")

#ros = RandomOverSampler()
#X_ros, y_ros = ros.fit_sample(X, y)

#cc = ClusterCentroids(sampling_strategy=0.5, n_jobs=-1)
#X_cc, y_cc = cc.fit_sample(X, y)
#print("Undersampling Finished")
#smote = SMOTE(sampling_strategy="minority")
#X_sm, y_sm = smote.fit_sample(X, y)

# Preprocess data 
#test_id = test["id"]
#features_test = test.drop('id', axis=1).to_numpy()
###
#scaler = StandardScaler()
#scaler.fit(X_rus)
#X_rus = scaler.transform(X_rus)
#X_test_rus = scaler.transform(features_test)
#X_rus_train,X_rus_val,y_rus_train,y_rus_val = train_test_split(X_rus, y_rus, test_size=0.2, random_state=42)
##
###scaler.fit(X_cc)
###X_cc = scaler.transform(X_cc)
###X_test_cc = scaler.transform(features_test)
###print("Data Preprocessing Finished")
#
################## Predictions #######################3
#sub_1 = pd.DataFrame()
#sub_1['id'] = test_id
#sub_1['target'] = np.zeros_like(test_id)
#
#sub_2 = pd.DataFrame()
#sub_2['id'] = test_id
#sub_2['target'] = np.zeros_like(test_id)
#
#scaler.fit(X)
#X_all = scaler.transform(X)
#X_all_test = scaler.transform(features_test)
#X_train,X_val,y_train,y_val = train_test_split(X_all, y, test_size=0.2, random_state=42)
#classifier = lgb.LGBMClassifier(boosting_type="gbdt",learning_rate=0.1,num_leaves=20,random_state=42,n_jobs=-1, silent=False)
#
##
#classifier.fit(X_rus_train,y_rus_train,verbose=True)
#plot_confusion_matrix(classifier,X_rus_val,y_rus_val,cmap="Blues")
##plt.show()
#plt.savefig("confusion_matrix_resampling.eps")
#plt.savefig("confusion_matrix_resampling.png")
##
#classifier.fit(X_train,y_train,verbose=True)
##imp = classifier.feature_importances_
#plot_confusion_matrix(classifier,X_val,y_val,cmap="Blues")
##plt.show()
#plt.savefig("confusion_matrix_no_resampling.eps")
#plt.savefig("confusion_matrix_no_resampling.png")

#pred1 = classifier.predict_proba(X_test_rus)
#imp_rus = classifier.feature_importances_
#sub_1["target"] = pred1[:,1]
#classifier.fit(X_rus,y_rus)
#pred2 = classifier.predict_proba(X_test_rus)
#sub_2["target"] = pred2[:,1]
    
#imp = pd.Series(imp)
#imp_rus = pd.Series(imp_rus)
#names = pd.Series(names)
#imp_df = pd.concat([names,imp],axis=1)
#imp_rus_df = pd.concat([names,imp_rus],axis=1)
#imp_df.columns = ["Feature","Importance"]
#imp_rus_df.columns = ["Feature","Importance"]
#imp_df = imp_df.sort_values(by=['Importance'],ascending=False)
#imp_rus_df = imp_rus_df.sort_values(by=["Importance"],ascending=False)
# Feature importance plots #######    
#fig = px.bar(imp_df, x='Feature', y='Importance')
#fig.write_image("Feature_importance.png")
#fig.write_image("Feature_importance.eps")
#fig.show()
#
#fig = px.bar(imp_rus_df, x='Feature', y='Importance')
#fig.write_image("Feature_importance_rus.png")
#fig.write_image("Feature_importance_rus.eps")
#fig.show()
##### Plot Confusion Matrix ##############33

#clf = SVC(random_state=0)
#>>> clf.fit(X_train, y_train)
#SVC(random_state=0)
#>>> plot_confusion_matrix(clf, X_test, y_test)  # doctest: +SKIP
#>>> plt.show()  # doctest: +SKIP
#
## No under sampling with gbdt
#gini_gbdt = []
#pred_gbdt = []
#eta_gbdt = []
#leaves_gbdt = []
#for i in range(0,6):
#    print("***** GBDT *****")
#    eta = (10**i)*0.00001
#    for j in range(0,21):
#        eta_gbdt.append(eta)
#        leaves = 10+(10*j)
#        leaves_gbdt.append(leaves)
#        classifier = lgb.LGBMClassifier(boosting_type="gbdt",learning_rate=eta,num_leaves=leaves,random_state=42,n_jobs=-1, silent=False)
#        classifier.fit(X_train,y_train,verbose=True)
##        y_pred_val = classifier.predict(X_val)
#        pred_val = classifier.predict_proba(X_val)
#        gini = gini_normalizedc(y_val, pred_val)
##        gini = eval_gini(y_val,y_pred_val)
#        gini_gbdt.append(gini)
#        pred = classifier.predict_proba(X_all_test)
#        pred_gbdt.append(pred)
#        print("eta: ",eta,"  leaves: ",leaves )
#        print("*** gini = ",gini,"***")
#
## No under sampling with dart
#gini_dart = []
#pred_dart = []
#eta_dart = []
#leaves_dart = []
#for i in range(0,6):
#    print("***** DART *****")
#    eta = (10**i)*0.00001
#    for j in range(0,21):
#        eta_dart.append(eta)
#        leaves = 10+(10*j)
#        leaves_dart.append(leaves)
#        classifier = lgb.LGBMClassifier(boosting_type="dart",learning_rate=eta,num_leaves=leaves,random_state=42,n_jobs=-1, silent=False)
#        classifier.fit(X_train,y_train,verbose=True)
##        y_pred_val = classifier.predict(X_val)
##        gini = eval_gini(y_val,y_pred_val)
#        pred_val = classifier.predict_proba(X_val)
#        gini = gini_normalizedc(y_val, pred_val)
#        gini_dart.append(gini)
#        pred = classifier.predict_proba(X_all_test)
#        pred_dart.append(pred)
#        print("eta: ",eta,"  leaves: ",leaves )
#        print("*** gini = ",gini,"***")
#
#
## Random Under Samplilng with gbdt
#gini_rus_gbdt = []
#pred_rus_gbdt = []
#eta_rus_gbdt = []
#leaves_rus_gbdt = []
#for i in range(0,6):
#    print("***** GBDT *****")
#    eta = (10**i)*0.00001
#    for j in range(0,21):
#        eta_rus_gbdt.append(eta)
#        leaves = 10+(10*j)
#        leaves_rus_gbdt.append(leaves)
#        classifier = lgb.LGBMClassifier(boosting_type="gbdt",learning_rate=eta,num_leaves=leaves,random_state=42,n_jobs=-1, silent=False)
#        classifier.fit(X_rus_train,y_rus_train,verbose=True)
##        y_pred_val = classifier.predict(X_rus_val)
##        gini = eval_gini(y_rus_val,y_pred_val)
#        pred_val = classifier.predict_proba(X_rus_val)
#        gini = gini_normalizedc(y_rus_val, pred_val)
#        gini_rus_gbdt.append(gini)
#        pred_rus = classifier.predict_proba(X_test_rus)
#        pred_rus_gbdt.append(pred_rus)
#        print("eta: ",eta,"  leaves: ",leaves )
#        print("*** gini = ",gini,"***")
#
## Random Under Sampling with dart
#gini_rus_dart = []
#pred_rus_dart = []
#eta_rus_dart = []
#leaves_rus_dart = []
#for i in range(0,6):
#    print("****** DART *****")
#    eta = (10**i)*0.00001
#    for j in range(0,21):
#        eta_rus_gbdt.append(eta)
#        leaves = 10+(10*j)
#        leaves_rus_dart.append(leaves)
#        classifier = lgb.LGBMClassifier(boosting_type="dart",learning_rate=eta,num_leaves=leaves,random_state=42, n_jobs=-1, silent=False)
#        classifier.fit(X_rus_train,y_rus_train,verbose=True)
##        y_pred_val = classifier.predict(X_rus_val)
##        gini = eval_gini(y_rus_val,y_pred_val)
#        pred_val = classifier.predict_proba(X_rus_val)
#        gini = gini_normalizedc(y_rus_val, pred_val)
#        gini_rus_dart.append(gini)
#        pred_rus = classifier.predict_proba(X_test_rus)
#        pred_rus_dart.append(pred_rus)
#        print("eta: ",eta,"  leaves: ",leaves )
#        print("*** gini = ",gini,"***")
        
#        sub_rus = pd.DataFrame()
#        sub_rus['id'] = test_id
#        sub_rus['target'] = pred_rus[:,1]

#classifier.fit(X_all,y)
#pred = classifier.predict_proba(X_all_test)
#
#sub_lgbm = pd.DataFrame()
#sub_lgbm['id'] = test_id
#sub_lgbm['target'] = pred[:,1]


 
# K-fold cross validationn
#k = 5
#skf = StratifiedKFold(n_splits=k, random_state=42)
#
## XGBoost parameters
#params = {'min_child_weight': 10.0,'objective': 'binary:logistic','max_depth': 7,'eta': 0.2,'gamma': 0.65,'num_boost_round' : 700}

# Define training examples and targets and prepare submission datafram



#sub_rus = pd.DataFrame()
#sub_rus['id'] = test_id
#sub_rus['target'] = np.zeros_like(test_id)
#
## Train model and make predictions
#for i, (train_index, test_index) in enumerate(skf.split(X_rus, y_rus)):
#    print('[Fold %d/%d]' % (i + 1, k))
#    # Split into training and validation sets
#    X_train, X_valid = X_rus[train_index], X_rus[test_index]
#    y_train, y_valid = y_rus[train_index], y_rus[test_index]
#    # Convert data into XGBoost format
#    d_train = xgb.DMatrix(X_train, y_train)
#    d_valid = xgb.DMatrix(X_valid, y_valid)
#    d_test = xgb.DMatrix(X_test_rus)
#    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#
#    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
#    # and the custom metric (maximize=True tells xgb that higher metric is better)
#    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)
#
#    print('[Fold %d/%d Prediciton:]' % (i + 1, k))
#    # Predict on our test data
#    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
#    sub_rus['target'] += p_test/k
  
#sub_cc = pd.DataFrame()
#sub_cc['id'] = test_id
#sub_cc['target'] = np.zeros_like(test_id)
#
## Train model and make predictions
#for i, (train_index, test_index) in enumerate(skf.split(X_cc, y_cc)):
#    print('[Fold %d/%d]' % (i + 1, k))
#    # Split into training and validation sets
#    X_train, X_valid = X_cc[train_index], X_cc[test_index]
#    y_train, y_valid = y_cc[train_index], y_cc[test_index]
#    # Convert data into XGBoost format
#    d_train = xgb.DMatrix(X_train, y_train)
#    d_valid = xgb.DMatrix(X_valid, y_valid)
#    d_test = xgb.DMatrix(features_test)
#    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#
#    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
#    # and the custom metric (maximize=True tells xgb that higher metric is better)
#    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)
#
#    print('[Fold %d/%d Prediciton:]' % (i + 1, k))
#    # Predict on our test data
#    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
#    sub_cc['target'] += p_test/k    