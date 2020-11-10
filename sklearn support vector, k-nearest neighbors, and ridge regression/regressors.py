import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# load and split dataset
boston = load_boston()
X = boston.data
y = boston.target

# perform preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X_pre = scaler.transform(X)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pre,y,test_size=0.2,random_state=42)
# train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.1,random_state=42)

# support vector regression
SVR_val_score_rbf = []
SVR_train_score_rbf = []
SVR_test_score_rbf = []

SVR_val_score_poly2 = []
SVR_train_score_poly2 = []
SVR_test_score_poly2 = []

SVR_val_score_poly3 = []
SVR_train_score_poly3 = []
SVR_test_score_poly3 = []

SVR_val_score_poly4 = []
SVR_train_score_poly4 = []
SVR_test_score_poly4 = []

SVR_val_score_poly5 = []
SVR_train_score_poly5 = []
SVR_test_score_poly5 = []

C = []

for c in range(1,21):
    reg_rbf = SVR(kernel="rbf",C=c, epsilon=0.1)
    reg_rbf.fit(X_train,y_train)
    score_rbf_train = reg_rbf.score(X_train,y_train)
    score_rbf_val = reg_rbf.score(X_val,y_val)
    score_rbf_test = reg_rbf.score(X_test,y_test)
    SVR_val_score_rbf.append(score_rbf_val)
    SVR_train_score_rbf.append(score_rbf_train)
    SVR_test_score_rbf.append(score_rbf_test)
    
    reg_poly = SVR(kernel="poly",degree=2, C=c, epsilon=0.1)
    reg_poly.fit(X_train,y_train)
    score_poly_train = reg_poly.score(X_train,y_train)
    score_poly_val = reg_poly.score(X_val,y_val)
    score_poly_test = reg_poly.score(X_test,y_test)
    SVR_val_score_poly2.append(score_poly_val)
    SVR_train_score_poly2.append(score_poly_train)
    SVR_test_score_poly2.append(score_poly_test)
    
    reg_poly = SVR(kernel="poly",degree=3, C=c, epsilon=0.1)
    reg_poly.fit(X_train,y_train)
    score_poly_train = reg_poly.score(X_train,y_train)
    score_poly_val = reg_poly.score(X_val,y_val)
    score_poly_test = reg_poly.score(X_test,y_test)
    SVR_val_score_poly3.append(score_poly_val)
    SVR_train_score_poly3.append(score_poly_train)
    SVR_test_score_poly3.append(score_poly_test)
    
    reg_poly = SVR(kernel="poly",degree=4, C=c, epsilon=0.1)
    reg_poly.fit(X_train,y_train)
    score_poly_train = reg_poly.score(X_train,y_train)
    score_poly_val = reg_poly.score(X_val,y_val)
    score_poly_test = reg_poly.score(X_test,y_test)
    SVR_val_score_poly4.append(score_poly_val)
    SVR_train_score_poly4.append(score_poly_train)
    SVR_test_score_poly4.append(score_poly_test)
    
    reg_poly = SVR(kernel="poly",degree=5, C=c, epsilon=0.1)
    reg_poly.fit(X_train,y_train)
    score_poly_train = reg_poly.score(X_train,y_train)
    score_poly_val = reg_poly.score(X_val,y_val)
    score_poly_test = reg_poly.score(X_test,y_test)
    SVR_val_score_poly5.append(score_poly_val)
    SVR_train_score_poly5.append(score_poly_train)
    SVR_test_score_poly5.append(score_poly_test)
    
    
    C.append(c)
    
plt.figure()
plt.plot(C,SVR_val_score_rbf, "r*", label = "rbf")
plt.plot(C,SVR_val_score_poly2,"b*",label="2nd Degree Polynomial")
plt.plot(C,SVR_val_score_poly3,"g*",label="3rd Degree Polynomial")
plt.plot(C,SVR_val_score_poly4,"c*",label="4th Degree Polynomial")
plt.plot(C,SVR_val_score_poly5,"m*",label="5th Degree Polynomial")
plt.xlabel("Regularization Parameter")
plt.ylabel("R2")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()




# k-nearest neighbors regression
knn_val_score = []
knn_train_score = []
knn_test_score = []
neighbors = []
for k in range (1,10):
    reg_knn = KNeighborsRegressor(n_neighbors=k)
    reg_knn.fit(X_train,y_train)
    score_knn_train = reg_knn.score(X_train,y_train)
    score_knn_val = reg_knn.score(X_val,y_val)
    score_knn_test = reg_knn.score(X_test,y_test)
    knn_val_score.append(score_knn_val)
    knn_train_score.append(score_knn_train)
    knn_test_score.append(score_knn_test)
    neighbors.append(k)
plt.figure()
plt.plot(neighbors,knn_val_score,"b*")
plt.xlabel("Number of Neighbors")
plt.ylabel("R2")
plt.show()

 Ridge regression
A = []
score_train_Ridge = []
score_val_Ridge = []
score_test_Ridge = []
for a in range(1,150):
    alpha = 150/a
    reg_Ridge = Ridge(alpha=alpha)
    reg_Ridge.fit(X_train,y_train)
    score_Ridge_train = reg_Ridge.score(X_train,y_train)
    score_Ridge_val = reg_Ridge.score(X_val,y_val)
    score_Ridge_test = reg_Ridge.score(X_test,y_test)
    A.append(alpha)
    score_train_Ridge.append(score_Ridge_train)
    score_val_Ridge.append(score_Ridge_val)
    score_test_Ridge.append(score_Ridge_test)
plt.figure()
plt.plot(A,score_val_Ridge,"b*")
plt.xlabel("Regularization Parameter")
plt.ylabel("R2")
plt.show()
