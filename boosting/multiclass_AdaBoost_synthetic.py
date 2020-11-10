import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DS import DecisionStump
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# generate noisy synthetic data
X, y = make_classification(n_samples=2000, n_features=20, n_informative=5, n_redundant=15, n_clusters_per_class=1, n_classes=5, random_state=42, flip_y=0.1)

# split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## YOUR CODE HERE

# AdaBoost loop
T = 5000
t = np.zeros((T,1))
m_train = y_train.shape[0]
m_test = y_test.shape[0]
D_1 = np.ones(m_train)/m_train
D_2 = np.ones(m_train)/m_train
D_3 = np.ones(m_train)/m_train
D_4 = np.ones(m_train)/m_train
D_5 = np.ones(m_train)/m_train
ds = DecisionStump()
hs_1 = np.zeros(m_train)
hs_2 = np.zeros(m_train)
hs_3 = np.zeros(m_train)
hs_4 = np.zeros(m_train)
hs_5 = np.zeros(m_train)
hs_1_test = np.zeros(m_test)
hs_2_test = np.zeros(m_test)
hs_3_test = np.zeros(m_test)
hs_4_test = np.zeros(m_test)
hs_5_test = np.zeros(m_test)
y_train_1 = np.ones(m_train)*-1
y_train_2 = np.ones(m_train)*-1
y_train_3 = np.ones(m_train)*-1
y_train_4 = np.ones(m_train)*-1
y_train_5 = np.ones(m_train)*-1
zeros = np.where(y_train == 0)
ones = np.where(y_train == 1)
twos = np.where(y_train == 2)
threes = np.where(y_train == 3)
fours = np.where (y_train ==4)
y_train_1[zeros] = 1
y_train_2[ones] = 1
y_train_3[twos] = 1
y_train_4[threes] = 1
y_train_5[fours] = 1

Training_Error = np.ones((T,1))
Test_Error = np.ones((T,1))
for tt in range(T):
    t[tt] = tt
    # k = 1 
    ds = DecisionStump()
    ds.fit(X_train, y_train_1, D_1)
    y_pred_1 = ds.predict(X_train)
    y_test_1 = ds.predict(X_test)

    et_1 = D_1.dot(y_train_1 != y_pred_1)
    wt_1 = np.log(1/et_1 - 1) / 2
    D_1 = D_1 * np.exp(-wt_1*y_train_1*y_pred_1) / sum(D_1 * np.exp(-wt_1*y_train_1*y_pred_1))
    hs_1 = hs_1 + wt_1*y_pred_1  
    y_pred_1 = np.sign(hs_1)
    hs_1_test = hs_1_test+wt_1*y_test_1
    y_test_1 = np.sign(hs_1_test)
    
    # k = 2 
    ds = DecisionStump()
    ds.fit(X_train, y_train_2, D_2)
    y_pred_2 = ds.predict(X_train)
    y_test_2 = ds.predict(X_test)

    et_2 = D_2.dot(y_train_2 != y_pred_2)
    wt_2 = np.log(1/et_2 - 1) / 2
    D_2 = D_2 * np.exp(-wt_2*y_train_2*y_pred_2) / sum(D_2 * np.exp(-wt_2*y_train_2*y_pred_2))
    hs_2 = hs_2 + wt_2*y_pred_2  
    y_pred_2 = np.sign(hs_2)
    hs_2_test = hs_2_test+wt_2*y_test_2
    y_test_2 = np.sign(hs_2_test)
    
    # k = 3 
    ds = DecisionStump()
    ds.fit(X_train, y_train_3, D_3)
    y_pred_3 = ds.predict(X_train)
    y_test_3 = ds.predict(X_test)

    et_3 = D_3.dot(y_train_3 != y_pred_3)
    wt_3 = np.log(1/et_3 - 1) / 2
    D_3 = D_3 * np.exp(-wt_3*y_train_3*y_pred_3) / sum(D_3 * np.exp(-wt_3*y_train_3*y_pred_3))
    hs_3 = hs_3 + wt_3*y_pred_3  
    y_pred_3 = np.sign(hs_3)
    hs_3_test = hs_3_test+wt_3*y_test_3
    y_test_3 = np.sign(hs_3_test)
    
    # k = 4 
    ds = DecisionStump()
    ds.fit(X_train, y_train_4, D_4)
    y_pred_4 = ds.predict(X_train)
    y_test_4 = ds.predict(X_test)

    et_4 = D_4.dot(y_train_4 != y_pred_4)
    wt_4 = np.log(1/et_4 - 1) / 2
    D_4 = D_4 * np.exp(-wt_4*y_train_4*y_pred_4) / sum(D_4 * np.exp(-wt_4*y_train_4*y_pred_4))
    hs_4 = hs_4 + wt_4*y_pred_4  
    y_pred_4 = np.sign(hs_4)
    hs_4_test = hs_4_test+wt_4*y_test_4
    y_test_4 = np.sign(hs_4_test)
    
    # k = 5 
    ds = DecisionStump()
    ds.fit(X_train, y_train_5, D_5)
    y_pred_5 = ds.predict(X_train)
    y_test_5 = ds.predict(X_test)

    et_5 = D_5.dot(y_train_5 != y_pred_5)
    wt_5 = np.log(1/et_5 - 1) / 2
    D_5 = D_5 * np.exp(-wt_5*y_train_5*y_pred_5) / sum(D_5 * np.exp(-wt_5*y_train_5*y_pred_5))
    hs_5 = hs_5 + wt_5*y_pred_5  
    y_pred_5 = np.sign(hs_5)
    hs_5_test = hs_5_test+wt_5*y_test_5
    y_test_5 = np.sign(hs_5_test)
    
    y_pred = np.concatenate((y_pred_1.reshape(y_train.shape[0],1),y_pred_2.reshape(y_train.shape[0],1),y_pred_3.reshape(y_train.shape[0],1),y_pred_4.reshape(y_train.shape[0],1),y_pred_5.reshape(y_train.shape[0],1)),axis=1)
    predictions = np.argmax(y_pred,axis=1)
    y_test_pred = np.concatenate((y_test_1.reshape(y_test.shape[0],1),y_test_2.reshape(y_test.shape[0],1),y_test_3.reshape(y_test.shape[0],1),y_test_4.reshape(y_test.shape[0],1),y_test_5.reshape(y_test.shape[0],1)),axis=1)
    predictions_test = np.argmax(y_test_pred,axis=1)
    
    error_training = sum(predictions!=y_train)/y_train.shape[0]
    Training_Error[tt] = error_training
    error_test = sum(predictions_test!=y_test)/y_test.shape[0]
    Test_Error[tt] = error_test

plt.figure()
plt.plot(t,Training_Error,"b*",label="Training Error")
plt.plot(t,Test_Error,"r*",label="Test Error")
plt.xlabel("Number of Boosting Rounds")
plt.ylabel("Error (%)")
plt.legend()
plt.show()
#
#plt.figure()
##plt.scatter(X[:,0], X[:,1], c=np.sign(hs), cmap='Spectral', s=1)
#plt.scatter(X[:,0], labels, c=predictions, cmap='Spectral', s=1)
#plt.title('predicted')
#plt.colorbar()
#plt.show()
#
#
#
