import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



cancer = load_breast_cancer()
X = cancer.data
y = 2 * cancer.target - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
m_train = X_train.shape[0]
m_test = X_test.shape[0]

# Gaussian Kernel function
def gaussian_kernel(X1,X2,sigma):  
    X1_norm = np.sum(X1**2,axis=-1)
    X2_norm = np.sum(X2**2,axis=-1)
    K = np.exp(-(X1_norm[:, None]+X2_norm[None,:]-2*np.dot(X1,X2.T))/2*sigma)
    return K

sigma = 0.0001 
l = 0.001  
K_train = gaussian_kernel(X_train, X_train, sigma)
K_test = gaussian_kernel(X_train, X_test, sigma)

# Implement Kernalized SGD for Solving Soft-SVM
B = np.zeros(m_train)
a = np.zeros(m_train)
a_sum = a
T = 10000
np.random.seed(42)
for t in range(1, T):  
    a = (1/(l*t))*B
    i = np.random.randint(m_train)
    if y_train[i] * np.dot(a, K_train[i, :]) < 1:
        B[i] = B[i] + y_train[i]
    a_sum = a_sum+a
a_bar = a_sum/T

# Make predictions
pred_train = np.sign(np.dot(a_bar,K_train))
pred_test = np.sign(np.dot(a_bar,K_test))

# Calcualte error
error_training = 100*(np.sum(pred_train != y_train)/m_train)
error_test = 100*(np.sum(pred_test != y_test)/m_test)
print('Training error:', error_training)
print('Test error:', error_test)
