import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


data = loadmat('mnistFull.mat')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train'][:,0]
y_test = data['y_test'][:,0]

K = 10
d = X_test.shape[0]
W = np.random.randn(d, K)
Ntrain = len(y_train)

# Define softmax calculation
def softmax(W,x):
    global smax
    num = np.exp(W.T@x)
    denom = sum(num)
    smax = num/denom
    return smax

# Define gradient calculation
def gradient(W,x,y):
    Y = np.zeros((K,1))
    Y[y] = 1
    grad = np.kron((softmax(W,x).reshape(K,1)-Y),x)
    return grad

# SGD iterations
w = np.random.randn(K,d)
iterations = 1
ERROR = np.zeros((iterations,1))
ACCURACY = np.zeros((iterations,1))
mu = 1e-2
## YOUR SGD LOOP GOES HERE
for i in range(iterations):
    rows = np.random.permutation(Ntrain)
    for r in range(Ntrain):
        row = rows[r]
        x = X_train[:,row]
        grad = gradient(w.reshape((d,K),order="F"),x,y_train[row])
        w = w-(mu*grad)
        ## reshape to get final weight matrix
        W = w.reshape((d, K), order='F')

    # report training and test error
    P = W.T@X_test
    predictions = np.argmax(P,axis=0)
    check = predictions-y_test.T
    incorrect_predictions = np.count_nonzero(check,axis=0)
    accuracy = 100*(incorrect_predictions/y_test.shape[0])
    error = 100-accuracy
    ERROR[i] = error
    ACCURACY[i] = accuracy

# Generate accuracy and error plots
n = np.linspace(1,iterations,iterations)
plt.figure(1)
plt.plot(n,ACCURACY,"b*")
plt.xticks(n)
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")

plt.figure(2)
plt.plot(n,ERROR,"b*")
plt.xticks(n)
plt.xlabel("Iterations")
plt.ylabel("Error (%)")