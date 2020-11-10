import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from DS import DecisionStump

m = 1000

# first dataset
X = np.sort(np.random.rand(m,1), axis=0)
y = np.zeros(m)
y[X[:,0] < 0.3] = 1
y[X[:,0] > 0.9] = 1

# convert labels to +/-1
y = 2*y - 1

# fit stump
D = np.ones(m)/m
ds = DecisionStump()
ds.fit(X, y, D)
ypred = ds.predict(X)
print('dataset 1 training error: ', ds.error(y, ypred))

plt.figure()
plt.scatter(X[:,0], y, c=y, cmap='Spectral', s=1)
plt.title('true')
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(X[:,0], y, c=ypred, cmap='Spectral', s=1)
plt.title('predicted')
plt.colorbar()
plt.show()

# second dataset
X, y = make_moons(m, noise=0.1)

# convert labels to +/-1
y = 2*y - 1

# fit stump
D = np.ones(m)/m
ds = DecisionStump()
ds.fit(X, y, D)
ypred = ds.predict(X)
print('dataset 2 training error: ', ds.error(y, ypred))

plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, cmap='Spectral', s=1)
plt.title('true')
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(X[:,0], X[:,1], c=ypred, cmap='Spectral', s=1)
plt.title('predicted')
plt.colorbar()
plt.show()
