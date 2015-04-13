from sklearn import decomposition
import matplotlib.pyplot as plt
from load import *

train = load('train.csv')

pca = decomposition.PCA()
pca.fit(train[['hour', 'age_seconds', 'weather']])

print pca.components_

plt.plot(pca.explained_variance_)
plt.show()
