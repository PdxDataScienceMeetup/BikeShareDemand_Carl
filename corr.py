import numpy
from load import load
from scipy import optimize
from matplotlib import pyplot as plt
import pandas
import sys

train = load('train.csv')
#train = train[train['month'] == '2011-01']
counts = train['count'].values
hourages = train['hourage'].values

countdiffs = numpy.subtract.outer(counts, counts).flatten()
hourdiffs = numpy.subtract.outer(hourages, hourages).flatten()

n = 10
corr = numpy.zeros(n)
devs = numpy.zeros(n)

for j in xrange(1, n):
  print 'correlation', j
  diffs = countdiffs[hourdiffs == j]
  print diffs
  corr[j] = numpy.sum(diffs) / float(len(diffs))
  devs[j] = numpy.linalg.norm(corr[j] - diffs) / numpy.sqrt(len(diffs))

#plt.plot(corr)
plt.gca().errorbar(range(len(corr)), corr, yerr=devs, fmt='o')
plt.show()
