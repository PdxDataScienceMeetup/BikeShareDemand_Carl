from sklearn import svm, linear_model
from sklearn.cross_validation import train_test_split, PredefinedSplit
from sklearn.preprocessing import scale, StandardScaler
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt
from load import load
import datetime
import pandas
import numpy
from scipy.optimize import fmin



#for h in xrange(24):
#  vec = numpy.zeros(len(X))
#  vec[(train['hour'] == h).values] = 1.0
#  X['h%d' % h] = vec


def cv_split(data, step=30, predict=10, start=30):
  """
  Take the last n days from each month and return all previous
  days as the training dataset. Returns iterator through months.
  If a month is given, return the split for only that month.

  Arguments:
    start - Where to start the rolling cross validation (days from
      first day).
    step - Step size in days to next cross-validation split.
    predict - Number of days to predict at each split.

  """
  start = 24 * start
  step = 24 * step
  predict = 24 * predict

  for i in xrange(start, len(data) - predict, step):
    yield data.iloc[:i], data.iloc[i:i + predict]



if False:
  for train, test in cv_split(load('train.csv')):
    print len(train), len(test)
  
  sys.exit()



class Features:
  def __init__(self, weights=numpy.ones(8)):
    self.scaler = StandardScaler()
    self._weights = numpy.diag(weights)

  def fit_transform(self, Xdf):
    X = self.pick(Xdf)
    X = self.scaler.fit_transform(X)
    return self.weight(X)

  def weight(self, X):
    # Reverse some scalings for important features
    X = numpy.dot(X, self._weights)
    #X[:,1] *= 2.0 # hour
    return X

  def transform(self, Xdf):
    X = self.pick(Xdf)
    X = self.scaler.transform(X)
    return self.weight(X)

  def pick(self, data):
    X = pandas.DataFrame()
    X['weather'] = data['weather']
    X['hour'] = data['hour']
    X['hourage'] = data['hourage']
    X['workingday'] = data['workingday']
    X['holiday'] = 1.0 * data['holiday']
    X['temp'] = data['temp']
    X['humidity'] = data['humidity']
    X['windspeed'] = data['windspeed']
    return X


def opt_svr():
  """
  Optimize weights in SVM kernel. A larger variance in a feature value
  increases its "importance" or "weight" in support vector regression
  with a Gaussian kernel. Optimize these weights for the best predictions.

  This takes a while and has not converged for me. However, it wasn't a total
  failure. I got some useful weights by printing them at each step of the opt,
  which improved things somewhat.
  """

  data = load('train.csv')

  def f(x):
    scores = []
    epsilon = x[0]
    weights = x[1:]
    print 'training with epsilon:', epsilon
    print 'training with weights:', weights

    for train, test in cv_split(data, start=100, step=100, predict=10):
      features = Features(weights=weights)
      X_train = features.fit_transform(train)
      X_test = features.transform(test)

      yscaler = StandardScaler()
      y_train = yscaler.fit_transform(numpy.asarray(train['count'].values, dtype='float'))
      y_test = yscaler.transform(numpy.asarray(test['count'].values, dtype='float'))

      svr = svm.SVR(kernel='rbf', degree=3, verbose=True, tol=1.0e-6, shrinking=False, epsilon=epsilon)
      svr.fit(X_train, y_train)
      #y_pred = svr.predict(X_test)

      #plt.plot_date(train['dates'], train['count'], c='b')
      #plt.plot_date(test['dates'], yscaler.inverse_transform(y_test), c='g')
      #plt.plot_date(test['dates'], yscaler.inverse_transform(y_pred), c='r')
      #plt.show()

      scores.append(1.0 - svr.score(X_test, y_test))
      print '  sub-score:', scores[-1]

    avg_score = numpy.mean(scores)
    print 'score:', avg_score
    print ''
    return avg_score

  # epsilon, feature-weights... results from a previous optimization. Negative
  # values should be identical to positive ??.
  x0 = [0.1, 1.40158166, 9.34358376, 2.03396555, 1.36130647, 1.0, 1.0, 1.0, 1.0]
  x0 = [0.02, 1.40148374, 9.34562319, 2.16667722, 1.43079, -2.87906537, -1.13396606, -1.08639336, 0.18894128]
  res = fmin(f, x0)
  print res



def grid_search():
  """
  Scikit learn grid search over SVR hyper-parameters.

  This takes a long time.
  """

  data = load('train.csv')
  test = load('test.csv')

  grid = [
    {'C': [1, 10, 100], 'epsilon': [0.05, 0.1, 0.2]}
  ]

  for train, test in cv_split(data, start=100, step=100, predict=10):
    features = Features()
    X_train = features.fit_transform(train)
    X_test = features.transform(test)

    yscaler = StandardScaler()
    y_train = yscaler.fit_transform(numpy.asarray(train['count'].values, dtype='float'))
    y_test = yscaler.transform(numpy.asarray(test['count'].values, dtype='float'))

    X = numpy.concatenate([X_train, X_test])
    y = numpy.concatenate([y_train, y_test])

    cv = PredefinedSplit(len(train) * [-1] + len(test) * [0])
    svr = svm.SVR(kernel='rbf', degree=3, verbose=True, tol=1.0e-6, shrinking=False)
    gs = GridSearchCV(svr, grid, cv=cv, scoring='r2', n_jobs=1)
    gs.fit(X, y)

    print 'Best params'
    print gs.best_params_

    for params, mean_score, scores in gs.grid_scores_:
      print("%0.3f (+/-%0.03f) for %r"
        % (mean_score, scores.std() * 2, params))

    #y_true, y_pred = y_test, clf.predict(X_test)
    #print(classification_report(y_true, y_pred))

    #y_pred = svr.predict(X_test)

    #plt.plot_date(train['dates'], train['count'], c='b')
    #plt.plot_date(test['dates'], yscaler.inverse_transform(y_test), c='g')
    #plt.plot_date(test['dates'], yscaler.inverse_transform(y_pred), c='r')
    #plt.show()

    scores.append(1.0 - svr.score(X_test, y_test))
    print '  sub-score:', scores[-1]




def compete(submit=False):
  train = load('train.csv')
  test = load('test.csv')

  # Wow. Bad.
  weights = [1.40148374, 9.34562319, 2.16667722, 1.43079, -2.87906537, -1.13396606, -1.08639336, 0.18894128]
  #weights = [1.40148374, 9.34562319, 2.16667722, 1.43079, 2.87906537, 1.13396606, 1.08639336, 0.18894128]

  # terminated opt
  #weights = [ 1.40158166  9.34358376  2.03396555  1.36130647 -2.9131261  -1.11267807  -1.09537982  0.29371516]

  features = Features(weights=weights)
  X = features.fit_transform(train)
  X_test = features.transform(test)

  yscaler = StandardScaler()
  y = yscaler.fit_transform(numpy.asarray(train['count'].values, dtype='float'))

  svr = svm.SVR(kernel='rbf', degree=3, verbose=True, tol=1.0e-6, shrinking=False, epsilon=0.05)
  svr.fit(X, y)
  y_pred = svr.predict(X_test)

  print 'score:', svr.score(X, y)

  plt.plot_date(train['dates'], train['count'], c='b')
  plt.plot_date(test['dates'], yscaler.inverse_transform(y_pred), c='g')
  plt.show()

  if submit:
    y_pred = yscaler.inverse_transform(y_pred)
    y_pred[y_pred < 0.0] = 0.0

    submission = pandas.DataFrame()
    submission['datetime'] = test['dates']
    submission['count'] = y_pred #.round().astype('int')
    submission.to_csv('submission-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.csv', index=False)


compete(submit=False)
