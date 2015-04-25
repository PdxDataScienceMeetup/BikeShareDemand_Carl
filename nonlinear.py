import numpy
import pandas
from load import load
from models import ProductModel, SumModel, Model, AsymmGaussian
from scipy import optimize
from matplotlib import pyplot as plt
import datetime
import sys

train = load('train.csv')
#train = train[train['month'] == '2011-01']
train['normcount'] = train['count'] / train['count'].mean()
mean_count = train['count'].mean()



class EnumModel(Model):
  """
  Generic model that returns the parameter value at index i
  if the value of the input in the specified column matches the
  choice at index i of the given choices array.

  For example

    model = EnumModel(train, 'hour', choices=range(24), ...)
    print model(x, beta)

  where x['hour'] == 11, would print the value of beta[11].
  """

  def __init__(self, train, column, choices=[], beta0=None, name='Enum model', bounds=None):
    Model.__init__(self, train)
    self.choices = choices
    self.column = column
    self.name = name
    self._beta0 = list(beta0)
    self._bounds = list(bounds)

  def bounds(self):
    return self._bounds

  def beta0(self):
    return self._beta0

  def show(self, beta):
    print self.name
    for coeff, choice in zip(beta, self.choices):
      print '  %s: %f' % (str(choice), coeff)

  def __call__(self, x, beta, grad=False):
    values = x[self.column]
    f = numpy.zeros(len(x))

    for i, choice in enumerate(self.choices):
      f[(values == choice).values] = beta[i]
  
    if grad == False:
      return f

    grad = numpy.zeros((len(x), len(beta)))

    for i, choice in enumerate(self.choices):
      grad[(values == choice).values, i] = 1.0

    return f, grad



class WeatherEffect(Model):
  """
  One can now use EnumModel instead for this...
  """

  def beta0(self):
    return [0.89310891, 0.50763792, 0.57599695]
    #return 3 * [1.0]

  def bounds(self):
    return 3 * [(0.0, 1.0)]

  def show(self, beta):
    print 'Weather effect'
    print ' ', beta

  def __call__(self, x, beta, grad=True):
    weather = x['weather']
    f = numpy.ones(len(weather))
  
    f[(weather == 2).values] = beta[0]
    f[(weather == 3).values] = beta[1]
    f[(weather == 4).values] = beta[2]
  
    if grad == False:
      return f

    grad = numpy.zeros((len(weather), len(beta)))
    grad[(weather == 2).values, 0] = 1.0
    grad[(weather == 3).values, 1] = 1.0
    grad[(weather == 4).values, 2] = 1.0

    return f, grad


class SmoothedWeatherEffect(AsymmGaussian):
  """
  Asymmetric gaussian fall-off as weather increases. Weather value is
  averaged over previous N hours.
  """

  def __init__(self, train):
    AsymmGaussian.__init__(self, self.smooth(train), lo=False, name='Smoothed weather effect')

  def smooth(self, x):
    return pandas.rolling_mean(x['weather'], window=2, min_periods=1).values

  def beta0(self):
    return [1.5, 0.5]

  def __call__(self, x, beta, grad=True):
    if isinstance(x, pandas.DataFrame):
      x = self.smooth(x)

    return AsymmGaussian.__call__(self, x, beta, grad=grad)

#SmoothedWeatherEffect(train).show([1.5, 1.0])

class DaytypeEffect(Model):
  def beta0(self):
    return 3 * [1.0]

  def bounds(self):
    return 3 * [(0.0, 1.0)]

  def show(self, beta):
    print '%s day type effect' % self.name
    print ' ', beta

  def __call__(self, x, beta, grad=True):
    N = len(x)
    res = numpy.zeros(N)
    res[(x['workingday'] == 1).values] = beta[0]
    res[((x['workingday'] == 0) & (x['holiday'] == 0)).values] = beta[1] #weekend
    res[(x['holiday'] == 1).values] = beta[2] #holiday

    if grad == False:
      return res

    grad = numpy.zeros((N, 3))
    grad[(x['workingday'] == 1).values, 0] = 1.0
    grad[((x['workingday'] == 0) & (x['holiday'] == 0)).values, 1] = 1.0
    grad[(x['holiday'] == 1).values, 2] = 1.0

    return res, grad


class DailyPattern(Model):
  """
  24 parameters. One for each hour of the day.
  """

  def __init__(self, train, name='Unnamed'):
    self.name = name
    Model.__init__(self, train)

  def beta0(self):
    """
    Initial guess is a vertically shifted cosine curve with some
    random perturbations to help optimization algorithms.
    """
    h = numpy.linspace(0.0, 23.0, 24)
    b0 = 0.5 * (1.0 - numpy.cos(numpy.pi * (h - 3.0) / 12.0))
    # perturb it.
    delta = 0.1
    b0 = b0 * (1.0 - delta * numpy.random.random(24))
    return list(b0)

  def bounds(self):
    return 24 * [(0.0, None)]

  def show(self, beta):
    print '%s daily pattern' % self.name
    print ' ', beta

  def __call__(self, x, beta, grad=True):
    N = len(x)
    res = numpy.zeros(N)
  
    for i, value in enumerate(beta):
      res[(x['hour'] == i).values] = value

    if grad == False:
      return res

    grad = numpy.zeros((N, 24))

    for hour in xrange(24):
      grad[(x['hour'] == hour).values, hour] = 1.0

    return res, grad



class WindspeedEffect(AsymmGaussian):
  """
  Asymmetric Gaussian factor. 1 at low windspeed, drops as wind increases.
  """

  def __init__(self, train):
    AsymmGaussian.__init__(self, train, lo=False, name='Windspeed effect')

  def __call__(self, x, beta, grad=False):
    return AsymmGaussian.__call__(self, x['windspeed'].values, beta, grad=grad)

  def beta0(self):
    return [35.0, 10.0]


class HumidityEffect(AsymmGaussian):
  """
  Asymmetric Gaussian factor. 1 at low humidity, drops with rising humidity.
  """

  def __init__(self, train):
    AsymmGaussian.__init__(self, train, lo=False, name='Humidity effect')

  def __call__(self, x, beta, grad=False):
    return AsymmGaussian.__call__(self, x['humidity'].values, beta, grad=grad)

  def beta0(self):
    return [60.0, 30.0]


class TempEffect(AsymmGaussian):
  """
  Asymmetric Gaussian factor. 1 at ideal temperature, drops to zero on
  either side.
  """

  def __init__(self, train, name='Temperature effect'):
    AsymmGaussian.__init__(self, train, name=name)

  def __call__(self, x, beta, grad=False):
    return AsymmGaussian.__call__(self, x['temp'].values, beta, grad=grad)

  def beta0(self):
    return [31.0, 15.0, 40.0]
    #ideal T: 31.5443258446
    #lower dT: 16.9986489988
    #upper dT: 40.3906223208


class LinearPopularity(Model):
  def __init__(self, train):
    Model.__init__(self, train)
    dayage = train['dayage'].values
    self.begin = dayage[0]
    self.end = dayage[-1]
    self.mean = train['count'].mean()

  def beta0(self):
    return 2 * [self.mean]

  def bounds(self):
    return 2 * [(0.0, None)]

  def show(self, beta):
    print 'Linear popularity'
    print '  beginning: %.1f' % beta[0]
    print '  ending:    %.1f' % beta[1]

  def __call__(self, x, beta, grad=True):
    """
    A linear approximation. Assume popularity changes steadily
    over the given data range. The two parameters are the popularities
    at the beginning and end of the data range (so we can bound them
    from below).
    """
    N = len(x)
    dayage = x['dayage'].values
    slope = (beta[1] - beta[0]) / (self.end - self.begin)
    f = slope * (dayage - self.begin) + beta[1]

    if grad == False:
      return f

    grad = numpy.ones((N, 2))
    grad[:, 0] = (self.end - dayage) / (self.end - self.begin)
    grad[:, 1] = (dayage - self.begin) / (self.end - self.begin)

    return f, grad


class IncreasingPopularity(Model):
  """
  Starts somewhere and increases linearly month-to-month.
  Parameters are specified as semi-positive popularity
  increases and one starting value.

  The popularity at any given time is a function of the
  starting value and all previous increases.
  """

  def __init__(self, train):
    Model.__init__(self, train)
    self.bymonth = bymonth = train.groupby('month')

    # First of each month and last of last month
    dayages = bymonth.first()['dayage'].values
    self.dayages = numpy.concatenate([dayages, train.iloc[-1:]['dayage'].values])

    self.means = bymonth['count'].mean()
    self.nsegments = len(bymonth)
    self.nparams = self.nsegments + 1

  def bounds(self):
    return (self.nsegments + 1) * [(0.0, None)]

  def show(self, beta):
    print 'Popularity'
    dates = self.bymonth.first()['dates'].values
    dates = numpy.concatenate([dates, self.bymonth.last()['dates'][-1:].values])
    plt.plot_date(dates, pandas.Series(beta).cumsum())
    plt.show()
    plt.clf()

  def beta0(self):
    """
    Initial guess is a linear from smallest to largest of monthly means.
    """
    start = self.means.min()
    end = self.means.max()
    dp = (end - start) / float(self.nsegments)
    return [start] + self.nsegments * [dp]

  def __call__(self, x, beta, grad=True):
    """
    In segment i, between days d_i and d_{i+1}, the popularity is

                               beta_{i+1} (d - d_i)
      p = sum_{n=1}^i beta_n + --------------------
                                   d_{i+1} - d_i


        dp                                  delta_{i+1,j} (d - d_i)
      ------- = sum_{n=1}^i delta_{n,j} + -------------------------
      dbeta_j                                   d_{i+1} - d_i

    which equals 1 when d < d_i, and (d-d_i)/(d_{i+1}-d_i) when
    d >= d_i

    """
    N = len(x)
    f = numpy.interp(x['dayage'].values, self.dayages, pandas.Series(beta).cumsum())

    if grad == False:
      return f

    grad = numpy.zeros((N, self.nparams))
    days = x['dayage'].values

    for i in xrange(self.nsegments):
      d_lo = self.dayages[i]
      d_hi = self.dayages[i + 1]
      indices = (days >= d_lo) & (days < d_hi)

      for j in xrange(i + 2):
        if j == i + 1:
          grad[indices, j] = (days[indices] - d_lo) / (d_hi - d_lo)
        else:
          grad[indices, j] = 1.0

    # Finally for predicted values
    grad[days >= d_hi,:] = 1.0

    return f, grad


class NonlinearBikeShareModel:
  def __init__(self, train, error_alpha=100.0):
    self.train = train
    self.weights = numpy.exp(-(train.iloc[-1]['dayage'] - train['dayage']) / error_alpha)

    self.model = SumModel(train, [
      ProductModel(train, [
        #IncreasingPopularity(train),
        WindspeedEffect(train),
        EnumModel(train, 'weather', choices=range(1, 5), name='Weather', beta0=numpy.ones(4), bounds=4 * [(0.0, 1.0)]),
        TempEffect(train, name='Workday temp effect'),
        EnumModel(train, 'workingday', choices=[0, 1], name='Workday', beta0=[0.0, 1.0], bounds=2 * [(0.0, None)]),
        DailyPattern(train, 'Workday pattern')
      ]),
      ProductModel(train, [
        #IncreasingPopularity(train),
        WindspeedEffect(train),
        EnumModel(train, 'weather', choices=range(1, 5), name='Weather', beta0=numpy.ones(4), bounds=4 * [(0.0, 1.0)]),
        TempEffect(train, name='Non-workday temp effect'),
        EnumModel(train, 'workingday', choices=[0, 1], name='Non-workday', beta0=[1.0, 0.0], bounds=2 * [(0.0, None)]),
        DailyPattern(train, 'Non-workday pattern')
      ])
    ])

  def bounds(self):
    return self.model.bounds()

  def beta0(self):
    return self.model.beta0()

  def __call__(self, beta):
    """
    Return error and gradient of error rather than prediction for
    optimization.
    """
    f, grad_f = self.model.fit(beta, grad=True)
    e, grad_e = self.error(self.train['count'].values, f, grad_f, weighted=True)

    if numpy.isnan(e):
      grad_e[:] = 0.0
      return 1.0e6, grad_e

    return e, grad_e

  def show(self, beta):
    return self.model.show(beta)

  def predict(self, test, beta):
    return self.model(test, beta, grad=False)

  def count(self, beta):
    return self.model.fit(beta, grad=False)

  def error(self, actual, predicted, grad=None, weighted=False):
    weights = self.weights

    if weighted == False:
      weights = numpy.ones(len(actual))

    n = len(predicted)
    evec = numpy.log(1.0 + actual) - numpy.log(1.0 + predicted)
    e = numpy.sqrt(numpy.dot(weights * evec, evec) / float(n))

    if grad is not None:
      grad_e = -numpy.dot(weights * evec / (1.0 + predicted), grad) / (e * float(n))
      return e, grad_e

    return e


def explore():
  """
  Prediction for beginning of July was bad. Consistently predicted
  weekend ridership too high. Probably because of July 4th holiday
  stuff!

  Dec. 7-8 count goes really low.
  """

  nbsm = NonlinearBikeShareModel(train)
  bounds = nbsm.bounds()
  beta0 = nbsm.beta0()
  
  print 'bounds', len(bounds)
  print 'beta0', len(beta0)
  
  res = optimize.minimize(nbsm, beta0, method='L-BFGS-B', jac=True, bounds=bounds, options={'disp': True, 'maxiter': 10000})
  #res = optimize.basinhopping(nbsm, beta0, minimizer_kwargs={'method':'L-BFGS-B', 'jac':True, 'bounds':bounds, 'options': {'disp': True, 'maxiter': 500}}, disp=True, niter=1)
  
  print res
  nbsm.show(res.x)
  
  plt.plot_date(train['dates'], train['count'])
  plt.plot_date(train['dates'], nbsm.count(res.x))
  plt.show()


# Cross-validation
def cv():
  width = 10 * 24 # predict 10 days into future
  tests = 20
  #starts = numpy.random.randint(width, len(train) - width, 20)
  step = 7 * 24
  start = 12 * 30 * 24
  splits = range(start, len(train) - width, step)
  beta0 = None

  for start in splits:
    cv_train = train.iloc[:start]
    cv_test = train.iloc[start:start + width]
    nbsm = NonlinearBikeShareModel(cv_train, error_alpha=500.0)

    if beta0 is None:
      beta0 = nbsm.beta0()

    res = optimize.minimize(nbsm, beta0, method='L-BFGS-B', tol=1.0e-7, jac=True, bounds=nbsm.bounds(), options={'disp': False, 'maxiter': 10000})
    #res = optimize.basinhopping(
    #  nbsm,
    #  nbsm.beta0(),
    #  minimizer_kwargs={
    #    'method':'L-BFGS-B',
    #    'jac':True,
    #    'bounds': nbsm.bounds(),
    #    'options': {'disp': False, 'maxiter': 500}
    #  },
    #  disp=True,
    #  niter=2
    #)
    y_test = cv_test['count'].values
    error_pred = nbsm.error(y_test, nbsm.predict(cv_test, res.x))
    print 'training error:', nbsm(res.x)[0], 'prediction error:', error_pred

    beta0 = res.x

    if 1:
      nbsm.show(res.x)

      plt.subplot(211)
      plt.plot_date(cv_train['dates'], cv_train['count'] - nbsm.count(res.x), label='train errs')

      plt.plot_date(cv_test['dates'], cv_test['count'] - nbsm.predict(cv_test, res.x), label='predicted errs')

      plt.plot_date(cv_train['dates'], 25.0 * cv_train['weather'], label='weather')
      plt.plot_date(cv_train['dates'], cv_train['humidity'], label='humidity')
      plt.plot_date(cv_train['dates'], cv_train['windspeed'], label='windspeed')
      plt.plot_date(cv_train['dates'], 2.0 * cv_train['temp'], label='temp')
      plt.plot_date(cv_train['dates'], 2.0 * cv_train['atemp'], label='atemp')
      #plt.plot_date(cv_train['dates'], 200.0 * cv_train['holiday'], label='holiday')
      plt.legend()

      plt.subplot(212)
      plt.plot_date(cv_train['dates'], cv_train['count'], label='train')
      plt.plot_date(cv_train['dates'], nbsm.count(res.x), label='train fit')
      plt.plot_date(cv_test['dates'], cv_test['count'], label='test')
      plt.plot_date(cv_test['dates'], nbsm.predict(cv_test, res.x), label='predicted')
      plt.plot_date(cv_train['dates'], 50.0 * nbsm.weights, label='weights')

      plt.legend()
      plt.show()
      plt.clf()


def compete(submit=False, history=10):
  test = load('test.csv')
  predicted = []
  beta0 = None

  i = 0
  for month, month_test in test.groupby('monthage'):
    i += 1
    #if i != 20:
    #  continue

    month_test = month_test.copy()
    # Use only last history months.
    #month_train = train[(train['monthage'] <= month) & (train['monthage'] > month - history)]
    month_train = train[train['monthage'] <= month]

    nbsm = NonlinearBikeShareModel(month_train)

    #if beta0 is None:
    #  beta0 = nbsm.beta0()

    res = optimize.minimize(nbsm, nbsm.beta0(), method='L-BFGS-B', jac=True, bounds=nbsm.bounds(), options={'disp': False, 'maxiter': 10000})
    #res = optimize.basinhopping(
    #  nbsm,
    #  nbsm.beta0(),
    #  minimizer_kwargs={
    #    'method':'L-BFGS-B',
    #    'jac':True,
    #    'bounds': nbsm.bounds(),
    #    'options': {'disp': False, 'maxiter': 500}
    #  },
    #  disp=True,
    #  niter=10
    #)
    print '%s training error:' % month, nbsm(res.x)[0]

    month_test['count'] = nbsm.predict(month_test, res.x)
    #nbsm.show(res.x)
    predicted.append(month_test)
    #beta0 = res.x

  predicted = pandas.concat(predicted)

  plt.plot_date(train['dates'], train['count'], label='train')
  plt.plot_date(predicted['dates'], predicted['count'], label='predicted')
  plt.legend()
  plt.show()
  plt.clf()

  if submit:
    submission = pandas.DataFrame()
    submission['datetime'] = test['dates']
    submission['count'] = predicted['count'] #.round().astype('int')
    submission.to_csv('submission-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.csv', index=False)
  #print res


#explore()
cv()
#compete(submit=False)
