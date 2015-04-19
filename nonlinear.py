import numpy
import pandas
from load import load
from models import ProductModel, SumModel, Model, AsymmGaussian
from scipy import optimize
from matplotlib import pyplot as plt
import sys

train = load('train.csv')
#train = train[train['month'] == '2011-01']
train['normcount'] = train['count'] / train['count'].mean()
mean_count = train['count'].mean()



class WeatherEffect(Model):
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


class RecDaytypeEffect(DaytypeEffect):
  def __init__(self, train):
    DaytypeEffect.__init__(self, train)
    self.name = 'Recreational'

  def beta0(self):
    return [0.02226672, 0.16870226, 0.10616399]


class CommuteDaytypeEffect(DaytypeEffect):
  def __init__(self, train):
    DaytypeEffect.__init__(self, train)
    self.name = 'Commute'

  def beta0(self):
    return [0.16633973, 0.01717605, 0.05540464]


class DailyPattern(Model):
  def beta0(self):
    return 24 * [1.0 / 24.0]

  def bounds(self):
    return 24 * [(0.0, None)]

  def show(self, beta):
    print '%s daily pattern' % self.name
    print ' ', beta

  def __call__(self, x, beta, grad=True):
    N = len(x)
    res = numpy.zeros(N)
    weights = beta #/ sum(beta)
  
    for i, weight in enumerate(weights):
      res[(x['hour'] == i).values] = weight
  
    if grad == False:
      return res

    grad = numpy.zeros((N, 24))

    for hour in xrange(24):
      grad[(x['hour'] == hour).values, hour] = 1.0

    return res, grad


class RecDailyPattern(DailyPattern):
  def __init__(self, train):
    DailyPattern.__init__(self, train)
    self.name = 'Recreational'


class CommuteDailyPattern(DailyPattern):
  def __init__(self, train):
    DailyPattern.__init__(self, train)
    self.name = 'Commute'


class WindspeedEffect(AsymmGaussian):
  def __init__(self, train):
    AsymmGaussian.__init__(self, train, lo=False, name='Windspeed effect')

  def __call__(self, x, beta, grad=False):
    return AsymmGaussian.__call__(self, x['windspeed'].values, beta, grad=grad)

  def beta0(self):
    return [35.0, 10.0]


class HumidityEffect(AsymmGaussian):
  def __init__(self, train):
    AsymmGaussian.__init__(self, train, lo=False, name='Humidity effect')

  def __call__(self, x, beta, grad=False):
    return AsymmGaussian.__call__(self, x['humidity'].values, beta, grad=grad)

  def beta0(self):
    return [60.0, 30.0]


class TempEffect(AsymmGaussian):
  def __init__(self, train):
    AsymmGaussian.__init__(self, train, name='Temperature effect')

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

  def beta0(self):
    return [1.0, 1.0]

  def bounds(self):
    return [(0.0, None)] * 2

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


class Recreational(ProductModel):
  def __init__(self, train):
    #terms = [HumidityEffect, SmoothedWeatherEffect, TempEffect, WindspeedEffect, RecDaytypeEffect, RecDailyPattern]
    terms = [RecDaytypeEffect, RecDailyPattern]
    ProductModel.__init__(self, train, terms)


class Commuter(ProductModel):
  def __init__(self, train):
    #terms = [HumidityEffect, SmoothedWeatherEffect, TempEffect, WindspeedEffect, CommuteDaytypeEffect, CommuteDailyPattern]
    terms = [CommuteDaytypeEffect, CommuteDailyPattern]
    ProductModel.__init__(self, train, terms)


class Amplitude(ProductModel):
  def __init__(self, train):
    #terms = [LinearPopularity]
    terms = [HumidityEffect, SmoothedWeatherEffect, LinearPopularity, TempEffect, WindspeedEffect]
    ProductModel.__init__(self, train, terms)


class Quantity(SumModel):
  def __init__(self, train):
    terms = [Recreational, Commuter]
    SumModel.__init__(self, train, terms)


class NonlinearBikeShareModel:
  def __init__(self, train):
    self.train = train
    self.terms = [Amplitude, Quantity]
    self.model = ProductModel(train, self.terms)

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
    e, grad_e = self.error(f, grad_f)

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

  def error(self, count, grad=None):
    actual = self.train['count'].values
    n = len(count)
    evec = numpy.log(1.0 + actual) - numpy.log(1.0 + count)
    e = numpy.sqrt(numpy.dot(evec, evec) / float(n))

    if grad is not None:
      grad_e = -numpy.dot(evec / (1.0 + count), grad) / (e * float(n))
      return e, grad_e

    return e


def explore():
  nbsm = NonlinearBikeShareModel(train)
  bounds = nbsm.bounds()
  beta0 = nbsm.beta0()
  
  print 'bounds', len(bounds)
  print 'beta0', len(beta0)
  
  #res = optimize.minimize(nbsm, beta0, method='L-BFGS-B', jac=True, bounds=bounds, options={'disp': True, 'maxiter': 1000})
  res = optimize.basinhopping(nbsm, beta0, minimizer_kwargs={'method':'L-BFGS-B', 'jac':True, 'bounds':bounds, 'options': {'disp': True, 'maxiter': 500}}, disp=True, niter=1)
  
  print res
  nbsm.show(res.x)
  
  plt.plot_date(train['dates'], train['count'])
  plt.plot_date(train['dates'], nbsm.count(res.x))
  plt.show()


# Cross-validation
def cv():
  width = 10 * 24 # predict 10 days into future
  tests = 20
  starts = numpy.random.randint(width, len(train) - width, 20)
  
  for start in starts:
    cv_train = train.iloc[:start]
    cv_test = train.iloc[start:start + width]
    nbsm = NonlinearBikeShareModel(cv_train)
    res = optimize.basinhopping(
      nbsm,
      nbsm.beta0(),
      minimizer_kwargs={
        'method':'L-BFGS-B',
        'jac':True,
        'bounds': nbsm.bounds(),
        'options': {'disp': False, 'maxiter': 500}
      },
      disp=True,
      niter=2
    )
    print 'error:', nbsm(res.x)[0]

    plt.plot_date(cv_train['dates'], cv_train['count'], label='train')
    plt.plot_date(cv_train['dates'], nbsm.count(res.x), label='train fit')
    plt.plot_date(cv_test['dates'], cv_test['count'], label='test')
    plt.plot_date(cv_test['dates'], nbsm.predict(cv_test, res.x), label='predicted')
    plt.legend()
    plt.show()
    plt.clf()


def compete():
  test = load('test.csv')
  predicted = [] #pandas.DataFrame(columns=test.columns)
  fit = [] #pandas.DataFrame(columns=test.columns)
  beta0 = None

  for month, month_test in test.groupby('month'):
    month_test = month_test.copy()
    month_train = train[train['month'] <= month]

    nbsm = NonlinearBikeShareModel(month_train)

    if beta0 is None:
      beta0 = nbsm.beta0()

    res = optimize.basinhopping(
      nbsm, beta0,
      minimizer_kwargs={
        'method':'L-BFGS-B',
        'jac':True,
        'bounds': nbsm.bounds(),
        'options': {'disp': False, 'maxiter': 500}
      },
      disp=True,
      niter=2
    )
    print '%s training error:' % month, nbsm(res.x)[0]

    month_test['count'] = nbsm.predict(month_test, res.x)
    predicted.append(month_test)
    #fit.append(nbsm.count(res.x))
    #predicted = pandas.concat([predicted, month_test])
    beta0 = res.x

  #nbsm = NonlinearBikeShareModel(train)
  predicted = pandas.concat(predicted)

  plt.plot_date(train['dates'], train['count'], label='train')
  #plt.plot_date(train['dates'], nbsm.count(beta0), label='train fit')
  #plt.plot_date(cv_test['dates'], cv_test['count'], label='test')
  plt.plot_date(predicted['dates'], predicted['count'], label='predicted')
  plt.legend()
  plt.show()
  plt.clf()

compete()
#cv()
