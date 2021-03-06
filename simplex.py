import numpy
import pandas
from load import load
from asymm_gaussian import AsymmGaussian
from models import ProductModel, SumModel
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
    train = pandas.rolling_mean(data['weather'], window=2, min_periods=1).values
    AsymmGaussian.__init__(self, train, lo=False, name='Smoothed weather effect')

  def beta0(self):
    return [1.5, 0.5]

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

    grad = numpy.zeros((len(data), 3))
    grad[(x['workingday'] == 1).values, 0] = 1.0
    grad[((x['workingday'] == 0) & (x['holiday'] == 0)).values, 1] = 1.0
    grad[(x['holiday'] == 1).values, 2] = 1.0

    return res, grad


class RecDaytypeEffect(DaytypeEffect):
  def __init__(self, data):
    DaytypeEffect.__init__(self, data)
    self.name = 'Recreational'

  def beta0(self):
    return [0.02226672, 0.16870226, 0.10616399]


class CommuteDaytypeEffect(DaytypeEffect):
  def __init__(self, data):
    DaytypeEffect.__init__(self, data)
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
  def __init__(self, data):
    DailyPattern.__init__(self, data)
    self.name = 'Recreational'


class CommuteDailyPattern(DailyPattern):
  def __init__(self, data):
    DailyPattern.__init__(self, data)
    self.name = 'Commute'


class WindspeedEffect(AsymmGaussian):
  def __init__(self, data):
    t = data['windspeed'].values
    AsymmGaussian.__init__(self, t, lo=False, name='Windspeed effect')

  def beta0(self):
    return [35.0, 10.0]


class HumidityEffect(AsymmGaussian):
  def __init__(self, data):
    t = data['humidity'].values
    AsymmGaussian.__init__(self, t, lo=False, name='Humidity effect')

  def beta0(self):
    return [60.0, 30.0]


class TempEffect(AsymmGaussian):
  def __init__(self, data):
    t = data['temp'].values
    AsymmGaussian.__init__(self, t, name='Temperature effect')

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
    self.terms = [Amplitude, Quantity]
    self.model = ProductModel(train, self.terms)

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
    actual = self._train['count'].values
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
        'bounds': nbsm_train.bounds(),
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

cv()
sys.exit()














def popularity(data, beta):
  global days
  global day0

  res = numpy.zeros(len(data))

  for i, day in enumerate(days):
    res[(data['dayage'] - day0 == day).values] = beta[i]

  return res

def grad_popularity(coeffs, beta):
  global days
  global day0

  grad = numpy.zeros((len(train), len(beta)))

  for i, day in enumerate(days):
    grad[(train['dayage'] - day0 == day).values, i] = 1.0

  grad = coeffs.reshape(len(train), 1) * grad
  return grad


def count(data, beta):
  global rec_pattern_slc
  global commute_pattern_slc
  global rec_daytype_slc
  global commute_daytype_slc
  global weather_slc
  global pop_slc
  global temp_slc
  global mean_count

  cr = daily_pattern(data, beta[rec_pattern_slc])
  cc = daily_pattern(data, beta[commute_pattern_slc])
  #print 'daily pattern recreational', cr
  #print 'daily pattern commuters', cc

  gr = daytype_effect(data, beta[rec_daytype_slc])
  gc = daytype_effect(data, beta[commute_daytype_slc])
  #print 'daytype recreational', gr
  #print 'daytype commuters', gc

  f = weather_effect(data, beta[weather_slc])
  t = temp_effect(data['temp'].values, beta[temp_slc])
  #p = popularity(data, beta[pop_slc])
  p = pop_linear(data, beta[pop_slc])
  #print 'popularity', p
  #print 'weather', f

  pft = p * f * t
  q = gc * cc + gr * cr

  c = pft * q

  return mean_count * c


def error(predicted, actual):
  n = len(predicted)
  evec = numpy.log(1.0 + actual) - numpy.log(1.0 + predicted)
  e = numpy.sqrt(numpy.dot(evec, evec) / float(n))
  return e


def func2(beta):
  global rec_pattern_slc
  global commute_pattern_slc
  global rec_daytype_slc
  global commute_daytype_slc
  global weather_slc
  global pop_slc
  global mean_count

  param = 0
  n = len(train)

  cr = daily_pattern(train, beta[rec_pattern_slc])
  cc = daily_pattern(train, beta[commute_pattern_slc])
  #print 'daily pattern recreational', cr
  #print 'daily pattern commuters', cc

  gr = daytype_effect(train, beta[rec_daytype_slc])
  gc = daytype_effect(train, beta[commute_daytype_slc])
  #print 'daytype recreational', gr
  #print 'daytype commuters', gc

  f = weather_effect(train, beta[weather_slc])
  t = temp_effect(train['temp'].values, beta[temp_slc])
  #p = popularity(train, beta[pop_slc])
  p = pop_linear(train, beta[pop_slc])
  #print 'popularity', p
  #print 'weather', f

  pft = p * f * t
  q = gc * cc + gr * cr

  c = mean_count * pft * q
  grad = numpy.zeros((len(train), params_length))

  grad[:,rec_pattern_slc] = grad_pattern(pft * gr, beta[rec_pattern_slc])
  grad[:,commute_pattern_slc] = grad_pattern(pft * gc, beta[commute_pattern_slc])
  grad[:,rec_daytype_slc] = grad_daytype(pft * cr, beta[rec_daytype_slc])
  grad[:,commute_daytype_slc] = grad_daytype(pft * cc, beta[commute_daytype_slc])
  grad[:,weather_slc] = grad_weather(p * t * q, beta[weather_slc])
  grad[:,temp_slc] = grad_temp_effect(train['temp'].values, beta[temp_slc], p * f * q)
  #grad[:,pop_slc] = grad_popularity(f * t * q, beta[pop_slc])
  grad[:,pop_slc] = grad_pop_linear(f * t * q, beta[pop_slc])
  grad *= mean_count

  evec = numpy.log(1.0 + train['count'].values) - numpy.log(1.0 + c)
  e = numpy.sqrt(numpy.dot(evec, evec) / float(n))
  #print grad.shape
  #for i in range(24):
  #  print grad[i]
  #print c
  #print train[['hour', 'weather', 'workingday', 'holiday']]
  #print evec / (1.0 + c)
  #print (evec / (1.0 + c)).shape

  grad = -numpy.dot(evec / (1.0 + c), grad) / (e * float(n))

  #print 'beta:', beta
  #print 'error:', e, 'grad:', grad
  #print 'error:', e
  if numpy.isnan(e):
    grad[:] = 0.0
    return 10.0, grad
    print 'Got a NaN!!!'
    print 'popularity extremum:', beta[pop_slc]
    print c
    sys.exit()
  return e, grad



beta0 = numpy.ones(params_length)
beta0[rec_pattern_slc] /= 24.0
beta0[commute_pattern_slc] /= 24.0
# Following parameters from last basinhopping run.
beta0[weather_slc] = [1.0, 0.76, 0.7]
beta0[rec_daytype_slc] = [0.1, 0.71, 0.53] #low on workdays, high on others
beta0[commute_daytype_slc] = [1.0, 0.0, 0.155] #high on workdays, low on others
beta0[temp_slc] = [30.0, 15.0, 40.0] #ideal, spread_lo, spread_hi
#beta0[pop_slc] = [0.03469893, 0.95632921] #linear
beta0[pop_slc] = 1.0

bounds = [(None, None)] * params_length
bounds[rec_pattern_slc] = [(0.0, None)] * slen(rec_pattern_slc)
bounds[commute_pattern_slc] = [(0.0, None)] * slen(commute_pattern_slc)
bounds[weather_slc] = [(0.0, 1.0)] * slen(weather_slc)
bounds[rec_daytype_slc] = [(0.0, 1.0)] * slen(rec_daytype_slc)
bounds[commute_daytype_slc] = [(0.0, 1.0)] * slen(commute_daytype_slc)
bounds[temp_slc] = [(0.0, None)] * slen(temp_slc)
bounds[pop_slc] = [(0.0, None)] * slen(pop_slc)

# Constraints
def rec_pattern_unity(beta):
  global rec_pattern_slc
  return 1.0 - sum(beta[rec_pattern_slc])

def commute_pattern_unity(beta):
  global commute_pattern_slc
  return 1.0 - sum(beta[commute_pattern_slc])

res = optimize.minimize(func2, beta0, method='L-BFGS-B', jac=True, bounds=bounds, options={'disp': True, 'maxiter': 500})
eqcons = [rec_pattern_unity, commute_pattern_unity]
constraints = [
  {'type': 'eq', 'fun': rec_pattern_unity},
  {'type': 'eq', 'fun': commute_pattern_unity}
]

#res = optimize.minimize(func2, beta0, method='SLSQP', jac=True, constraints=constraints, bounds=bounds, tol=1.0e-12, options={'disp': 2, 'maxiter': 1000, 'iprint': 2})
#res = optimize.basinhopping(func2, beta0, minimizer_kwargs={'method':'L-BFGS-B', 'jac':True, 'bounds':bounds, 'options': {'disp': True, 'maxiter': 500}}, disp=True, niter=20)
#res = optimize.basinhopping(func2, beta0, minimizer_kwargs={'method':'SLSQP', 'jac':True, 'constraints':constraints, 'bounds':bounds, 'options':{'maxiter':1000, 'iprint':2, 'disp':2}}, disp=True)
print res

print 'rec pattern:', res.x[rec_pattern_slc]
print 'commute pattern:', res.x[commute_pattern_slc]
print 'rec daytype effect:', res.x[rec_daytype_slc]
print 'commute daytype effect:', res.x[commute_daytype_slc]
print 'weather effect:', res.x[weather_slc]
print 'temp effect:'
print '  ideal T:', res.x[temp_slc][0]
print '  lower dT:', res.x[temp_slc][1]
print '  upper dT:', res.x[temp_slc][2]
print 'popularity:', res.x[pop_slc]

print
print 'rec pattern sum:', sum(res.x[rec_pattern_slc])
print 'commute pattern sum:', sum(res.x[commute_pattern_slc])

#print 'Contest error:', error(train['count'].values, count(train, res.x))

plt.plot_date(train['dates'], train['count'])
plt.plot_date(train['dates'], count(train, res.x))

#plt.plot(res.x[:48])
plt.show()

plt.clf()
#plt.plot(range(len(res.x[pop_slc])), res.x[pop_slc], 'o')

plt.scatter(train['temp'], temp_effect(train['temp'].values, res.x[temp_slc]))
plt.show()
