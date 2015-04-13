import numpy
from load import load
from scipy import optimize
from matplotlib import pyplot as plt
import sys

train = load('train.csv')
#train = train[train['month'] == '2011-01']
train['normcount'] = train['count'] / train['count'].mean()
mean_count = train['count'].mean()


# Weekend and holiday data grouped by day
byday = train[train['workingday'] == 0].groupby('dayage')

# 
normalize = lambda g: g / g.mean()
normed_byday = byday.transform(normalize)
print normed_byday #['count'].mean()
sys.exit()


plt.scatter(byday['weather'].mean(), byday['count'].mean())
plt.show()
sys.exit()







def kernel(xi, Xj, beta):
  """
  The kernel measures the "proximity" of one input vector to another,
  given the parameters of the model in beta. The proximity is used to
  correlate input vectors with counts

  The model looks for correlations between:
    - Current sample and

  Arguments:
    xi: (Series)
    Xj: (DataFrame)
    beta: (Series)
      The parameters for the model

        [ comb_history, comb_amplitude ]

  x is:
    [ time (seconds), hour (1-24), weather (1-4) ]
  """

  comb_history = beta[0]
  comb_amplitude = beta[1]

  comb_samples = Xj[Xj['hour'] == xi['hour']].iloc[-int(comb_history):]
  last_day = float(comb_samples.iloc[-1]['dayage'])
  comb_weights = comb_amplitude * (oldest_day - comb_samples['dayage']) / comb_history

  return comb_weights.sum()

  #time_alpha, hour_alpha, weather_alpha = beta
  time_alpha, = beta
  #t1, h1, w1 = x1
  #t2, h2, w2 = x2
  res = numpy.exp(-time_alpha**2 * (xi['age_seconds'] - Xj['age_seconds'].values)**2)
  if numpy.sum(res) == 0.0:
    #print time_alpha, xi['age_seconds'] - Xj['age_seconds']
    print time_alpha, time_alpha**2 * (xi['age_seconds'] - Xj['age_seconds'].values)**2
    sys.exit()
  return res




day0 = train.iloc[0]['dayage']
days = [(day - day0) for (day, group) in train.groupby('dayage')]
print len(days), 'days in training dataset'

rec_pattern_slc = slice(0, 24)
commute_pattern_slc = slice(24, 48)
rec_daytype_slc = slice(48, 51)
commute_daytype_slc = slice(51, 54)
weather_slc = slice(54, 57)
pop_slc = slice(57, 57 + len(days))
params_length = 57 + len(days)

def slen(slc):
  return slc.stop - slc.start



def weather_effect(data, beta):
  weather = data['weather']
  res = numpy.ones(len(weather))

  res[(weather == 2).values] = beta[0]
  res[(weather == 3).values] = beta[1]
  res[(weather == 4).values] = beta[2]

  return res


def grad_weather(coeffs, beta):
  weather = train['weather']
  grad = numpy.zeros((len(train), len(beta)))
  grad[(weather == 2).values, 0] = 1.0
  grad[(weather == 3).values, 1] = 1.0
  grad[(weather == 4).values, 2] = 1.0

  return coeffs.reshape(len(train), 1) * grad


def daytype_effect(data, beta):
  res = numpy.zeros(len(data))
  res[(data['workingday'] == 1).values] = beta[0]
  res[((data['workingday'] == 0) & (data['holiday'] == 0)).values] = beta[1] #weekend
  res[(data['holiday'] == 1).values] = beta[2] #holiday

  return res


def grad_daytype(coeffs, beta):
  grad = numpy.zeros((len(train), len(beta)))
  grad[(train['workingday'] == 1).values, 0] = 1.0
  grad[((train['workingday'] == 0) & (train['holiday'] == 0)).values, 1] = 1.0
  grad[(train['holiday'] == 1).values, 2] = 1.0

  grad = coeffs.reshape(len(train), 1) * grad
  return grad


def daily_pattern(data, beta):
  res = numpy.zeros(len(data))
  weights = beta #/ sum(beta)

  for i, weight in enumerate(weights):
    res[(data['hour'] == i).values] = weight

  return res


def grad_pattern(coeffs, beta):
  grad = numpy.zeros((len(train), len(beta)))
  norm = sum(beta)
  inv_norm = 1.0 / norm
  beta_over_norm2 = beta / (norm * norm)

  for hour in xrange(24):
    #grad[(train['hour'] == hour).values] = -beta_over_norm2[hour]
    #grad[(train['hour'] == hour).values, hour] += inv_norm
    grad[(train['hour'] == hour).values, hour] = 1.0

  return coeffs.reshape(len(train), 1) * grad



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
  p = popularity(data, beta[pop_slc])
  #print 'popularity', p
  #print 'weather', f

  pf = p * f
  q = gc * cc + gr * cr

  c = pf * q

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
  p = popularity(train, beta[pop_slc])
  #print 'popularity', p
  #print 'weather', f

  pf = p * f
  q = gc * cc + gr * cr

  c = mean_count * pf * q
  grad = numpy.zeros((len(train), params_length))

  grad[:,rec_pattern_slc] = grad_pattern(pf * gr, beta[rec_pattern_slc])
  grad[:,commute_pattern_slc] = grad_pattern(pf * gc, beta[commute_pattern_slc])
  grad[:,rec_daytype_slc] = grad_daytype(pf * cr, beta[rec_daytype_slc])
  grad[:,commute_daytype_slc] = grad_daytype(pf * cc, beta[commute_daytype_slc])
  grad[:,weather_slc] = grad_weather(p * q, beta[weather_slc])
  grad[:,pop_slc] = grad_popularity(f * q, beta[pop_slc])
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
beta0[pop_slc] = 1.0

bounds = [(None, None)] * params_length
bounds[rec_pattern_slc] = [(0.0, None)] * slen(rec_pattern_slc)
bounds[commute_pattern_slc] = [(0.0, None)] * slen(commute_pattern_slc)
bounds[weather_slc] = [(0.0, 1.0)] * slen(weather_slc)
bounds[rec_daytype_slc] = [(0.0, 1.0)] * slen(rec_daytype_slc)
bounds[commute_daytype_slc] = [(0.0, 1.0)] * slen(commute_daytype_slc)
bounds[pop_slc] = [(0.0, None)] * slen(pop_slc)

# Constraints
def rec_pattern_unity(beta):
  global rec_pattern_slc
  return 1.0 - sum(beta[rec_pattern_slc])

def commute_pattern_unity(beta):
  global commute_pattern_slc
  return 1.0 - sum(beta[commute_pattern_slc])


#res = optimize.minimize(func2, beta0, method='L-BFGS-B', jac=True, bounds=bounds, options={'disp': True, 'maxiter': 100})
eqcons = [rec_pattern_unity, commute_pattern_unity]
constraints = [
  {'type': 'eq', 'fun': rec_pattern_unity},
  {'type': 'eq', 'fun': commute_pattern_unity}
]

res = optimize.minimize(func2, beta0, method='SLSQP', jac=True, constraints=constraints, bounds=bounds, options={'disp': 2, 'maxiter': 200, 'iprint': 2})
#res = optimize.basinhopping(func2, beta0, minimizer_kwargs={'method':'L-BFGS-B', 'jac':True, 'bounds':bounds}, disp=True)
#res = optimize.basinhopping(func2, beta0, minimizer_kwargs={'method':'SLSQP', 'jac':True, 'constraints':constraints, 'bounds':bounds, 'options':{'maxiter':200, 'iprint':2, 'disp':2}}, disp=True)
print res

print 'rec pattern:', res.x[rec_pattern_slc]
print 'commute pattern:', res.x[commute_pattern_slc]
print 'rec daytype effect:', res.x[rec_daytype_slc]
print 'commute daytype effect:', res.x[commute_daytype_slc]
print 'weather effect:', res.x[weather_slc]
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
plt.plot(range(len(res.x[pop_slc])), res.x[pop_slc], 'o')

plt.show()

sys.exit()

plot_num = 0

def func(beta):
  """
  The cost function. For every sample, it feeds it and all previous samples to
  the kernel function, along with the kernel parameters.

  Model:
    Superposition of two daily ridership curves: work and recreational. Each curve is
    subdued by the same weather factor.

      W * ( a(workday) * w(h, d) + b(workday) * r(h, d) )

  Arguments:
    beta:
      kernel parameters.

  Returns:
    A number measuring how well the parameters of the model predict the known
    training samples.
  """

  global plot_num

  length = len(train) - 1
  f = numpy.zeros(len(train) - 1)
  ws = []
  ys = []
  xis = []
  actual = numpy.zeros(len(train))
  predicted = numpy.zeros(len(train))
  n = 0

  comb_history = beta[0]      # days in history that affect prediction
  comb_amplitude = 1.0        # scale
  weather1_effect = beta[1]**2
  weather2_effect = beta[2]**2
  weather3_effect = beta[3]**2
  #work_effect = beta[5]

  for i in train.index[1:]:
    xi = train.iloc[i]
    Xj = train.iloc[:i]

    comb_samples = Xj[(Xj['hour'] == xi['hour']) & (xi['dayage'] - Xj['dayage'] < comb_history)]

    if len(comb_samples) == 0:
      continue

    comb_weights = comb_amplitude * (1.0 - (xi['dayage'] - comb_samples['dayage']) / comb_history)

    #if numpy.any(comb_weights.values < 0):
    #  raise ValueError('Weight < 0')

    # workingday effect. Correlation between
    #comb_weights += work_effect * numpy.abs(comb_samples['workingday'] - xi['workingday'])

    # Approximate weather effect. Generally, weight more heavily the previous days that have
    # a matching weather pattern.
    weather_diff = xi['weather'] - comb_samples['weather']
    comb_weights[weather_diff == -3] *= weather3_effect
    comb_weights[weather_diff == -2] *= weather2_effect
    comb_weights[weather_diff == -1] *= weather1_effect
    comb_weights[weather_diff == 1] /= weather1_effect
    comb_weights[weather_diff == 2] /= weather2_effect
    comb_weights[weather_diff == 3] /= weather3_effect

    weights = comb_weights.values
    y = comb_samples['count'].values
    #numpy.dot(comb_samples['count'].values, weights) / numpy.sum(weights)

    #print 'count', y.values, 'inputs', xi.values
    #weights = kernel(xi, train.iloc[:i], beta)
    #ws.append(weights)
    #print 'weights', weights
    #f[i - 1] = numpy.dot(y, weights) / numpy.sum(weights)
    n += 1
    actual[i] = xi['count']

    norm = numpy.sum(weights)
    if norm <= 0.0:
      predicted[i] = numpy.sum(y) / float(len(y))
    else:
      predicted[i] = numpy.dot(y, weights) / numpy.sum(weights)
    #logdiff[i - 1] = numpy.dot(y, weights) / numpy.sum(weights)
    #print f[i-1]

  if plot_num < 3:
    plot_num += 1
    plt.plot_date(train['dates'], actual)
    plt.plot_date(train['dates'], predicted)
    plt.show()

  logy = numpy.log(1.0 + actual) #train.iloc[1:]['count'].values)
  logf = numpy.log(1.0 + predicted)
  logdiff = logy - logf
  error = numpy.sqrt(numpy.dot(logdiff, logdiff) / float(n))
  print beta, 'gives', error
  if numpy.isnan(error):
    print f
    #print numpy.array(ws)[numpy.isnan(f)]
    #print numpy.array(ys)[numpy.isnan(f)]
    #print numpy.array(xis)[numpy.isnan(f)]

    #print logy
    #print logf
    #print logdiff
  return error


beta0 = [
  10.0, # comb_length,
  1.0,  # weather1_effect
  1.0,  # weather2_effect
  1.0   # weather3_effect
]

#res = optimize.fmin(func, beta0, xtol=1.0e-6, ftol=1.0e-3, maxiter=100, disp=True)

print res
