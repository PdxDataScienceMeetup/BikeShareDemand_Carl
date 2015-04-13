import numpy
import pandas
import pylab
import datetime
import sys
from sklearn import linear_model


# There are vivid peaks at morning and evening rush hour in the later
# months. Afternoon rush tends to be more spread out, people leaving work
# varying times, but everyone going in together?

# Counts oscillate with a day period, amplitude grows as the service
# becomes popular.

# Registered users seem to match total count more often on workdays.
# 



date_format = '%Y-%m-%d %H:%M:%S'
epoch = datetime.datetime.utcfromtimestamp(0)

def load(filename):
  data = pandas.read_csv(filename)

  dates = [datetime.datetime.strptime(s, date_format) for s in data['datetime']]
  hours = [date.hour for date in dates]
  months = [date.strftime('%Y-%m') for date in dates]
  days = [date.strftime('%Y-%m-%d') for date in dates]
  
  data['dates'] = dates
  data['hour'] = hours
  data['day'] = days
  data['month'] = months
  data['monthday'] = [date.day for date in dates]
  data['dayage'] = [(date - epoch).days for date in dates]
  data['weekage'] = data['dayage'].astype('int').values / 7

  return data


def plot_stuff(data):
  working = data[data['workingday'] == 1]
  holiday = data[data['workingday'] == 0]
  
  working_day_avgs = working.groupby('day').aggregate(numpy.mean)
  working_day_avgs['dates'] = working_day_avgs.index.map(lambda day: datetime.datetime.strptime(day, '%Y-%m-%d'))
  holiday_day_avgs = holiday.groupby('day').aggregate(numpy.mean)
  holiday_day_avgs['dates'] = holiday_day_avgs.index.map(lambda day: datetime.datetime.strptime(day, '%Y-%m-%d'))
  
  pylab.plot_date(working_day_avgs['dates'], working_day_avgs['count'])
  pylab.plot_date(holiday_day_avgs['dates'], holiday_day_avgs['count'])
  pylab.show()
  
  pylab.plot_date(working['dates'], working.groupby('day').aggregate(numpy.mean)['count'])
  pylab.plot(data.groupby('day').aggregate(numpy.mean)['count'])
  pylab.plot(data.groupby('day').aggregate(numpy.mean)['atemp'])
  pylab.plot(data.groupby('day').aggregate(numpy.mean)['windspeed'])
  pylab.show()


def unity(*args):
  return 1.0

class Regression:
  """
  We fit to two different average daily shapes, for workdays and holidays, and
  scale them from day to day based on the mean count. For prediction, we
  take the last M days and linearly extrapolate the mean user count as the
  scaling factor for the daily shapes.
  """

  def __init__(self, multiplier=None, inv=None, trend_period=10, trend_degree=1, fit_intercept=True):
    self.multiplier = multiplier
    self.inv = inv
    if multiplier is None:
      self.multiplier = unity
      self.inv = unity
    self.epoch = datetime.datetime.utcfromtimestamp(0)
    self.trend_period = trend_period
    self.trend_degree = trend_degree
    self.fit_intercept = fit_intercept

  def predictors(self, data):
    """
    A predictor for a given time is simply the mean count for that day
    placed in the correct hour slot.

    Arguments:
      - data (DataFrame): Should be equipped with a 'popularity' column.
    """

    predictors = pandas.DataFrame()
    
    data['daytype'] = data['workingday']
    data.loc[data['workingday'] == 1, 'daytype'] = 'w'
    data.loc[data['workingday'] == 0, 'daytype'] = 'h'

#    for day_type in ['workday', 'holiday']:
#      for hour in range(24):
#        for ride_type in ['commute', 'fun']:
#          for weather in range(1, 5):
#            col = '%s-h%.2d-%s-weather%d' % (day_type, hour, ride_type, weather)
#            predictors[col] = data['popularity'] * self.inv(data) * ((data['hour'] == hour) & (data['daytype'] == day_type))
#

    for day_type in ['w', 'h']:
      for hour in range(24):
        col = '%s%.2d' % (day_type, hour)
        predictors[col] = data['popularity'] * self.inv(data) * ((data['hour'] == hour) & (data['daytype'] == day_type))

    # Throw in some continuous temperature data
    #predictors['atemp'] = data['atemp']
    #predictors['humidity'] = data['humidity'].astype('float64')
    #predictors['weather'] = data['weather']**pandas.rolling_mean(data['weather'], 24)
    #null = pandas.isnull(predictors['weather'])
    #predictors.loc[null, 'weather'] = data.loc[null, 'weather']

    return predictors

  def train(self, data, show=False):
    self.data = data

    # To measure the effect a really bad weather day has on ridership,
    # take all weather-4 hours, and compare them to weather-1 hours for
    # the same time of day.

    #w1 = data[(data['workingday'] == 1) & (data['weather'] == 1)].groupby(['month', 'hour'])
    #w2 = data[(data['workingday'] == 1) & (data['weather'] == 2)].groupby(['month', 'hour'])
    #w3 = data[(data['workingday'] == 1) & (data['weather'] == 3)].groupby(['month', 'hour'])
    #w4 = data[(data['workingday'] == 1) & (data['weather'] == 4)].groupby(['month', 'hour'])
    #print w1.mean()
    #print w2.mean()
    #print w3['count'].mean()
    #print w4['count'].mean()
    #print w4.mean() / w1.mean()
    #sys.exit()

    data['popularity'] = self.multiplier(data) * data['count']

    by_day = data.groupby('day')
    for day, dayta in by_day['popularity']:
      data.loc[dayta.index, 'popularity'] = dayta.mean()

    #print data['popularity']
    predictors = self.predictors(data)

    self.lr = linear_model.LinearRegression(fit_intercept=self.fit_intercept)
    #self.lr.fit(predictors.values, numpy.log(1.0 + data[['registered', 'count']]))
    #self.lr.fit(predictors.values, data[['registered', 'count']])

    #self.lr = linear_model.ARDRegression(fit_intercept=self.fit_intercept)
    self.lr.fit(predictors.values, data['count'])

    if show:
      res = self.lr.predict(predictors.values)
      pylab.plot_date(data['dates'], res[:,0])
      pylab.plot_date(data['dates'], res[:,1])
      pylab.show()

    return self

  def calculate_trend(self, month):
    """
    Calculate the trend in popularity for the last days
    of the training data in the given month.

    Returns:
      An interpolating polynomial. Feed it days since the epoch, and it will
      return a popularity score for each day.
    """

    # Check out historical ratio of holiday popularity to surrounding
    # workday popularity
    #working = self.data[(self.data['month'] <= month) & (self.data['workingday'] == 1)].groupby('weekage')
    #holiday = self.data[(self.data['month'] <= month) & (self.data['workingday'] == 0)].groupby('weekage')
    #print working['count'].mean()
    #print holiday['count'].mean()
    #self.ratio = working['count'].mean() / holiday['count'].mean()
    #self.ratio = working.mean() / holiday.mean()
    #ratio = (working['count'].mean() / holiday['count'].mean()).mean()
    #print ratio
    #sys.exit()
    #ratio = working.iloc[-nw:]['count'].mean() / holiday.iloc[-nh:]['count'].mean()
    #ratio = working['count'].mean() / holiday['count'].mean()
    #print ratio

    #month_data = self.data[self.data['month'] == month]
    history = self.data[self.data['month'] <= month].copy()
    num_months = len(history.groupby('month'))
    # Use only working days to assess popularity.
    #month_data = self.data[(self.data['month'] == month) & (self.data['workingday'] == 1)]
    history['adjusted_count'] = self.multiplier(history) * history['count']

    # With weather adjustment
    avgs = history.groupby('day')['adjusted_count'].mean()[-self.trend_period:] #.apply(lambda dayta: (self.multiplier(dayta) * dayta['count']).mean())
    avgs.index = avgs.index.map(lambda day: (datetime.datetime.strptime(day, '%Y-%m-%d') - self.epoch).days)
    #print avgs
    #print min(num_months, self.trend_degree)

    #avgs = month_data.groupby('day').aggregate(numpy.mean)['count'][-self.trend_period:]
    #avgs.index = avgs.index.map(lambda day: (datetime.datetime.strptime(day, '%Y-%m-%d') - self.epoch).days)
    #weights = numpy.concatenate([numpy.linspace(0.5, 1, len(avgs) / 2), numpy.ones(len(avgs) - len(avgs) / 2)])
    weights = numpy.ones(len(avgs))

    coeffs = numpy.polyfit(avgs.index, avgs.values, min(num_months, self.trend_degree), w=weights)
    #print 'Trend:', avgs, coeffs
    return numpy.poly1d(coeffs)

  def predict_popularity(self, test):
    """
    First identify contiguous ranges of dates. For each contiguous range, extrapolate
    popularity from previous days.

    Returns the full dataset with the 'popularity' column added.
    """

    slices = []

    for month, test_slc in test.groupby('month'):
      fit = self.calculate_trend(month)
      epoch_days = test_slc['day'].map(lambda day: (datetime.datetime.strptime(day, '%Y-%m-%d') - self.epoch).days)
      slices.append(pandas.Series(fit(epoch_days), index=test_slc.index))

    return pandas.concat(slices)

  def predict(self, test):
    """
    Extrapolate latest popularity over requested timeframe and
    get result from linear regressor.

    Arguments:
      X: should be a DataFrame with a 'dates' column.
    """

    test['popularity'] = self.predict_popularity(test)
    predictors = self.predictors(test)
    #result = pandas.DataFrame(self.lr.predict(predictors.values), columns=['registered', 'count'])
    result = pandas.DataFrame(self.lr.predict(predictors.values), columns=['count'])
    #result = numpy.exp(result) - 1.0
    #result.loc[result['registered'] < 0, 'registered'] = 0.0
    result.loc[result['count'] < 0, 'count'] = 0.0
    return result


W1 = [1.0, 1.0, 2.5, 4.0]

def mult1(data):
  """
  Return a weather coefficient that multiplies the count based on the weather.
  """
  multiplier = pandas.Series(numpy.ones(len(data)), index=data.index)
  multiplier[data['weather'] == 1] = W1[0]
  multiplier[data['weather'] == 2] = W1[1]
  multiplier[data['weather'] == 3] = W1[2]
  multiplier[data['weather'] == 4] = W1[3]
  return pandas.ewma(multiplier, span=6)

def inv1(data):
  return 1.0 / mult1(data)


def validate():
  data = load('train.csv')

  # Slice up training data; take only first n days of each month
  n = 10
  data = data[data['month'] < '2011-05']
  train = data[data['monthday'] < n].copy()
  test = data[data['monthday'] >= n].copy()
  print train, test

  reg = Regression(multiplier=mult1, inv=inv1, trend_period=60, trend_degree=3, fit_intercept=False)
  reg.train(train, show=False)
  res = reg.predict(test)

  #pylab.plot(reg.ratio['count'])
  #pylab.show()
  #pylab.clf()

  #print test['count']
  #print res['count']
  logres = numpy.log(1.0 + res['count'].values)
  logtest = numpy.log(1.0 + test['count'].values)

  print 'Test has NaN?', numpy.any(numpy.isnan(logtest))
  print 'Result has NaN?', numpy.any(numpy.isnan(logres))

  logdiff = logres - logtest
  print 'Score:', numpy.sqrt(numpy.dot(logdiff, logdiff) / float(res.size))

  pylab.plot_date(test['dates'], res['count'], label='predicted')
  pylab.plot_date(test['dates'], test['count'], label='true')
  pylab.plot_date(train['dates'], train['count'], label='training')
  pylab.plot(data['dates'], 125*data['weather'], label='weather')
  pylab.legend()
  pylab.show()


#reg.train(data[:24*10].copy(), show=False)
#res = reg.predict(data[24*10:24*20], show=True)

#plot_stuff(train)

def compete():
  train = load('train.csv')
  test = load('test.csv')

  #print train.iloc[:5]
  #sys.exit()

  #reg = Regression(multiplier=mult1, inv=inv1, trend_period=20, trend_degree=1, fit_intercept=False)
  reg = Regression(multiplier=mult1, inv=inv1, trend_period=40, trend_degree=1, fit_intercept=False)
  reg.train(train)
  res = reg.predict(test)
  
  dates = train['dates']

  pylab.plot_date(dates, train['count'])
  pylab.plot_date(test['dates'], res['count'])
  #pylab.plot(dates, 100 * train['holiday'])
  #pylab.plot(dates, 100 * train['workingday'])
  #pylab.plot(dates, 100 * train['weather'])
  pylab.plot_date(test['dates'], 100 * test['weather'])
  pylab.show()

  #submission = pandas.DataFrame()
  #submission['datetime'] = test['dates']
  #submission['count'] = res['count'] #.round().astype('int')
  #submission.to_csv('submission-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.csv', index=False)
  #print res


#validate()
compete()
