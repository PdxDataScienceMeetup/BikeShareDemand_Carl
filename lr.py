import numpy
import pandas
import pylab
import datetime
import sys
from sklearn import linear_model

date_format = '%Y-%m-%d %H:%M:%S'

train = numpy.genfromtxt('train.csv', delimiter=',')
data = pandas.read_csv('train.csv')

dates = [datetime.datetime.strptime(s, date_format) for s in data['datetime']]
hours = [date.hour for date in dates]
days = [date.day for date in dates]

data['hour'] = hours
data['day'] = days

pylab.plot_date(dates, data['count'])
pylab.plot_date(dates, data['registered'])
#pylab.plot_date(dates, data['atemp'])
#pylab.show()

# There are vivid peaks at morning and evening rush hour in the later
# months. Afternoon rush tends to be more spread out, people leaving work
# varying times, but everyone going in together?

# Counts oscillate with a day period, amplitude grows as the service
# becomes popular.

# Registered users seem to match total count more often on workdays.
# 

# Model function:
#
# count = pop(b1, ..., bk) * ( sin(T = day) + exp(sigma = 1hr) + exp(


def bump(hour, center, half_width):
  offset = hour - center
  if abs(offset) > half_width:
    return 0.0
  else:
    return 0.5 * numpy.cos(numpy.pi * offset / half_width) + 0.5


MORNING_HALF_WIDTH = 1.0 #hours
MORNING_RUSH_HOUR = 8.0 #AM

def morning_bump_fn(hour):
  return bump(hour, 8.0, 1.0)

def evening_bump_fn(hour):
  return bump(hour, 16.0, 2.0)

# The popularity will be the average of the previous day's counts
work_popularity = 0.0
work_counts = 0

holiday_popularity = 0.0
holiday_counts = 0


def workday_predictor(row, popularity):
  """
  The predictor
  """
  date = datetime.datetime.strptime(row['datetime'], date_format)
  return [
    popularity * (1.0 - numpy.cos(2.0 * numpy.pi * date.hour / 24.0)),
    popularity * morning_bump(date.hour),
    popularity * evening_bump(date.hour),
    row['atemp'],
    row['humidity']
  ]


def holiday_predictor(row, popularity):
  """
  Arguments:
    - row: A data sample.
    - popularity (float): Scales the bump features. Should reflect the day's
      popularity measure.
  """

  date = datetime.datetime.strptime(row['datetime'], date_format)
  return [
    popularity * (1.0 - numpy.cos(2.0 * numpy.pi * date.hour / 24.0)),
    0.0,
    0.0,
    row['atemp'],
    row['humidity']
  ]



morning_bump = pandas.Series(range(0, 24), dtype='float')
morning_bump.map(morning_bump_fn)

evening_bump = pandas.Series(range(0, 24), dtype='float')
evening_bump.map(evening_bump_fn)

day_bump = pandas.Series(range(0, 24), dtype='float')
day_bump.map(lambda hour: 1.0 - numpy.cos(2.0 * numpy.pi * hour / 24.0))

# Gather training data.

day = 0
date = None
hour = 0
predictors = None #pandas.DataFrame(columns=['day_bump', 'morning_bump', 'evening_bump'])
i = 0

while i < len(data):
  day_start = i
  current_day = data.loc[i, 'day']
  # Get day slice
  while i < len(data) and data.loc[i, 'day'] == current_day:
    i += 1
  day_end = i

  dayta = data[day_start:day_end]
  #print 'Day', day, 'data'
  #print dayta
  #print data.loc[day_start].datetime, 'to', data.loc[day_end].datetime
  popularity = dayta['count'].mean()

  day_predictors = pandas.DataFrame()
  day_predictors['day_bump'] = dayta['hour'].map(lambda hour: popularity * (1.0 - numpy.cos(2.0 * numpy.pi * hour / 24.0)))

  if dayta.loc[day_start, 'workingday'] == 1:
    day_predictors['morning_bump'] = dayta['hour'].map(lambda hour: popularity * morning_bump_fn(hour))
    day_predictors['evening_bump'] = dayta['hour'].map(lambda hour: popularity * evening_bump_fn(hour))
  else:
    day_predictors['morning_bump'] = numpy.zeros(len(dayta))
    day_predictors['evening_bump'] = numpy.zeros(len(dayta))

  day_predictors.index = dayta.index
  #print day_predictors
  if predictors is None:
    predictors = day_predictors
  else:
    predictors = predictors.append(day_predictors)

  day += 1

#print predictors

#pylab.plot_date(dates, predictors['day_bump'])
#pylab.plot_date(dates, predictors['morning_bump'])
#pylab.plot_date(dates, predictors['evening_bump'])


lr = linear_model.LinearRegression()
lr.fit(predictors.values, data[['registered', 'count']])
res = lr.predict(predictors.values)

print res.shape

pylab.plot_date(dates, res[:,0])
pylab.plot_date(dates, res[:,1])
pylab.show()

#numpy.linalg.svd()
