import pandas
import datetime

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

  # 0-7 inclusive. Includes special holiday day.
  data['dayoweek'] = [date.weekday() for date in dates]
  data.loc[data['holiday'] == 1, 'dayoweek'] = 7

  data['month'] = months
  data['monthage'] = [(12 * date.year + date.month - 1) for date in dates]
  data['monthday'] = [date.day for date in dates]
  data['dayage'] = [(date - epoch).days for date in dates]
  data['hourage'] = [24*(date - epoch).days + date.hour for date in dates]
  data['weekage'] = data['dayage'].astype('int').values / 7
  data['age_seconds'] = [(date - epoch).days * 24 * 3600. + (date - epoch).seconds for date in dates]

  return data
