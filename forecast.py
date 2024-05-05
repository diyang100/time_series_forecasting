import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# adjusts labels in plot to reduce overlappting
from adjustText import adjust_text

# format: month,year,number
df = pd.read_csv('inputdata.csv')

# create index column 
df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))

# determine which years of data is present
years = set()
for val in df['year']:
    if val not in years:
        years.add(val)

# determine last record
last_record = df.tail(1)
last_month, last_year = last_record.iloc[0]['month'], last_record.iloc[0]['year']

df = df.set_index(['date'])

print(df.head())

# decompose time series
decompose_result = seasonal_decompose(df['total'],model='multiplicative')
decompose_result.plot()

# fit the data with triple Holt-Winters Exponential Smoothing
fig, ax = plt.subplots(1, figsize=[15,5])
df['HWES3_ADD'] = ExponentialSmoothing(df['total'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
fitted_model = ExponentialSmoothing(df['total'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
df['HWES3_MUL'] = fitted_model.fittedvalues
df[['total','HWES3_ADD','HWES3_MUL']].plot(title='Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality', ax=ax)
ax.set_yticklabels([f"${t:0.0f}" for t in ax.get_yticks()])

# predict next 12 months with HWES3_MUL
test_predictions = fitted_model.forecast(12)
print("+++++++++++++++++++", type(test_predictions))

# plot all years data + forecast
fig, ax = plt.subplots(1, figsize=[15,5])
df['total'].plot(legend=True,label='TRAIN',ax=ax)
test_predictions.plot(legend=True,label='PREDICTION',ax=ax)
plt.title('Past and Forecasted Data using Holt Winters')
ax.set_yticklabels([f"${t:0.0f}" for t in ax.get_yticks()])

# plot only forecast
fig, ax = plt.subplots(1, figsize=[15,5])
test_predictions.plot(ax=ax)
plt.title('Forecasted Data using Holt Winters')
ax.set_yticklabels([f"${t:0.0f}" for t in ax.get_yticks()])

print(test_predictions)
totals = test_predictions.values
dates = test_predictions.index
texts = []
for i, y in enumerate(totals):
    x = dates[i]
    txt = str(int(y)//1000) + 'K'
    texts.append(plt.text(x, y, txt))

adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

# plot all years data
fig, ax = plt.subplots(1, figsize=[15,5])
df[['total']].plot(title='Total income per month (all years)', marker='o', color='b', ax=ax)
ax.set_yticklabels([f"${t:0.0f}" for t in ax.get_yticks()])

totals = df.loc[:,'total']
dates = df.index
texts = []
for i, y in enumerate(totals):
    x = dates[i]
    txt = str(int(y)//1000) + 'K'
    texts.append(plt.text(x, y, txt))

adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

# plot data by year
df_list = [d for _, d in df.groupby(['year'])]
for i, df in enumerate(df_list):
    fig, ax = plt.subplots(1, figsize=[15,5])
    df[['total']].plot(title='Total income per month', marker='o', color='b', ax=ax)
    ax.set_yticklabels([f"${t:0.0f}" for t in ax.get_yticks()])


plt.show()