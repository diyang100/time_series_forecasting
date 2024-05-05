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

from adjustText import adjust_text

# format: month,year,number
df = pd.read_csv('inputdata.csv')

df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
# df['month_year'] = df['date'].dt.to_period('M')

years = set()
for val in df['year']:
    if val not in years:
        years.add(val)

last_record = df.tail(1)
last_month, last_year = last_record.iloc[0]['month'], last_record.iloc[0]['year']

df = df.set_index(['date'])

print(df.head())


# plot all years data
fig, ax = plt.subplots(1, figsize=[15,5])
df[['total']].plot(title='Total income per month (all years)', marker='o', color='b', ax=ax)
ax.set_yticklabels([f"{t:0.0f}" for t in ax.get_yticks()])

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
    ax.set_yticklabels([f"{t:0.0f}" for t in ax.get_yticks()])


plt.show()