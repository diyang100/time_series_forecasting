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



# format: month,year,number
df = pd.read_csv('inputdata.csv')

# df['date'] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
# df['month_year'] = df['date'].dt.to_period('M')

years = set()
for val in df['year']:
    if val not in years:
        years.add(val)

last_record = df.tail(1)
last_month, last_year = last_record.iloc[0]['month'], last_record.iloc[0]['year']

df = df.set_index(['year', 'month'])

print(df.head())

ax = df[['total']].plot(title='Total income per month (all years)', marker='o', color='b')
for idx, row in df.iterrows():
    ax.annotate(idx, (idx, row['total']) )

# plt.figure(1)
# plt.plot(ridge_grad_desc_10[2], ridge_grad_desc_10[3], label='Training Loss per Iteration')
# plt.xlabel('iterations')
# plt.ylabel('training loss')
# plt.title('Training Loss per Iteration for Gradient Descent with Lambda = 10')
# plt.legend()
# plt.show()

df_list = [d for _, d in df.groupby(['year'])]
for df in df_list:
    ax = df[['total']].plot(title='Total income per month', marker='o', color='b')
    ax.set_xticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.index)

plt.show()