# %%
import numpy as np
import pandas as pd
%matplotlib inline

df = pd.read_csv('C:/Users/30071747/Downloads/DailyDelhiClimateTrain.csv',index_col=0,parse_dates=False)
df.head(60)

# %% [markdown]
# As mentioned in above code, we have to import the python libraries,
# The code that we will call the submodule Numpy as "np", pandas as "pd"
# Numpy is used to make the arithmetic operations, Pandas is used to analyse the data.
# By analysing the data output,we get the accurate data of date and mean temperature.
# In the Graph plotted below, it represents the cyclic trend between the meanpressure and the date of 5 years, by the given data "Daily Delhi Climate Train.csv"

# %%
ax = df['meantemp'].plot(figsize=(15,10))
ax.autoscale(axis='x',tight=True)
ax.set(ylabel='meantemp');

# %%
from statsmodels.tsa.filters.hp_filter import hpfilter

#Tuple unpacking
mean_cycle,mean_trend = hpfilter(df['meantemp'],lamb=1600)

# %% [markdown]
# Here we use hpfilter to get a smooth curve, this filter is applied for the analysis to remove short term-fluctucations associated with the cycle, which helps in to get the results for the long term

# %%
mean_cycle

# %%
mean_trend.plot(figsize=(15,10))

# %%
df['trend']=mean_trend

# %%
df.head()

# %%
type(mean_cycle)

# %%
df['trend'] = mean_trend

# %%
df[['trend','meantemp']].plot(figsize=(15,10)).autoscale(axis='x',tight=True);

# %%
df[['trend','meantemp']]['01/01/2017':].plot(figsize=(12,8)).autoscale(axis='x',tight=True);

# %% [markdown]
#  The test is set 12 months,by using the unbiased way of assessing the model. Moving average is technical indicator used to determine the period of time, since we need to know the peroid of time for 12 months, we use the Simple moving average to know the value of 12 months mean temperature.

# %%
df['12-month-SMA']=df['meantemp'].rolling(window=12).mean()

# %%
df.head(15)

# %%
df.plot(figsize=(10,5))

# %%
df['EWMA12']=df['meantemp'].ewm(span=12,adjust=False).mean()

# %%
df[['meantemp','EWMA12']].plot(figsize=(15,10));

# %%
df[['meantemp','EWMA12','12-month-SMA']].plot(figsize=(12,8)).autoscale(axis='x',tight=True);

# %% [markdown]
# Holt Winters seasonal method which comprises the forecast the equation and makes smoothing equations, which is also used for linear exponential smoothing, which is also know for the smoothing model of forcasting data that has trend. Here we perform Holt Winter to get the smoothing curve.

# %% [markdown]
# Holt Winters 

# %%
import pandas as pd
import numpy as np
%matplotlib inline

# %%
df = pd.read_csv('C:/Users/30071747/Downloads/DailyDelhiClimateTrain.csv',index_col='date',parse_dates=False)

# %%
df.dropna(inplace=True)

# %%
df.index

# %%
df.head()

# %%
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 12
alpha = 2/(span+1)

df['EWMA12'] = df['meantemp'].ewm(alpha=alpha,adjust=False).mean()
df['SES12']=SimpleExpSmoothing(df['meantemp']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
df.head()

# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df['DESadd12'] = ExponentialSmoothing(df['meantemp'], trend='add').fit().fittedvalues.shift(-1)
df.head()

# %%
df[['meantemp','EWMA12','DESadd12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True)

# %% [markdown]
# Here in this Graph, we can see the difference between the EWMA and mean temperature. where there is change in the mean temperature through out the months, 

# %%
df['DESmul12'] = ExponentialSmoothing(df['meantemp'], trend='mul').fit().fittedvalues.shift(-1)
df.head()

# %%
df[['meantemp','EWMA12','DESadd12']].iloc[:24].plot(figsize=(12,6)).autoscale(axis='x',tight=True)

# %%
df['TESadd12'] = ExponentialSmoothing(df['meantemp'], trend='add').fit().fittedvalues.shift(-1)
df.head()

# %%
df[['meantemp','TESadd12',]].plot(figsize=(15,10)).autoscale(axis='x',tight=True);

# %%
df[['meantemp','TESadd12',]].iloc[:24].plot(figsize=(15,10)).autoscale(axis='x',tight=True)

# %%


# %%


# %% [markdown]
# Multivarient regression

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

# %%
data = pd.read_csv('C:/Users/30071747/Downloads/DailyDelhiClimateTrain.csv')

# %%
data

# %%
data.describe()

# %%
y = data['humidity']
x1 = data[['meantemp','humidity','wind_speed','meanpressure']]

# %%
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

# %%


# %% [markdown]
# MASE or MAE

# %%
# Calculating the MAE with a custom function
import numpy as np

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

true = [-4.719,2.533,1.82e+16,-2.196,-0.412]
predicted = [0.000,0.011,0.000,0.028,0.680]

print(mae(true, predicted))

# %%


# %%


# %%
import pandas as pd
import numpy as np
%matplotlib inline

# %%
df = pd.read_csv('C:/Users/30071747/Downloads/DailyDelhiClimateTrain.csv', index_col='date',parse_dates=True)

# %%
df.dropna(inplace=True)

# %%
df.head()

# %%
df.plot()

# %%
df[df.index.duplicated()]

# %%
df['meantemp'].plot()

# %%
import numpy as np
import pandas as pd
import seaborn as sns

# %%
data = pd.read_csv('C:/Users/30071747/Downloads/DailyDelhiClimateTrain.csv', index_col='country')

# %%
x_scaled = data.copy()
x_scaled = x_scaled.drop(['meantemp'],axis=1)

# %%
x_scaled

# %%



