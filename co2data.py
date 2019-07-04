#%%
import pymc3 as pm
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import datetime as dt
import numpy as np

py.init_notebook_mode(connected=True)

def dates_to_idx(timelist):
    reference_time = pd.to_datetime('1958-03-15')
    t = (timelist - reference_time) / pd.Timedelta(1, "Y")
    return np.asarray(t)


data_monthly = pd.read_csv(pm.get_data(
    "monthly_in_situ_co2_mlo.csv"), header=56)
data_monthly.replace(to_replace=-99.99, value=np.nan, inplace=True)


cols = ["year", "month", "--", "--", "CO2", "seasonaly_adjusted", "fit",
        "seasonally_adjusted_fit", "CO2_filled", "seasonally_adjusted_filled"]
data_monthly.columns = cols
cols.remove("--")
cols.remove("--")
data_monthly = data_monthly[cols]

data_monthly["day"] = 15
data_monthly.index = pd.to_datetime(data_monthly[["year", "month", "day"]])
cols.remove("year")
cols.remove("month")
data_monthly = data_monthly[cols]


t = dates_to_idx(data_monthly.index)

# normalize CO2 levels
y = data_monthly["CO2"].values
first_co2 = y[0]
std_co2 = np.std(y)
y_n = (y - first_co2) / std_co2

data_monthly = data_monthly.assign(t=t)
data_monthly = data_monthly.assign(y_n=y_n)

data_monthly = data_monthly.reset_index()
data_monthly = data_monthly.iloc[:, 0:3]

data_monthly['calc'] = data_monthly.iloc[:, 1]-data_monthly.iloc[:, 2]
data_monthly = data_monthly[['index', 'calc']]

data_monthly = data_monthly.dropna(how='any')
data_monthly['index']=dates_to_idx(data_monthly.iloc[:,0])

data_monthly.iloc[:, 0] = (
    data_monthly.iloc[:, 0]-np.mean(data_monthly.iloc[:, 0]))/np.std(data_monthly.iloc[:, 0])


data_early = data_monthly[data_monthly['index'] < 0.5]
data_later = data_monthly[data_monthly['index'] >= 0.5]



data_early.to_pickle('/home/work/work/Bayesian_yarin/data/data_early.pkl')
data_later.to_pickle('/home/work/work/Bayesian_yarin/data/data_later.pkl')



