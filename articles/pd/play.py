import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata
from matplotlib.collections import LineCollection
import time
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, \
    is_bool_dtype
from sklearn.ensemble.partial_dependence import partial_dependence, \
    plot_partial_dependence
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from pdpbox import pdp
from rfpimp import *
from scipy.integrate import cumtrapz
from stratx.partdep import *
from stratx.ice import *
import inspect
import statsmodels.api as sm


def df_string_to_cat(df: pd.DataFrame) -> dict:
    catencoders = {}
    for colname in df.columns:
        if is_string_dtype(df[colname]) or is_object_dtype(df[colname]):
            df[colname] = df[colname].astype('category').cat.as_ordered()
            catencoders[colname] = df[colname].cat.categories
    return catencoders


def toy_weather_data():
    def temp(x): return np.sin((x + 365 / 2) * (2 * np.pi) / 365)

    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df = pd.DataFrame()
    df['dayofyear'] = range(1, 365 + 1)
    df['state'] = np.random.choice(['CA', 'CO', 'AZ', 'WA'], len(df))
    df['temperature'] = temp(df['dayofyear'])
    df.loc[df['state'] == 'CA', 'temperature'] = 70 + df.loc[
        df['state'] == 'CA', 'temperature'] * noise('CA')
    df.loc[df['state'] == 'CO', 'temperature'] = 40 + df.loc[
        df['state'] == 'CO', 'temperature'] * noise('CO')
    df.loc[df['state'] == 'AZ', 'temperature'] = 90 + df.loc[
        df['state'] == 'AZ', 'temperature'] * noise('AZ')
    df.loc[df['state'] == 'WA', 'temperature'] = 60 + df.loc[
        df['state'] == 'WA', 'temperature'] * noise('WA')
    return df

df_yr1 = toy_weather_data()
df_yr1['year'] = 1980
df_yr2 = toy_weather_data()
df_yr2['year'] = 1981
df_yr3 = toy_weather_data()
df_yr3['year'] = 1982
df_raw = pd.concat([df_yr1, df_yr2, df_yr3], axis=0)
df = df_raw.copy()
catencoders = df_string_to_cat(df_raw.copy())
# states = catencoders['state']
# print(states)
#
# df_cat_to_catcode(df)

names = {'CO': 5, 'CA': 10, 'AZ': 15, 'WA': 20}
df['state'] = df['state'].map(names)
catnames = OrderedDict()
for k,v in names.items():
    catnames[v] = k

X = df.drop('temperature', axis=1)
y = df['temperature']

# leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored_ = \
#     partial_dependence(X=X, y=y, colname='dayofyear',
#                        verbose=True)

# print(pdpx)
# print(pdpy)


plot_catstratpd(X, y, 'state', 'temperature', catnames=catnames,
                # min_samples_leaf=30,
                n_trials=5,
                min_y_shifted_to_zero=True,
                show_x_counts=False,
                alpha=.3,
                yrange=(-2, 60),
                )

plt.show()
