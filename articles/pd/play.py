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

def weather():
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
                    n_trials=10,
                    min_y_shifted_to_zero=True,
                    show_x_counts=False,
                    alpha=.3,
                    yrange=(-2, 60),
                    figsize=(2.1,2.5)
                    )

    plt.show()

def bigX_data(n):
    x1 = np.random.uniform(-1, 1, size=n)
    x2 = np.random.uniform(-1, 1, size=n)
    x3 = np.random.uniform(-1, 1, size=n)

    y = 0.2 * x1 - 5 * x2 + 10 * x2 * np.where(x3 >= 0, 1, 0) + np.random.normal(0, 1,
                                                                                 size=n)
    df = pd.DataFrame()
    df['x1'] = x1
    df['x2'] = x2
    df['x3'] = x3
    df['y'] = y
    return df

def bigX():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    df = bigX_data(n=n)
    X = df.drop('y', axis=1)
    y = df['y']

    # plot_stratpd_gridsearch(X, y, 'x2', 'y',
    #                         min_samples_leaf_values=[2,5,10,20,30],
    #                         #                            nbins_values=[1,3,5,6,10],
    #                         yrange=(-4,4))
    #
    # plt.tight_layout()
    # plt.show()
    # return

    # Partial deriv is just 0.2 so this is correct. flat deriv curve, net effect line at slope .2
    # ICE is way too shallow and not line at n=1000 even
    fig, axes = plt.subplots(2, 2, figsize=(4, 4), sharey=True)

    # Partial deriv wrt x2 is -5 plus 10 about half the time so about 0
    # Should not expect a criss-cross like ICE since deriv of 1_x3>=0 is 0 everywhere
    # wrt to any x, even x3. x2 *is* affecting y BUT the net effect at any spot
    # is what we care about and that's 0. Just because marginal x2 vs y shows non-
    # random plot doesn't mean that x2's net effect is nonzero. We are trying to
    # strip away x1/x3's effect upon y. When we do, x2 has no effect on y.
    # Ask what is net effect at every x2? 0.
    plot_stratpd(X, y, 'x2', 'y', ax=axes[0, 0], yrange=(-4, 4),
                 show_slope_lines=True,
                 n_trials=1,
                 min_samples_leaf=20,
                 pdp_marker_size=2)

    # Partial deriv wrt x3 of 1_x3>=0 is 0 everywhere so result must be 0
    plot_stratpd(X, y, 'x3', 'y', ax=axes[1, 0], yrange=(-4, 4),
                 show_slope_lines=True,
                 n_trials=1,
                 min_samples_leaf=20,
                 pdp_marker_size=2)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = predict_ice(rf, X, 'x2', 'y', numx=100)
    plot_ice(ice, 'x2', 'y', ax=axes[0, 1], yrange=(-4, 4))

    ice = predict_ice(rf, X, 'x3', 'y', numx=100)
    plot_ice(ice, 'x3', 'y', ax=axes[1, 1], yrange=(-4, 4))

    axes[0, 1].get_yaxis().set_visible(False)
    axes[1, 1].get_yaxis().set_visible(False)

    axes[0, 0].set_title("StratPD", fontsize=10)
    axes[0, 1].set_title("PD/ICE", fontsize=10)
    plt.show()

bigX()