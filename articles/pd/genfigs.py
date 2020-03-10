"""
MIT License

Copyright (c) 2019 Terence Parr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
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
from stratx.support import *
from stratx.partdep import *
from stratx.ice import *
import inspect
import statsmodels.api as sm

import shap
import xgboost as xgb

np.random.seed(1)  # pick seed for reproducible article images

# This genfigs.py code is just demonstration code to generate figures for the paper.
# There are lots of programming sins committed here; to not take this to be
# our idea of good code. ;)

# For data sources, please see notebooks/examples.ipynb

def df_split_dates(df,colname):
    df["saleyear"] = df[colname].dt.year
    df["salemonth"] = df[colname].dt.month
    df["saleday"] = df[colname].dt.day
    df["saledayofweek"] = df[colname].dt.dayofweek
    df["saledayofyear"] = df[colname].dt.dayofyear
    df[colname] = df[colname].astype(np.int64) # convert to seconds since 1970


def df_string_to_cat(df: pd.DataFrame) -> dict:
    catencoders = {}
    for colname in df.columns:
        if is_string_dtype(df[colname]) or is_object_dtype(df[colname]):
            df[colname] = df[colname].astype('category').cat.as_ordered()
            catencoders[colname] = df[colname].cat.categories
    return catencoders


def df_cat_to_catcode(df):
    for col in df.columns:
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes + 1


def addnoise(df, n=1, c=0.5, prefix=''):
    if n == 1:
        df[f'{prefix}noise'] = np.random.random(len(df)) * c
        return
    for i in range(n):
        df[f'{prefix}noise{i + 1}'] = np.random.random(len(df)) * c


def fix_missing_num(df, colname):
    df[colname + '_na'] = pd.isnull(df[colname])
    df[colname].fillna(df[colname].median(), inplace=True)


def savefig(filename, pad=0):
    plt.tight_layout(pad=pad, w_pad=0, h_pad=0)
    plt.savefig(f"images/{filename}.pdf",
                bbox_inches="tight", pad_inches=0)
    # plt.savefig(f"images/{filename}.png", dpi=150)

    plt.tight_layout()
    plt.show()

    plt.close()


def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n // 2
    nwomen = n // 2
    df['sex'] = ['M'] * nmen + ['F'] * nwomen
    df.loc[df['sex'] == 'F', 'pregnant'] = np.random.randint(0, 2, size=(nwomen,))
    df.loc[df['sex'] == 'M', 'pregnant'] = 0
    df.loc[df['sex'] == 'M', 'height'] = 5 * 12 + 8 + np.random.uniform(-7, +8,
                                                                        size=(nmen,))
    df.loc[df['sex'] == 'F', 'height'] = 5 * 12 + 5 + np.random.uniform(-4.5, +5,
                                                                        size=(nwomen,))
    df.loc[df['sex'] == 'M', 'education'] = 10 + np.random.randint(0, 8, size=nmen)
    df.loc[df['sex'] == 'F', 'education'] = 12 + np.random.randint(0, 8, size=nwomen)
    df['weight'] = 120 \
                   + (df['height'] - df['height'].min()) * 10 \
                   + df['pregnant'] * 30 \
                   - df['education'] * 1.5
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    return df


def rent():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X,y = load_rent(n=10_000)
    df_rent = X.copy()
    df_rent['price'] = y
    figsize = (9, 2.7)
    colname = 'bedrooms'
    xticks = [0, 2, 4, 6, 8]
    colname = 'bathrooms'
    xticks = list(range(0,9))

    TUNE_RF = False
    # TUNE_SVM = False
    TUNE_XGB = False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if TUNE_RF:
        rf, bestparams = tune_RF(X, y)
        # bedrooms
        # RF best: {'max_features': 0.3, 'min_samples_leaf': 1, 'n_estimators': 125}
        # validation R^2 0.7873724127323822
        # bathrooms
        # RF best: {'max_features': 0.3, 'min_samples_leaf': 1, 'n_estimators': 150}
        # validation R^2 0.7797272189776407
    else:
        rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=1, max_features=.3,
                                   oob_score=True)
        rf.fit(X, y) # Use full data set for plotting
        print("RF OOB R^2", rf.oob_score_)

    if TUNE_XGB:
        tuned_parameters = {'n_estimators': [50, 100, 150, 200, 250],
                            'max_depth': [3, 5, 7, 9]}
        grid = GridSearchCV(
            xgb.XGBRegressor(), tuned_parameters, scoring='r2',
            cv=5,
            n_jobs=-1
            # verbose=2
        )
        grid.fit(X, y)  # does CV on entire data set
        b = grid.best_estimator_
        print("XGB best:", grid.best_params_)
        b.fit(X_train, y_train)
        print("XGB validation R^2", b.score(X_test, y_test))
        # bedrooms
        # XGB best: {'max_depth': 7, 'n_estimators': 250}
        # XGB validation R^2 0.7945797751555217
        # bathrooms
        # XGB best: {'max_depth': 9, 'n_estimators': 250}
        # XGB validation R^2 0.7907897088836073
    else:
        b = xgb.XGBRegressor(n_estimators=250, max_depth=9)
        b.fit(X_train, y_train)
        print("XGB validation R^2", b.score(X_test, y_test))
        b.fit(X, y)  # Use full data set for plotting

    '''
    if TUNE_SVM:
        tuned_parameters = {'kernel': ['poly', 'linear'],
                            'gamma': np.linspace(0.0004, 1, num=5),
                            'C': [1e2,1e3,1e4,1e5,1e6,1e7,1e8]}
        grid = GridSearchCV(svm.SVR(), cv=5, param_grid=tuned_parameters,
                            n_jobs=-1, verbose=2)
        grid.fit(X, y)
        svr = grid.best_estimator_
        print("SVM best:",grid.best_params_)
        svr.fit(X_train, y_train)
        print("SVM validation R^2", svr.score(X_test, y_test))
    else:
        # svr = svm.SVR(C=1e6, gamma='scale', kernel='poly')
        # svr = svm.SVR(C=1e5, gamma='scale', kernel='linear')
        svr = svm.SVR(C=1, gamma='scale', kernel='poly')
        svr.fit(X_train, y_train)
        print("SVM validation R^2", svr.score(X_test, y_test))
        svr.fit(X, y) # Use full data set for drawing
    '''

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    axes[0].set_title("(a) Marginal", fontsize=10)
    axes[0].set_xlim(0,8);
    axes[0].set_xticks(xticks)

    axes[1].set_title("(b) RF PD/ICE", fontsize=10)
    axes[1].set_xlim(0,8); axes[1].set_xticks(xticks)

    axes[2].set_title("(c) XGBoost PD/ICE", fontsize=10)
    axes[2].set_xlim(0,8); axes[2].set_xticks(xticks)

    axes[3].set_title("(d) OLS PD/ICE", fontsize=10)
    axes[3].set_xlim(0,8); axes[3].set_xticks(xticks)

    avg_per_baths = df_rent.groupby(colname).mean()['price']
    axes[0].scatter(df_rent[colname], df_rent['price'], alpha=0.07,
                       s=5)  # , label="observation")
    axes[0].scatter(np.unique(df_rent[colname]), avg_per_baths, s=6, c='black',
                       label="average price/{colname}")
    axes[0].set_ylabel("price")  # , fontsize=12)
    axes[0].set_ylim(0, 12_000)
    axes[0].set_yticks(np.array(range(0,12))*1000)


    ice = predict_ice(rf, X, colname, 'price', numx=30, nlines=100)
    plot_ice(ice, colname, 'price', alpha=.3, ax=axes[1], show_xlabel=False,
             show_ylabel=False)
    # axes[1].set_ylim(-1000, 5000)

    ice = predict_ice(b, X, colname, 'price', numx=30, nlines=100)
    plot_ice(ice, colname, 'price', alpha=.3, ax=axes[2], show_ylabel=True)
    # axes[2].set_ylim(-1000, 5000)

    lm = LinearRegression()
    lm.fit(X_train, y_train)
    # OLS validation R^2 0.6529604563013247
    print("OLS validation R^2", lm.score(X_test, y_test))
    lm.fit(X, y)
    ice = predict_ice(lm, X, colname, 'price', numx=30, nlines=100)
    plot_ice(ice, colname, 'price', alpha=.3, ax=axes[3], show_ylabel=True)
    # axes[3].set_ylim(-1000, 5000)

    savefig(f"{colname}_vs_price")


def tune_RF(X, y, verbose=2):
    tuned_parameters = {'n_estimators': [50, 100, 125, 150, 200],
                        'min_samples_leaf': [1, 3, 5, 7],
                        'max_features': [.1, .3, .5, .7, .9]}
    grid = GridSearchCV(
        RandomForestRegressor(), tuned_parameters, scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=verbose
    )
    grid.fit(X, y)  # does CV on entire data set
    rf = grid.best_estimator_
    print("RF best:", grid.best_params_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf.fit(X_train, y_train)
    print("validation R^2", rf.score(X_test, y_test))
    return rf, grid.best_params_


'''
def rent_grid():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent[-10_000:]  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    plot_stratpd_gridsearch(X, y, 'latitude', 'price',
                            min_samples_leaf_values=[5,10,30,50],
                            yrange=(-500,3500),
                            show_slope_lines=True,
                            marginal_alpha=0.05
                            )

    savefig("latitude_meta")

    plot_stratpd_gridsearch(X, y, 'longitude', 'price',
                            min_samples_leaf_values=[5,10,30,50],
                            yrange=(1000,-4000),
                            show_slope_lines=True,
                            marginal_alpha=0.05
                            )

    savefig("longitude_meta")

    plot_stratpd_gridsearch(X, y, 'bathrooms', 'price',
                            min_samples_leaf_values=[5,10,30,50],
                            yrange=(-500,4000),
                            show_slope_lines=True,
                            slope_line_alpha=.15)

    savefig("bathrooms_meta")


def rent_alone():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent[-10_000:]  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    def onevar(colname, row, col, yrange=None, slope_line_alpha=.2):
        plot_stratpd(X, y, colname, 'price', ax=axes[row, col],
                     min_samples_leaf=20,
                     yrange=yrange,
                     slope_line_alpha=slope_line_alpha,
                     pdp_marker_size=2 if row >= 2 else 8)
        plot_stratpd(X, y, colname, 'price', ax=axes[row, col + 1],
                     min_samples_leaf=20,
                     yrange=yrange,
                     slope_line_alpha=slope_line_alpha,
                     pdp_marker_size=2 if row >= 2 else 8)

    fig, axes = plt.subplots(4, 2, figsize=(5, 8))#, sharey=True)
    # for i in range(1, 4):
    #     axes[0, i].get_yaxis().set_visible(False)
    #     axes[1, i].get_yaxis().set_visible(False)
    #     axes[2, i].get_yaxis().set_visible(False)

    onevar('bedrooms', row=0, col=0, yrange=(0, 3000))
    onevar('bathrooms', row=1, col=0, yrange=(-500, 3000))
    onevar('latitude', row=2, col=0, yrange=(-500, 3000))
    onevar('longitude', row=3, col=0, slope_line_alpha=.08, yrange=(-3000, 1000))

    savefig(f"rent_all")
    plt.close()
'''

def plot_with_noise_col(df, colname):
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
    features_with_noise = ['bedrooms', 'bathrooms', 'latitude', 'longitude',
                           colname + '_noise']

    type = "noise"

    fig, axes = plt.subplots(2, 2, figsize=(5, 5), sharey=True, sharex=True)

    df = df.copy()
    addnoise(df, n=1, c=50, prefix=colname + '_')

    X = df[features]
    y = df['price']

    # STRATPD ON ROW 1
    X = df[features]
    y = df['price']
    plot_stratpd(X, y, colname, 'price', ax=axes[0, 0], slope_line_alpha=.15, show_xlabel=True,
                 show_ylabel=False)
    axes[0, 0].set_ylim(-1000, 5000)
    axes[0, 0].set_title(f"StratPD")

    X = df[features_with_noise]
    y = df['price']
    plot_stratpd(X, y, colname, 'price', ax=axes[0, 1], slope_line_alpha=.15,
                 show_ylabel=False)
    axes[0, 1].set_ylim(-1000, 5000)
    axes[0, 1].set_title(f"StratPD w/{type} col")

    # ICE ON ROW 2
    X = df[features]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True,
                               n_jobs=-1)
    rf.fit(X, y)
    # do it w/o dup'd column
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    uniq_x, pdp_curve = \
        plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 0], show_xlabel=True)
    axes[1, 0].set_ylim(-1000, 5000)
    axes[1, 0].set_title(f"PD/ICE")

    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlim(0, 6)

    X = df[features_with_noise]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True,
                               n_jobs=-1)
    rf.fit(X, y)
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    uniq_x_, pdp_curve_ = \
        plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 1], show_xlabel=True,
                 show_ylabel=False)
    axes[1, 1].set_ylim(-1000, 5000)
    axes[1, 1].set_title(f"PD/ICE w/{type} col")
    # print(f"max ICE curve {np.max(pdp_curve):.0f}, max curve with dup {np.max(pdp_curve_):.0f}")

    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 1].get_xaxis().set_visible(False)


def plot_with_dup_col(df, colname, min_samples_leaf):
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
    features_with_dup = ['bedrooms', 'bathrooms', 'latitude', 'longitude',
                         colname + '_dup']

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5), sharey=True, sharex=True)

    type = "dup"
    verbose = False

    df = df.copy()
    df[colname + '_dup'] = df[colname]
    # df_rent[colname+'_dupdup'] = df_rent[colname]

    # STRATPD ON ROW 1
    X = df[features]
    y = df['price']
    print(f"shape is {X.shape}")
    plot_stratpd(X, y, colname, 'price', ax=axes[0, 0], slope_line_alpha=.15,
                 show_xlabel=True,
                 min_samples_leaf=min_samples_leaf,
                 show_ylabel=True,
                 verbose=verbose)
    axes[0, 0].set_ylim(-1000, 5000)
    axes[0, 0].set_title(f"StratPD")

    X = df[features_with_dup]
    y = df['price']
    print(f"shape with dup is {X.shape}")
    plot_stratpd(X, y, colname, 'price', ax=axes[0, 1], slope_line_alpha=.15, show_ylabel=False,
                 min_samples_leaf=min_samples_leaf,
                 verbose=verbose)
    axes[0, 1].set_ylim(-1000, 5000)
    axes[0, 1].set_title(f"StratPD w/{type} col")

    plot_stratpd(X, y, colname, 'price', ax=axes[0, 2], slope_line_alpha=.15, show_xlabel=True,
                 min_samples_leaf=min_samples_leaf,
                 show_ylabel=False,
                 n_trees=15,
                 max_features=1,
                 bootstrap=False,
                 verbose=verbose
                 )
    axes[0, 2].set_ylim(-1000, 5000)
    axes[0, 2].set_title(f"StratPD w/{type} col")
    axes[0, 2].text(.2, 4000, "ntrees=15")
    axes[0, 2].text(.2, 3500, "max features per split=1")

    # ICE ON ROW 2
    X = df[features]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True,
                               n_jobs=-1)
    rf.fit(X, y)

    # do it w/o dup'd column
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 0], show_xlabel=True)
    axes[1, 0].set_ylim(-1000, 5000)
    axes[1, 0].set_title(f"PD/ICE")

    for i in range(2):
        for j in range(3):
            axes[i, j].set_xlim(0, 6)

    # with dup'd column
    X = df[features_with_dup]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True,
                               n_jobs=-1)
    rf.fit(X, y)
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 1], show_xlabel=True, show_ylabel=False)
    axes[1, 1].set_ylim(-1000, 5000)
    axes[1, 1].set_title(f"PD/ICE w/{type} col")
    # print(f"max ICE curve {np.max(pdp_curve):.0f}, max curve with dup {np.max(pdp_curve_):.0f}")

    axes[1, 2].set_title(f"PD/ICE w/{type} col")
    axes[1, 2].text(.2, 4000, "Cannot compensate")
    axes[1, 2].set_xlabel(colname)

    # print(f"max curve {np.max(curve):.0f}, max curve with dup {np.max(curve_):.0f}")

    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 1].get_xaxis().set_visible(False)


def rent_ntrees():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X, y = load_rent(n=10_000)

    trees = [1, 5, 10, 30]

    supervised = True

    def onevar(colname, row, yrange=None):
        alphas = [.1,.08,.05,.04]
        for i, t in enumerate(trees):
            plot_stratpd(X, y, colname, 'price', ax=axes[row, i], slope_line_alpha=alphas[i],
                         # min_samples_leaf=20,
                         yrange=yrange,
                         supervised=supervised,
                         show_ylabel=t == 1,
                         pdp_marker_size=2 if row==2 else 8,
                         n_trees=t,
                         max_features='auto',
                         bootstrap=True,
                         verbose=False)

    fig, axes = plt.subplots(3, 4, figsize=(8, 6), sharey=True)
    for i in range(1, 4):
        axes[0, i].get_yaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)

    for i in range(0, 4):
        axes[0, i].set_title(f"{trees[i]} trees")

    onevar('bedrooms', row=0, yrange=(-500, 4000))
    onevar('bathrooms', row=1, yrange=(-500, 4000))
    onevar('latitude', row=2, yrange=(-500, 4000))

    savefig(f"rent_ntrees")
    plt.close()


def meta_boston():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    boston = load_boston()
    print(len(boston.data))
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']


    plot_stratpd_gridsearch(X, y, 'AGE', 'MEDV',
                            show_slope_lines=True,
                            min_samples_leaf_values=[2,5,10,20,30],
                            yrange=(-10,10))

    # yranges = [(-30, 0), (0, 30), (-8, 8), (-11, 0)]
    # for nbins in range(6):
    #     plot_meta_multivar(X, y, colnames=['LSTAT', 'RM', 'CRIM', 'DIS'], targetname='MEDV',
    #                        nbins=nbins,
    #                        yranges=yranges)

    savefig(f"meta_boston_age_medv")


def plot_meta_multivar(X, y, colnames, targetname, nbins, yranges=None):
    min_samples_leaf_values = [2, 5, 10, 30, 50, 100, 200]

    nrows = len(colnames)
    ncols = len(min_samples_leaf_values)
    fig, axes = plt.subplots(nrows, ncols + 2, figsize=((ncols + 2) * 2.5, nrows * 2.5))

    if yranges is None:
        yranges = [None] * len(colnames)

    row = 0
    for i, colname in enumerate(colnames):
        marginal_plot_(X, y, colname, targetname, ax=axes[row, 0])
        col = 2
        for msl in min_samples_leaf_values:
            print(
                f"---------- min_samples_leaf={msl}, nbins={nbins:.2f} ----------- ")
            plot_stratpd(X, y, colname, targetname, ax=axes[row, col],
                         min_samples_leaf=msl,
                         yrange=yranges[i],
                         n_trees=1)
            axes[row, col].set_title(
                f"leafsz={msl}, nbins={nbins:.2f}",
                fontsize=9)
            col += 1
        row += 1

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    row = 0
    for i, colname in enumerate(colnames):
        ice = predict_ice(rf, X, colname, targetname)
        plot_ice(ice, colname, targetname, ax=axes[row, 1])
        row += 1


def unsup_rent():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X, y = load_rent(n=10_000)

    fig, axes = plt.subplots(3, 2, figsize=(4, 6))

    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 0], yrange=(-500,4000), slope_line_alpha=.2, supervised=False)
    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 1], yrange=(-500,4000), slope_line_alpha=.2, supervised=True)

    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 0], yrange=(-500,4000), slope_line_alpha=.2, supervised=False)
    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 1], yrange=(-500,4000), slope_line_alpha=.2, supervised=True)

    plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 0], yrange=(-500,4000), slope_line_alpha=.2, supervised=False)
    plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 1], yrange=(-500,4000), slope_line_alpha=.2, supervised=True)

    axes[0, 0].set_title("Unsupervised")
    axes[0, 1].set_title("Supervised")

    for i in range(3):
        axes[i, 1].get_yaxis().set_visible(False)

    savefig(f"rent_unsup")
    plt.close()


def toy_weather_data():
    def temp(x): return np.sin((x + 365 / 2) * (2 * np.pi) / 365)

    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df = pd.DataFrame()
    df['dayofyear'] = range(1, 365 + 1)
    df['state'] = np.random.choice(['CA', 'CO', 'AZ', 'WA'], len(df))
    df['temperature'] = temp(df['dayofyear'])
    df.loc[df['state'] == 'CA', 'temperature'] = \
        70 + df.loc[df['state'] == 'CA', 'temperature'] * noise('CA')
    df.loc[df['state'] == 'CO', 'temperature'] = \
        40 + df.loc[df['state'] == 'CO', 'temperature'] * noise('CO')
    df.loc[df['state'] == 'AZ', 'temperature'] = \
        90 + df.loc[df['state'] == 'AZ', 'temperature'] * noise('AZ')
    df.loc[df['state'] == 'WA', 'temperature'] = \
        60 + df.loc[df['state'] == 'WA', 'temperature'] * noise('WA')
    return df


def weather():
    print(f"----------- {inspect.stack()[0][3]} -----------")
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
    # cats = catencoders['state'].values
    # cats = np.insert(cats, 0, None) # prepend a None for catcode 0

    figsize = (2.5, 2.5)
    """
    The scale diff between states, obscures the sinusoidal nature of the
    dayofyear vs temp plot. With noise N(0,5) gotta zoom in -3,3 on mine too.
    otherwise, smooth quasilinear plot with lots of bristles showing volatility.
    Flip to N(-5,5) which is more realistic and we see sinusoid for both, even at
    scale. yep, the N(0,5) was obscuring sine for both. 
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_stratpd(X, y, 'dayofyear', 'temperature', ax=ax,
                 yrange=(-10, 10),
                 pdp_marker_size=2, slope_line_alpha=.5, n_trials=1)

    ax.set_title("(b) StratPD")
    savefig(f"dayofyear_vs_temp_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'state', 'temperature', catnames=catnames,
                    min_samples_leaf=30,
                    # alpha=.3,
                    ax=ax,
                    yrange=(-2, 60))

    ax.set_title("(b) StratPD")
    savefig(f"state_vs_temp_stratpd")

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'dayofyear', 'temperature')
    plot_ice(ice, 'dayofyear', 'temperature', ax=ax, yrange=(-15, 15))
    ax.set_title("(c) PD/ICE")
    savefig(f"dayofyear_vs_temp_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_catice(rf, X, 'state', 'temperature')
    plot_catice(ice, 'state', 'temperature', catnames=catnames, ax=ax,
                pdp_marker_size=10,
                yrange=(-2, 60))
    ax.set_title("(c) PD/ICE")
    savefig(f"state_vs_temp_pdp")

    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    # rtreeviz_univar(ax,
    #                 X['state'], y,
    #                 feature_name='state',
    #                 target_name='y',
    #                 fontsize=10, show={'splits'})
    #
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(X['state'], y, alpha=.05, s=15)
    ax.set_xticks([5,10,15,20])
    ax.set_xticklabels(catnames.values())
    ax.set_xlabel("state")
    ax.set_ylabel("temperature")
    ax.set_title("(a) Marginal")
    savefig(f"state_vs_temp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df = df_raw.copy()
    avgtmp = df.groupby(['state', 'dayofyear'])[['temperature']].mean()
    avgtmp = avgtmp.reset_index()
    ca = avgtmp.query('state=="CA"')
    co = avgtmp.query('state=="CO"')
    az = avgtmp.query('state=="AZ"')
    wa = avgtmp.query('state=="WA"')
    ax.plot(ca['dayofyear'], ca['temperature'], lw=.5, c='#fdae61', label="CA")
    ax.plot(co['dayofyear'], co['temperature'], lw=.5, c='#225ea8', label="CO")
    ax.plot(az['dayofyear'], az['temperature'], lw=.5, c='#41b6c4', label="AZ")
    ax.plot(wa['dayofyear'], wa['temperature'], lw=.5, c='#a1dab4', label="WA")
    ax.legend(loc='lower left', borderpad=0, labelspacing=0)
    ax.set_xlabel("dayofyear")
    ax.set_ylabel("temperature")
    ax.set_title("(a) State/day vs temp")

    savefig(f"dayofyear_vs_temp")
    plt.close()


def meta_weather():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    # np.random.seed(66)

    nyears = 5
    years = []
    for y in range(1980, 1980 + nyears):
        df_ = toy_weather_data()
        df_['year'] = y
        years.append(df_)

    df_raw = pd.concat(years, axis=0)

    # df_raw.drop('year', axis=1, inplace=True)
    df = df_raw.copy()
    print(df.head(5))

    names = {'CO': 5, 'CA': 10, 'AZ': 15, 'WA': 20}
    df['state'] = df['state'].map(names)
    catnames = {v:k for k,v in names.items()}

    X = df.drop('temperature', axis=1)
    y = df['temperature']

    plot_catstratpd_gridsearch(X, y, 'state', 'temp',
                               min_samples_leaf_values=[2, 5, 20, 40, 60],
                               catnames=catnames,
                               yrange=(-5,60),
                               cellwidth=2
                               )
    savefig(f"state_temp_meta")

    plot_stratpd_gridsearch(X, y, 'dayofyear', 'temp',
                            show_slope_lines=True,
                            min_samples_leaf_values=[2,5,10,20,30],
                            yrange=(-10,10),
                            slope_line_alpha=.15)
    savefig(f"dayofyear_temp_meta")


def weight():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(2000)
    df = df_raw.copy()
    df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']
    figsize = (2.5, 2.5)

    TUNE_RF = False

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_stratpd(X, y, 'education', 'weight', ax=ax,
                 show_x_counts=False,
                 pdp_marker_size=5,
                 yrange=(-12, 0.05), slope_line_alpha=.1, show_ylabel=True)
    #    ax.get_yaxis().set_visible(False)
    ax.set_title("StratPD", fontsize=10)
    ax.set_xlim(10,18)
    ax.set_xticks([10,12,14,16,18])
    savefig(f"education_vs_weight_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_stratpd(X, y, 'height', 'weight', ax=ax,
                 pdp_marker_size=.2,
                 show_x_counts=False,
                 yrange=(0, 160), show_ylabel=False)
    #    ax.get_yaxis().set_visible(False)
    ax.set_title("StratPD", fontsize=10)
    savefig(f"height_vs_weight_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'sex', 'weight', ax=ax,
                    show_x_counts=False,
                    catnames={1: 'F', 2: 'M'},
                    yrange=(-1, 35),
                    )
    ax.set_title("StratPD", fontsize=10)
    savefig(f"sex_vs_weight_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=ax,
                    show_x_counts=False,
                    catnames={0:False, 1:True},
                    yrange=(-5, 35),
                    )
    ax.set_title("StratPD", fontsize=10)
    savefig(f"pregnant_vs_weight_stratpd")

    if TUNE_RF:
        rf, bestparams = tune_RF(X, y)
        # RF best: {'max_features': 0.9, 'min_samples_leaf': 1, 'n_estimators': 200}
        # validation R^2 0.9996343699640691
    else:
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, max_features=0.9, oob_score=True)
        rf.fit(X, y) # Use full data set for plotting
        print("RF OOB R^2", rf.oob_score_)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'education', 'weight')
    plot_ice(ice, 'education', 'weight', ax=ax, yrange=(-12, 0), min_y_shifted_to_zero=True)
    ax.set_xlim(10,18)
    ax.set_xticks([10,12,14,16,18])
    ax.set_title("PD/ICE", fontsize=10)
    savefig(f"education_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'height', 'weight')
    plot_ice(ice, 'height', 'weight', ax=ax, yrange=(0, 160), min_y_shifted_to_zero=True)
    ax.set_title("PD/ICE", fontsize=10)
    ax.set_title("PD/ICE", fontsize=10)
    savefig(f"height_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_catice(rf, X, 'sex', 'weight')
    plot_catice(ice, 'sex', 'weight', catnames=df_raw['sex'].unique(), ax=ax, yrange=(0, 30),
                pdp_marker_size=15)
    ax.set_title("PD/ICE", fontsize=10)
    savefig(f"sex_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_catice(rf, X, 'pregnant', 'weight', cats=df_raw['pregnant'].unique())
    plot_catice(ice, 'pregnant', 'weight', catnames=df_raw['pregnant'].unique(), ax=ax,
                yrange=(-5, 35), pdp_marker_size=15)
    ax.set_title("PD/ICE", fontsize=10)
    savefig(f"pregnant_vs_weight_pdp")


def shap_weight(feature_perturbation, twin=False):
    n = 2000
    shap_test_size = 2000
    df_raw = toy_weight_data(n=n)
    df = df_raw.copy()
    df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']
    figsize = (2.5, 2.5)

    # parameters from tune_RF() called in rent()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1,
                               max_features=0.9,
                               oob_score=True)
    rf.fit(X, y)  # Use full data set for plotting
    print("RF OOB R^2", rf.oob_score_)

    if feature_perturbation=='interventional':
        explainer = shap.TreeExplainer(rf, data=shap.sample(X, 500), feature_perturbation='interventional')
        xlabel = "height\n(b)"
    else:
        explainer = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
        xlabel = "height\n(a)"
    shap_sample = X[:shap_test_size]
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    GREY = '#444443'
    fig, ax = plt.subplots(1, 1, figsize=(3.8,3.2))

    shap.dependence_plot("height", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=1)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['top'].set_linewidth(.5)

    ax.set_ylabel("Height (SHAP values)", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.plot([70,70], [-75,75], '--', lw=.6, color=GREY)
    ax.text(69.8,60, "Max female height", horizontalalignment='right',
            fontsize=9)

    leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
        partial_dependence(X=X, y=y, colname='height')

    ax.set_ylim(-77,75)
    # ax.set_xlim(min(pdpx), max(pdpx))
    ax.set_xticks([60,65,70,75])
    ax.set_yticks([-75,-60,-40,-20,0,20,40,60,75])

    ax.set_title(f"SHAP {feature_perturbation}", fontsize=12)
    # ax.set_ylim(-40,70)

    print(min(pdpx), max(pdpx))
    print(min(pdpy), max(pdpy))
    rise = max(pdpy) - min(pdpy)
    run = max(pdpx) - min(pdpx)
    slope = rise/run
    print(slope)
    # ax.plot([min(pdpx),max(pdpyX['height'])], [0,]

    if twin:
        ax2 = ax.twinx()
        # ax2.set_xlim(min(pdpx), max(pdpx))
        ax2.set_ylim(min(pdpy)-5, max(pdpy)+5)
        ax2.set_xticks([60,65,70,75])
        ax2.set_yticks([0,20,40,60,80,100,120,140,150])
        ax2.set_ylabel("weight", fontsize=12)

        ax2.plot(pdpx, pdpy, '.', markersize=1, c='k')
        # ax2.text(65,25, f"StratPD slope = {slope:.1f}")
        ax2.annotate(f"StratPD (slope={slope:.1f})", (64.65,39), xytext=(66,18),
                     horizontalalignment='left',
                     arrowprops=dict(facecolor='black', width=.5, headwidth=5, headlength=5),
                     fontsize=9)

    savefig(f"weight_{feature_perturbation}_shap")


def yearmade():
    n = 10_000
    shap_test_size = 1000
    TUNE_RF = False

    X, y = load_bulldozer(n=n)

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.2))
    ax.scatter(X['YearMade'], y, s=3, alpha=.1, c='#1E88E5')
    ax.set_xlim(1960,2010)
    ax.set_xlabel("YearMade\n(a)", fontsize=11)
    ax.set_ylabel("SalePrice ($)", fontsize=11)
    ax.set_title("Marginal plot", fontsize=13)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    savefig(f"bulldozer_YearMade_marginal")

    if TUNE_RF:
        rf, _ = tune_RF(X, y)
        # RF best: {'max_features': 0.9, 'min_samples_leaf': 1, 'n_estimators': 150}
        # validation R^2 0.8001628465688546
    else:
        rf = RandomForestRegressor(n_estimators=150, n_jobs=-1,
                                   max_features=0.9,
                                   min_samples_leaf=1, oob_score=True)
        rf.fit(X, y)
        print("RF OOB R^2", rf.oob_score_)

    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100),
                                   feature_perturbation='interventional')
    shap_values = explainer.shap_values(X.sample(n=shap_test_size),
                                        check_additivity=False)

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.2))
    shap.dependence_plot("YearMade", shap_values, X.sample(n=shap_test_size),
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=.5)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_title("SHAP", fontsize=13)
    ax.set_ylabel("Impact on SalePrice\n(YearMade SHAP)", fontsize=11)
    ax.set_xlabel("YearMade\n(b)", fontsize=11)
    ax.set_xlim(1960, 2010)
    ax.tick_params(axis='both', which='major', labelsize=10)

    savefig(f"bulldozer_YearMade_shap")

    fig, ax = plt.subplots(1, 1, figsize=(3.8,3.2))
    plot_stratpd(X, y, colname='YearMade', targetname='SalePrice',
                 n_trials=10,
                 bootstrap=True,
                 show_slope_lines=False,
                 show_x_counts=True,
                 show_xlabel=False,
                 show_impact=False,
                 pdp_marker_size=4,
                 pdp_marker_alpha=1,
                 ax=ax
                 )
    ax.set_title("StratPD", fontsize=13)
    ax.set_xlabel("YearMade\n(c)", fontsize=11)
    ax.set_xlim(1960,2010)
    ax.set_ylim(-10000,30_000)
    savefig(f"bulldozer_YearMade_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=(3.8,3.2))
    ice = predict_ice(rf, X, "YearMade", 'SalePrice', numx=30, nlines=100)
    plot_ice(ice, "YearMade", 'SalePrice', alpha=.3, ax=ax, show_ylabel=True,
             yrange=(-10000,30_000),
             min_y_shifted_to_zero=True)
    ax.set_xlim(1960, 2010)
    savefig(f"bulldozer_YearMade_pdp")


def unsup_weight():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(2000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 0],
                 yrange=(-12, 0), slope_line_alpha=.1, supervised=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 1],
                 yrange=(-12, 0), slope_line_alpha=.1, supervised=True)

    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 0],
                    catnames=df_raw['pregnant'].unique(),
                    yrange=(-5, 35))
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 1],
                    catnames=df_raw['pregnant'].unique(),
                    yrange=(-5, 35))

    axes[0, 0].set_title("Unsupervised")
    axes[0, 1].set_title("Supervised")

    axes[0, 1].get_yaxis().set_visible(False)
    axes[1, 1].get_yaxis().set_visible(False)

    savefig(f"weight_unsup")
    plt.close()


def weight_ntrees():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    trees = [1, 5, 10, 30]

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for i in range(1, 4):
        axes[0, i].get_yaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

    for i in range(0, 4):
        axes[0, i].set_title(f"{trees[i]} trees")

    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 0],
                 min_samples_leaf=5,
                 yrange=(-12, 0), slope_line_alpha=.1, pdp_marker_size=10, show_ylabel=True,
                 n_trees=1, max_features=1.0, bootstrap=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 1],
                 min_samples_leaf=5,
                 yrange=(-12, 0), slope_line_alpha=.1, pdp_marker_size=10, show_ylabel=False,
                 n_trees=5, max_features='auto', bootstrap=True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 2],
                 min_samples_leaf=5,
                 yrange=(-12, 0), slope_line_alpha=.08, pdp_marker_size=10, show_ylabel=False,
                 n_trees=10, max_features='auto', bootstrap=True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 3],
                 min_samples_leaf=5,
                 yrange=(-12, 0), slope_line_alpha=.05, pdp_marker_size=10, show_ylabel=False,
                 n_trees=30, max_features='auto', bootstrap=True)

    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 0],
                    catnames={0:False, 1:True}, show_ylabel=True,
                    yrange=(0, 35),
                    n_trees=1, max_features=1.0, bootstrap=False)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 1],
                    catnames={0:False, 1:True}, show_ylabel=False,
                    yrange=(0, 35),
                    n_trees=5, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 2],
                    catnames={0:False, 1:True}, show_ylabel=False,
                    yrange=(0, 35),
                    n_trees=10, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 3],
                    catnames={0:False, 1:True}, show_ylabel=False,
                    yrange=(0, 35),
                    n_trees=30, max_features='auto', bootstrap=True)

    savefig(f"education_pregnant_vs_weight_ntrees")
    plt.close()


def meta_weight():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    plot_stratpd_gridsearch(X, y, colname='education', targetname='weight',
                            show_slope_lines=True,
                            xrange=(10,18),
                            yrange=(-12,0))
    savefig("education_weight_meta")

    plot_stratpd_gridsearch(X, y, colname='height', targetname='weight', yrange=(0,150),
                            show_slope_lines=True)
    savefig("height_weight_meta")


def additivity_data(n, sd=1.0):
    x1 = np.random.uniform(-1, 1, size=n)
    x2 = np.random.uniform(-1, 1, size=n)

    y = x1 ** 2 + x2 + np.random.normal(0, sd, size=n)
    df = pd.DataFrame()
    df['x1'] = x1
    df['x2'] = x2
    df['y'] = y
    return df


def additivity():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    df = additivity_data(n=n, sd=1)  # quite noisy
    X = df.drop('y', axis=1)
    y = df['y']

    fig, axes = plt.subplots(2, 2, figsize=(4, 4))  # , sharey=True)
    plot_stratpd(X, y, 'x1', 'y',
                 min_samples_leaf=10,
                 ax=axes[0, 0], yrange=(-3, 3))

    plot_stratpd(X, y, 'x2', 'y',
                 min_samples_leaf=10,
                 ax=axes[1, 0],
                 yrange=(-3,3))

    # axes[0, 0].set_ylim(-2, 2)
    # axes[1, 0].set_ylim(-2, 2)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = predict_ice(rf, X, 'x1', 'y', numx=20, nlines=700)
    plot_ice(ice, 'x1', 'y', ax=axes[0, 1], yrange=(-3, 3), show_ylabel=False)

    ice = predict_ice(rf, X, 'x2', 'y', numx=20, nlines=700)
    plot_ice(ice, 'x2', 'y', ax=axes[1, 1], yrange=(-3, 3), show_ylabel=False)

    axes[0, 0].set_title("StratPD", fontsize=10)
    axes[0, 1].set_title("PD/ICE", fontsize=10)

    savefig(f"additivity")


def meta_additivity():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    noises = [0, .5, .8, 1.0]
    sizes = [2, 10, 30, 50]

    fig, axes = plt.subplots(len(noises) + 1, len(sizes), figsize=(7, 8), sharey=True,
                             sharex=True)

    row = 0
    for sd in noises:
        df = additivity_data(n=n, sd=sd)
        X = df.drop('y', axis=1)
        y = df['y']
        col = 0
        for s in sizes:
            if row == 3:
                show_xlabel = True
            else:
                show_xlabel = False
            print(f"------------------- noise {sd}, SIZE {s} --------------------")
            if col > 1: axes[row, col].get_yaxis().set_visible(False)
            plot_stratpd(X, y, 'x1', 'y', ax=axes[row, col],
                         min_samples_leaf=s,
                         yrange=(-1.5, .5),
                         pdp_marker_size=1,
                         slope_line_alpha=.4,
                         show_ylabel=False,
                         show_xlabel=show_xlabel)
            if col == 0:
                axes[row, col].set_ylabel(f'$y, \epsilon \sim N(0,{sd:.2f})$')

            if row == 0:
                axes[row, col].set_title("Min $x_{\\overline{c}}$ leaf " + f"{s}",
                                         fontsize=12)
            col += 1
        row += 1

    lastrow = len(noises)

    axes[lastrow, 0].set_ylabel(f'$y$ vs $x_c$ partition')

    # row = 0
    # for sd in noises:
    #     axes[row, 0].scatter(X['x1'], y, slope_line_alpha=.12, label=None)
    #     axes[row, 0].set_xlabel("x1")
    #     axes[row, 0].set_ylabel("y")
    #     axes[row, 0].set_ylim(-5, 5)
    #     axes[row, 0].set_title(f"$y = x_1^2 + x_2 + \epsilon$, $\epsilon \sim N(0,{sd:.2f})$")
    #     row += 1

    col = 0
    for s in sizes:
        rtreeviz_univar(axes[lastrow, col],
                        X['x2'], y,
                        min_samples_leaf=s,
                        feature_name='x2',
                        target_name='y',
                        fontsize=10, show={'splits'},
                        split_linewidth=.5,
                        markersize=5)
        axes[lastrow, col].set_xlabel("x2")
        col += 1

    savefig(f"meta_additivity_noise")


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
                 min_samples_leaf=5,
                 pdp_marker_size=2)

    # Partial deriv wrt x3 of 1_x3>=0 is 0 everywhere so result must be 0
    plot_stratpd(X, y, 'x3', 'y', ax=axes[1, 0], yrange=(-4, 4),
                 min_samples_leaf=5,
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

    savefig(f"bigx")
    plt.close()


def unsup_boston():
    # np.random.seed(42)

    print(f"----------- {inspect.stack()[0][3]} -----------")
    boston = load_boston()
    print(len(boston.data))
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    fig, axes = plt.subplots(1, 4, figsize=(9, 2))

    axes[0].scatter(df['AGE'], y, s=5, alpha=.7)
    axes[0].set_ylabel('MEDV')
    axes[0].set_xlabel('AGE')

    axes[0].set_title("Marginal")
    axes[1].set_title("Unsupervised StratPD")
    axes[2].set_title("Supervised StratPD")
    axes[3].set_title("PD/ICE")

    plot_stratpd(X, y, 'AGE', 'MEDV', ax=axes[1], yrange=(-20, 20),
                 n_trees=20,
                 bootstrap=True,
                 # min_samples_leaf=10,
                 max_features='auto',
                 supervised=False, show_ylabel=False,
                 verbose=True,
                 slope_line_alpha=.1)
    plot_stratpd(X, y, 'AGE', 'MEDV', ax=axes[2], yrange=(-20, 20),
                 min_samples_leaf=5,
                 n_trees=1,
                 supervised=True, show_ylabel=False)

    axes[1].text(5, 15, f"20 trees, bootstrap")
    axes[2].text(5, 15, f"1 tree, no bootstrap")

    rf = RandomForestRegressor(n_estimators=100, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = predict_ice(rf, X, 'AGE', 'MEDV', numx=10)
    plot_ice(ice, 'AGE', 'MEDV', ax=axes[3], yrange=(-20, 20), show_ylabel=False)

    # axes[0,1].get_yaxis().set_visible(False)
    # axes[1,1].get_yaxis().set_visible(False)

    savefig(f"boston_unsup")
    # plt.tight_layout()
    # plt.show()


def lm_plot(X, y, colname, targetname, ax=None):
    ax.scatter(X[colname], y, alpha=.12, label=None)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    col = X[colname]
    # y_pred_hp = r_col.predict(col.values.reshape(-1, 1))
    # ax.plot(col, y_pred_hp, ":", linewidth=1, c='red', label='y ~ horsepower')

    r = LinearRegression()
    r.fit(X[['horsepower', 'weight']], y)

    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = 0 if colname == 'horsepower' else 1
    # use beta from y ~ hp + weight
    # ax.plot(xcol, xcol * r.coef_[ci] + r.intercept_, linewidth=1, c='orange')
    # ax.text(min(xcol)*1.02, max(y)*.95, f"$\\beta_{{{colname}}}$={r.coef_[ci]:.3f}")

    # r = LinearRegression()
    # r.fit(X[['horsepower','weight']], y)
    # xcol = np.linspace(np.min(col), np.max(col), num=100)
    # ci = X.columns.get_loc(colname)
    # # ax.plot(xcol, xcol * r.coef_[ci] + r_col.intercept_, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    # left40 = xcol[int(len(xcol) * .4)]
    # ax.text(min(xcol), max(y)*.94, f"$\hat{{y}} = \\beta_0 + \\beta_1 x_{{horsepower}} + \\beta_2 x_{{weight}}$")
    # i = 1 if colname=='horsepower' else 2
    # # ax.text(left40, left40*r.coef_[ci] + r_col.intercept_, f"$\\beta_{i}$={r.coef_[ci]:.3f}")


def cars():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_cars = pd.read_csv("../notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)

    X = df_cars[['horsepower', 'weight']]
    y = df_cars['mpg']

    fig, axes = plt.subplots(2, 3, figsize=(9, 4))
    lm_plot(X, y, 'horsepower', 'mpg', ax=axes[0, 0])

    lm_plot(X, y, 'weight', 'mpg', ax=axes[1, 0])

    plot_stratpd(X, y, 'horsepower', 'mpg', ax=axes[0, 1],
                 min_samples_leaf=10,
                 xrange=(45, 235), yrange=(-20, 20), show_ylabel=False)
    plot_stratpd(X, y, 'weight', 'mpg', ax=axes[1, 1],
                 min_samples_leaf=10,
                 xrange=(1600, 5200), yrange=(-20, 20), show_ylabel=False)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    ice = predict_ice(rf, X, 'horsepower', 'mpg', numx=100)
    plot_ice(ice, 'horsepower', 'mpg', ax=axes[0, 2], yrange=(-20, 20), show_ylabel=False)
    ice = predict_ice(rf, X, 'weight', 'mpg', numx=100)
    plot_ice(ice, 'weight', 'mpg', ax=axes[1, 2], yrange=(-20, 20), show_ylabel=False)

    # draw regr line for horsepower
    r = LinearRegression()
    r.fit(X, y)
    colname = 'horsepower'
    col = X[colname]
    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = X.columns.get_loc(colname)
    beta0 = -r.coef_[ci] * min(col)  # solved for beta0 to get y-intercept
    # axes[0,1].plot(xcol, xcol * r.coef_[ci], linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    # axes[0,2].plot(xcol, xcol * r.coef_[ci], linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")

    # draw regr line for weight
    colname = 'weight'
    col = X[colname]
    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = X.columns.get_loc(colname)
    beta0 = -r.coef_[ci] * min(col)  # solved for beta0 to get y-intercept
    # axes[1,1].plot(xcol, xcol * r.coef_[ci]+11, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    # axes[1,2].plot(xcol, xcol * r.coef_[ci]+13, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    axes[1, 2].set_xlim(1600, 5200)
    savefig("cars")


def meta_cars():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_cars = pd.read_csv("../notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)

    X = df_cars[['horsepower', 'weight']]
    y = df_cars['mpg']

    plot_stratpd_gridsearch(X, y, colname='horsepower', targetname='mpg',
                            show_slope_lines=True,
                            min_samples_leaf_values=[2,5,10,20,30],
                            nbins_values=[1,2,3,4,5],
                            yrange=(-20, 20))

    savefig("horsepower_meta")

    plot_stratpd_gridsearch(X, y, colname='weight', targetname='mpg',
                            show_slope_lines=True,
                            min_samples_leaf_values=[2,5,10,20,30],
                            nbins_values=[1,2,3,4,5],
                            yrange=(-20, 20))

    savefig("weight_meta")

'''
def bulldozer():  # warning: takes like 5 minutes to run
    print(f"----------- {inspect.stack()[0][3]} -----------")

    # np.random.seed(42)

    def onecol(X, y, colname, axes, row, xrange, yrange):
        axes[row, 0].scatter(X[colname], y, alpha=0.07, s=1)
        axes[row, 0].set_ylabel("SalePrice")  # , fontsize=12)
        axes[row, 0].set_xlabel(colname)  # , fontsize=12)

        plot_stratpd(X, y, colname, 'SalePrice', ax=axes[row, 1], xrange=xrange,
                     yrange=yrange, show_ylabel=False,
                     verbose=False, slope_line_alpha=.07)

        rf = RandomForestRegressor(n_estimators=20, min_samples_leaf=1, n_jobs=-1,
                                   oob_score=True)
        rf.fit(X, y)
        print(f"{colname} PD/ICE: RF OOB R^2 {rf.oob_score_:.3f}, training R^2 {rf.score(X,y)}")
        ice = predict_ice(rf, X, colname, 'SalePrice', numx=130, nlines=500)
        plot_ice(ice, colname, 'SalePrice', alpha=.05, ax=axes[row, 2], show_ylabel=False,
                 xrange=xrange, yrange=yrange)
        axes[row, 1].set_xlabel(colname)  # , fontsize=12)
        axes[row, 1].set_ylim(*yrange)

    n = 10_000
    X, y = load_bulldozer(n=n)
    print(f"Avg bulldozer price is {np.mean(y):.2f}$")

    fig, axes = plt.subplots(3, 3, figsize=(7, 6))

    onecol(X, y, 'YearMade', axes, 0, xrange=(1960, 2012), yrange=(-1000, 60000))
    onecol(X, y, 'MachineHours', axes, 1, xrange=(0, 35_000),
           yrange=(-40_000, 40_000))

    # show marginal plot sorted by model's sale price
    sort_indexes = y.argsort()

    modelids = X['ModelID'].values
    sorted_modelids = modelids[sort_indexes]
    sorted_ys = y.values[sort_indexes]
    cats = modelids[sort_indexes]
    ncats = len(cats)

    axes[2, 0].set_xticks(range(1, ncats + 1))
    axes[2, 0].set_xticklabels([])
    # axes[2, 0].get_xaxis().set_visible(False)

    xlocs = np.arange(1, ncats + 1)
    axes[2, 0].scatter(xlocs, sorted_ys, alpha=0.2, s=2)  # , label="observation")
    axes[2, 0].set_ylabel("SalePrice")  # , fontsize=12)
    axes[2, 0].set_xlabel('ModelID')  # , fontsize=12)
    axes[2, 0].tick_params(axis='x', which='both', bottom=False)

    plot_catstratpd(X, y, 'ModelID', 'SalePrice',
                    min_samples_leaf=5,
                    ax=axes[2, 1],
                    yrange=(0, 130000),
                    show_ylabel=False,
                    alpha=0.1,
                    # style='strip',
                    marker_size=3,
                    show_xticks=False,
                    verbose=False)

    # plt.tight_layout()
    # plt.show()
    # return

    rf = RandomForestRegressor(n_estimators=20, min_samples_leaf=1, oob_score=True,
                               n_jobs=-1)
    rf.fit(X, y)
    print(
        f"ModelID PD/ICE: RF OOB R^2 {rf.oob_score_:.3f}, training R^2 {rf.score(X, y)}")

    # too slow to do all so get 1000
    ucats = np.unique(X['ModelID'])
    ucats = np.random.choice(ucats, size=1000, replace=False)
    ice = predict_catice(rf, X, 'ModelID', 'SalePrice', cats=ucats)
    plot_catice(ice, 'ModelID', targetname='SalePrice', catnames=ucats,
                alpha=.05, ax=axes[2, 2], yrange=(0, 130000), show_ylabel=False,
                marker_size=3,
                sort='ascending',
                show_xticks=False)

    axes[0, 0].set_title("Marginal")
    axes[0, 1].set_title("StratPD")
    axes[0, 2].set_title("PD/ICE")

    savefig("bulldozer")
    # plt.tight_layout()
    # plt.show()
'''

def multi_joint_distr():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    # np.random.seed(42)
    n = 1000
    min_samples_leaf = 30
    nbins = 2
    df = pd.DataFrame(np.random.multivariate_normal([6, 6, 6, 6],
                                                    [
                                                        [1, 5, .7, 3],
                                                        [5, 1, 2, .5],
                                                        [.7, 2, 1, 1.5],
                                                        [3, .5, 1.5, 1]
                                                    ],
                                                    n),
                      columns=['x1', 'x2', 'x3', 'x4'])
    df['y'] = df['x1'] + df['x2'] + df['x3'] + df['x4']
    X = df.drop('y', axis=1)
    y = df['y']

    r = LinearRegression()
    r.fit(X, y)
    print(r.coef_)  # should be all 1s

    yrange = (-2, 15)

    fig, axes = plt.subplots(6, 4, figsize=(7.5, 8.5), sharey=False)  # , sharex=True)

    axes[0, 0].scatter(X['x1'], y, s=5, alpha=.08)
    axes[0, 0].set_xlim(0, 13)
    axes[0, 0].set_ylim(0, 45)
    axes[0, 1].scatter(X['x2'], y, s=5, alpha=.08)
    axes[0, 1].set_xlim(0, 13)
    axes[0, 1].set_ylim(3, 45)
    axes[0, 2].scatter(X['x3'], y, s=5, alpha=.08)
    axes[0, 2].set_xlim(0, 13)
    axes[0, 2].set_ylim(3, 45)
    axes[0, 3].scatter(X['x4'], y, s=5, alpha=.08)
    axes[0, 3].set_xlim(0, 13)
    axes[0, 3].set_ylim(3, 45)

    axes[0, 0].text(1, 38, 'Marginal', horizontalalignment='left')
    axes[0, 1].text(1, 38, 'Marginal', horizontalalignment='left')
    axes[0, 2].text(1, 38, 'Marginal', horizontalalignment='left')
    axes[0, 3].text(1, 38, 'Marginal', horizontalalignment='left')

    axes[0, 0].set_ylabel("y")

    for i in range(6):
        for j in range(1, 4):
            axes[i, j].get_yaxis().set_visible(False)

    for i in range(6):
        for j in range(4):
            axes[i, j].set_xlim(0, 15)

    pdpx, pdpy, ignored = \
        plot_stratpd(X, y, 'x1', 'y', ax=axes[1, 0], xrange=(0, 13),
                     min_samples_leaf=min_samples_leaf,
                     yrange=yrange, show_xlabel=False, show_ylabel=True)


    r = LinearRegression()
    r.fit(pdpx.reshape(-1, 1), pdpy)
    axes[1, 0].text(1, 10, f"Slope={r.coef_[0]:.2f}")

    pdpx, pdpy, ignored = \
        plot_stratpd(X, y, 'x2', 'y', ax=axes[1, 1], xrange=(0, 13),
                     # show_dx_line=True,
                     min_samples_leaf=min_samples_leaf,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(pdpx.reshape(-1, 1), pdpy)
    axes[1, 1].text(1, 10, f"Slope={r.coef_[0]:.2f}")

    pdpx, pdpy, ignored = \
        plot_stratpd(X, y, 'x3', 'y', ax=axes[1, 2], xrange=(0, 13),
                     # show_dx_line=True,
                     min_samples_leaf=min_samples_leaf,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(pdpx.reshape(-1, 1), pdpy)
    axes[1, 2].text(1, 10, f"Slope={r.coef_[0]:.2f}")

    pdpx, pdpy, ignored = \
        plot_stratpd(X, y, 'x4', 'y', ax=axes[1, 3], xrange=(0, 13),
                     # show_dx_line=True,
                     min_samples_leaf=min_samples_leaf,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(pdpx.reshape(-1, 1), pdpy)
    axes[1, 3].text(1, 10, f"Slope={r.coef_[0]:.2f}")

    axes[1, 0].text(1, 12, 'StratPD', horizontalalignment='left')
    axes[1, 1].text(1, 12, 'StratPD', horizontalalignment='left')
    axes[1, 2].text(1, 12, 'StratPD', horizontalalignment='left')
    axes[1, 3].text(1, 12, 'StratPD', horizontalalignment='left')

    # plt.show()
    # return

    nfeatures = 4
    regrs = [
        RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True),
        svm.SVR(gamma=1 / nfeatures),  # gamma='scale'),
        LinearRegression(),
        KNeighborsRegressor(n_neighbors=5)]
    row = 2
    for regr in regrs:
        regr.fit(X, y)
        rname = regr.__class__.__name__
        if rname == 'SVR':
            rname = "SVM PD/ICE"
        if rname == 'RandomForestRegressor':
            rname = "RF PD/ICE"
        if rname == 'LinearRegression':
            rname = 'Linear PD/ICE'
        if rname == 'KNeighborsRegressor':
            rname = 'kNN PD/ICE'

        show_xlabel = True if row == 5 else False

        axes[row, 0].text(.5, 11, rname, horizontalalignment='left')
        axes[row, 1].text(.5, 11, rname, horizontalalignment='left')
        axes[row, 2].text(.5, 11, rname, horizontalalignment='left')
        axes[row, 3].text(.5, 11, rname, horizontalalignment='left')
        ice = predict_ice(regr, X, 'x1', 'y')
        plot_ice(ice, 'x1', 'y', ax=axes[row, 0], xrange=(0, 13), yrange=yrange,
                 alpha=.08,
                 show_xlabel=show_xlabel, show_ylabel=True)
        ice = predict_ice(regr, X, 'x2', 'y')
        plot_ice(ice, 'x2', 'y', ax=axes[row, 1], xrange=(0, 13), yrange=yrange,
                 alpha=.08,
                 show_xlabel=show_xlabel, show_ylabel=False)
        ice = predict_ice(regr, X, 'x3', 'y')
        plot_ice(ice, 'x3', 'y', ax=axes[row, 2], xrange=(0, 13), yrange=yrange,
                 alpha=.08,
                 show_xlabel=show_xlabel, show_ylabel=False)
        ice = predict_ice(regr, X, 'x4', 'y')
        plot_ice(ice, 'x4', 'y', ax=axes[row, 3], xrange=(0, 13), yrange=yrange,
                 alpha=.08,
                 show_xlabel=show_xlabel, show_ylabel=False)
        row += 1

    # plt.tight_layout()
    # plt.show()
    savefig("multivar_multimodel_normal")


if __name__ == '__main__':
    # FROM PAPER:
    # yearmade()
    # rent()
    # rent_ntrees()
    # unsup_rent()
    # unsup_boston()
    # weight()
    # shap_weight(feature_perturbation='tree_path_dependent', twin=True) # more biased but faster
    # shap_weight(feature_perturbation='interventional', twin=True) # takes 04:45 minutes
    # weight_ntrees()
    # unsup_weight()
    # meta_weight()
    # weather()
    meta_weather()
    additivity()
    meta_additivity()
    bigX()
    multi_joint_distr()

    # EXTRA GOODIES
    # meta_boston()
    # rent_alone()
    # cars()
    # meta_cars()
