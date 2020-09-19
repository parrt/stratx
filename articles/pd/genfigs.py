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
from sklearn.inspection import partial_dependence
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import svm
from sklearn.datasets import load_boston

from articles.pd.support import load_rent, load_bulldozer, load_flights, \
                                toy_weather_data, toy_weight_data, \
                                df_cat_to_catcode, df_split_dates, \
                                df_string_to_cat, synthetic_interaction_data
from stratx import plot_stratpd, plot_catstratpd, \
                           plot_stratpd_gridsearch, plot_catstratpd_gridsearch
from stratx.partdep import partial_dependence
from stratx.plot import marginal_plot_, plot_ice, plot_catice
from stratx.ice import predict_ice, predict_catice, friedman_partial_dependence
import inspect
import matplotlib.patches as mpatches
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

import shap
import xgboost as xgb
from colour import rgb2hex, Color

from dtreeviz.trees import tree, ShadowDecTree

figsize = (2.5, 2)
figsize2 = (3.8, 3.2)
GREY = '#444443'


# This genfigs.py code is just demonstration code to generate figures for the paper.
# There are lots of programming sins committed here; to not take this to be
# our idea of good code. ;)

# For data sources, please see notebooks/examples.ipynb

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
    plt.savefig(f"images/{filename}.pdf", bbox_inches="tight", pad_inches=0)
    # plt.savefig(f"images/{filename}.png", dpi=150)

    plt.tight_layout()
    plt.show()

    plt.close()


def rent():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    np.random.seed(1)  # pick seed for reproducible article images
    X,y = load_rent(n=10_000)
    df_rent = X.copy()
    df_rent['price'] = y
    colname = 'bedrooms'
    colname = 'bathrooms'

    TUNE_RF = False
    TUNE_XGB = False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if TUNE_RF:
        rf, bestparams = tune_RF(X, y)   # does CV on entire data set to tune
        # bedrooms
        # RF best: {'max_features': 0.3, 'min_samples_leaf': 1, 'n_estimators': 125}
        # validation R^2 0.7873724127323822
        # bathrooms
        # RF best: {'max_features': 0.3, 'min_samples_leaf': 1, 'n_estimators': 200}
        # validation R^2 0.8066593395345907
    else:
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, max_features=.3,
                                   oob_score=True, n_jobs=-1)
    rf.fit(X_train, y_train) # Use training set for plotting
    print("RF OOB R^2", rf.oob_score_)
    rf_score = rf.score(X_test, y_test)
    print("RF validation R^2", rf_score)

    if TUNE_XGB:
        tuned_parameters = {'n_estimators': [400, 450, 500, 600, 1000],
                            'learning_rate': [0.008, 0.01, 0.02, 0.05, 0.08, 0.1, 0.11],
                            'max_depth': [3, 4, 5, 6, 7, 8, 9]}
        grid = GridSearchCV(
            xgb.XGBRegressor(), tuned_parameters, scoring='r2',
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        grid.fit(X, y)  # does CV on entire data set to tune
        print("XGB best:", grid.best_params_)
        b = grid.best_estimator_
        # bedrooms
        # XGB best: {'max_depth': 7, 'n_estimators': 250}
        # XGB validation R^2 0.7945797751555217
        # bathrooms
        # XGB best: {'learning_rate': 0.11, 'max_depth': 6, 'n_estimators': 1000}
        # XGB train R^2 0.9834399795800324
        # XGB validation R^2 0.8244958014380593
    else:
        b = xgb.XGBRegressor(n_estimators=1000,
                             max_depth=6,
                             learning_rate=.11,
                             verbose=2,
                             n_jobs=8)

    b.fit(X_train, y_train)
    xgb_score = b.score(X_test, y_test)
    print("XGB validation R^2", xgb_score)

    lm = LinearRegression()
    lm.fit(X_train, y_train)
    lm_score = lm.score(X_test, y_test)
    print("OLS validation R^2", lm_score)
    lm.fit(X, y)

    model, r2_keras = rent_deep_learning_model(X_train, y_train, X_test, y_test)

    fig, axes = plt.subplots(1, 6, figsize=(10, 1.8),
                             gridspec_kw = {'wspace':0.15})

    for i in range(len(axes)):
        axes[i].set_xlim(0-.3,4+.3)
        axes[i].set_xticks([0,1,2,3,4])
        axes[i].set_ylim(1800, 9000)
        axes[i].set_yticks([2000,4000,6000,8000])

    axes[1].get_yaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    axes[3].get_yaxis().set_visible(False)
    axes[4].get_yaxis().set_visible(False)

    axes[0].set_title("(a) Marginal", fontsize=10)

    axes[1].set_title("(b) RF", fontsize=10)
    axes[1].text(2,8000, f"$R^2=${rf_score:.3f}", horizontalalignment='center', fontsize=9)

    axes[2].set_title("(c) XGBoost", fontsize=10)
    axes[2].text(2,8000, f"$R^2=${xgb_score:.3f}", horizontalalignment='center', fontsize=9)

    axes[3].set_title("(d) OLS", fontsize=10)
    axes[3].text(2,8000, f"$R^2=${lm_score:.3f}", horizontalalignment='center', fontsize=9)

    axes[4].set_title("(e) Keras", fontsize=10)
    axes[4].text(2,8000, f"$R^2=${r2_keras:.3f}", horizontalalignment='center', fontsize=9)

    axes[5].set_title("(f) StratPD", fontsize=10)

    avg_per_baths = df_rent.groupby(colname).mean()['price']
    axes[0].scatter(df_rent[colname], df_rent['price'], alpha=0.07, s=5)
    axes[0].scatter(np.unique(df_rent[colname]), avg_per_baths, s=6, c='black',
                       label="average price/{colname}")
    axes[0].set_ylabel("price")  # , fontsize=12)
    axes[0].set_xlabel("bathrooms")
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)

    ice = predict_ice(rf, X, colname, 'price', numx=30, nlines=100)
    plot_ice(ice, colname, 'price', alpha=.3, ax=axes[1], show_xlabel=True,
             show_ylabel=False)

    ice = predict_ice(b, X, colname, 'price', numx=30, nlines=100)
    plot_ice(ice, colname, 'price', alpha=.3, ax=axes[2], show_ylabel=False)

    ice = predict_ice(lm, X, colname, 'price', numx=30, nlines=100)
    plot_ice(ice, colname, 'price', alpha=.3, ax=axes[3], show_ylabel=False)

    scaler = StandardScaler()
    X_train_ = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    # y_pred = model.predict(X_)
    # print("Keras training R^2", r2_score(y, y_pred)) # y_test in y
    ice = predict_ice(model, X_train_, colname, 'price', numx=30, nlines=100)
    # replace normalized unique X with unnormalized
    ice.iloc[0, :] = np.linspace(np.min(X_train[colname]), np.max(X_train[colname]), 30, endpoint=True)
    plot_ice(ice, colname, 'price', alpha=.3, ax=axes[4], show_ylabel=True)

    pdpx, pdpy, ignored = \
        plot_stratpd(X, y, colname, 'price', ax=axes[5],
                     pdp_marker_size=6,
                     show_x_counts=False,
                     hide_top_right_axes=False,
                     show_xlabel=True, show_ylabel=False)
    print(f"StratPD ignored {ignored} records")
    axes[5].yaxis.tick_right()
    axes[5].yaxis.set_label_position('right')
    axes[5].set_ylim(-250,2250)
    axes[5].set_yticks([0,1000,2000])
    axes[5].set_ylabel("price")

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
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # rf.fit(X_train, y_train)
    # print("validation R^2", rf.score(X_test, y_test))
    return rf, grid.best_params_


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
    axes[1, 0].set_title(f"FPD/ICE")

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
    axes[1, 1].set_title(f"FPD/ICE w/{type} col")
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
    axes[1, 0].set_title(f"FPD/ICE")

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
    axes[1, 1].set_title(f"FPD/ICE w/{type} col")
    # print(f"max ICE curve {np.max(pdp_curve):.0f}, max curve with dup {np.max(pdp_curve_):.0f}")

    axes[1, 2].set_title(f"FPD/ICE w/{type} col")
    axes[1, 2].text(.2, 4000, "Cannot compensate")
    axes[1, 2].set_xlabel(colname)

    # print(f"max curve {np.max(curve):.0f}, max curve with dup {np.max(curve_):.0f}")

    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 1].get_xaxis().set_visible(False)


def rent_ntrees():
    np.random.seed(1)  # pick seed for reproducible article images
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
    np.random.seed(1)  # pick seed for reproducible article images
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
    np.random.seed(1)  # pick seed for reproducible article images
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
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X, y = load_rent(n=10_000)

    fig, axes = plt.subplots(4, 2, figsize=(4, 8))

    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 0], yrange=(-500,4000),
                 slope_line_alpha=.2, supervised=False)
    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 1], yrange=(-500,4000),
                 slope_line_alpha=.2, supervised=True)

    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 0], yrange=(-500,4000),
                 slope_line_alpha=.2, supervised=False)
    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 1], yrange=(-500,4000),
                 slope_line_alpha=.2, supervised=True)

    plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 0], yrange=(-500,2000),
                 slope_line_alpha=.2, supervised=False, verbose=True)
    plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 1], yrange=(-500,2000),
                 slope_line_alpha=.2, supervised=True, verbose=True)

    plot_stratpd(X, y, 'longitude', 'price', ax=axes[3, 0], yrange=(-500,500),
                 slope_line_alpha=.2, supervised=False)
    plot_stratpd(X, y, 'longitude', 'price', ax=axes[3, 1], yrange=(-500,500),
                 slope_line_alpha=.2, supervised=True)

    axes[0, 0].set_title("Unsupervised")
    axes[0, 1].set_title("Supervised")

    for i in range(3):
        axes[i, 1].get_yaxis().set_visible(False)

    savefig(f"rent_unsup")
    plt.close()


def weather():
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    TUNE_RF = False
    df_raw = toy_weather_data()
    df = df_raw.copy()

    df_string_to_cat(df)
    names = np.unique(df['state'])
    catnames = OrderedDict()
    for i,v in enumerate(names):
        catnames[i+1] = v
    df_cat_to_catcode(df)

    X = df.drop('temperature', axis=1)
    y = df['temperature']
    # cats = catencoders['state'].values
    # cats = np.insert(cats, 0, None) # prepend a None for catcode 0

    if TUNE_RF:
        rf, bestparams = tune_RF(X, y)
        # RF best: {'max_features': 0.9, 'min_samples_leaf': 5, 'n_estimators': 150}
        # validation R^2 0.9500072628270099
    else:
        rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.9, oob_score=True)
        rf.fit(X, y) # Use full data set for plotting
        print("RF OOB R^2", rf.oob_score_)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df = df_raw.copy()
    avgtmp = df.groupby(['state', 'dayofyear'])[['temperature']].mean()
    avgtmp = avgtmp.reset_index()
    ca = avgtmp.query('state=="CA"')
    co = avgtmp.query('state=="CO"')
    az = avgtmp.query('state=="AZ"')
    wa = avgtmp.query('state=="WA"')
    nv = avgtmp.query('state=="NV"')
    ax.plot(ca['dayofyear'], ca['temperature'], lw=.5, c='#fdae61', label="CA")
    ax.plot(co['dayofyear'], co['temperature'], lw=.5, c='#225ea8', label="CO")
    ax.plot(az['dayofyear'], az['temperature'], lw=.5, c='#41b6c4', label="AZ")
    ax.plot(wa['dayofyear'], wa['temperature'], lw=.5, c='#a1dab4', label="WA")
    ax.plot(nv['dayofyear'], nv['temperature'], lw=.5, c='#a1dab4', label="NV")
    ax.legend(loc='upper left', borderpad=0, labelspacing=0)
    ax.set_xlabel("dayofyear")
    ax.set_ylabel("temperature")
    ax.set_title("(a) State/day vs temp")
    savefig(f"dayofyear_vs_temp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_stratpd(X, y, 'dayofyear', 'temperature', ax=ax,
                 show_x_counts=False,
                 yrange=(-10, 10),
                 pdp_marker_size=2, slope_line_alpha=.5, n_trials=1)

    ax.set_title("(b) StratPD")
    savefig(f"dayofyear_vs_temp_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'state', 'temperature', catnames=catnames,
                    show_x_counts=False,
                    # min_samples_leaf=30,
                    min_y_shifted_to_zero=True,
                    # alpha=.3,
                    ax=ax,
                    yrange=(-1, 55))
    ax.set_yticks([0,10,20,30,40,50])

    ax.set_title("(d) CatStratPD")
    savefig(f"state_vs_temp_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'dayofyear', 'temperature')
    plot_ice(ice, 'dayofyear', 'temperature', ax=ax)
    ax.set_title("(c) FPD/ICE")
    savefig(f"dayofyear_vs_temp_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_catice(rf, X, 'state', 'temperature')
    plot_catice(ice, 'state', 'temperature', catnames=catnames, ax=ax,
                pdp_marker_size=15,
                min_y_shifted_to_zero = True,
                yrange=(-2, 50)
                )
    ax.set_yticks([0,10,20,30,40,50])
    ax.set_title("(b) FPD/ICE")
    savefig(f"state_vs_temp_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(X['state'], y, alpha=.05, s=15)
    ax.set_xticks(range(1,len(names)))
    ax.set_xticklabels(catnames.values())
    ax.set_xlabel("state")
    ax.set_ylabel("temperature")
    ax.set_title("(a) Marginal")
    savefig(f"state_vs_temp")

    plt.close()


def meta_weather():
    np.random.seed(1)  # pick seed for reproducible article images
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
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X, y, df_raw, eqn = toy_weight_data(2000)

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
    ax.set_xticks([60,65,70,75])
    savefig(f"height_vs_weight_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=(1.3,2))
    plot_catstratpd(X, y, 'sex', 'weight', ax=ax,
                    show_x_counts=False,
                    catnames={0:'M',1:'F'},
                    yrange=(-1, 35),
                    )
    ax.set_title("CatStratPD", fontsize=10)
    savefig(f"sex_vs_weight_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=(1.5,1.8))
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=ax,
                    show_x_counts=False,
                    catnames={0:False, 1:True},
                    yrange=(-1, 45),
                    )
    ax.set_title("CatStratPD", fontsize=10)
    savefig(f"pregnant_vs_weight_stratpd")

    if TUNE_RF:
        rf, bestparams = tune_RF(X, y)
        # RF best: {'max_features': 0.9, 'min_samples_leaf': 1, 'n_estimators': 200}
        # validation R^2 0.9996343699640691
    else:
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, max_features=0.9, oob_score=True)
        rf.fit(X, y) # Use full data set for plotting
        print("RF OOB R^2", rf.oob_score_)

    # show pregnant female at max range drops going taller
    X_test = np.array([[1, 1, 70, 10]])
    y_pred = rf.predict(X_test)
    print("pregnant female at max range", X_test, "predicts", y_pred)
    X_test = np.array([[1, 1, 72, 10]]) # make them taller
    y_pred = rf.predict(X_test)
    print("pregnant female in male height range", X_test, "predicts", y_pred)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'education', 'weight')
    plot_ice(ice, 'education', 'weight', ax=ax, yrange=(-12, 0), min_y_shifted_to_zero=True)
    ax.set_xlim(10,18)
    ax.set_xticks([10,12,14,16,18])
    ax.set_title("FPD/ICE", fontsize=10)
    savefig(f"education_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=(2.4, 2.2))
    ice = predict_ice(rf, X, 'height', 'weight')
    plot_ice(ice, 'height', 'weight', ax=ax, pdp_linewidth=2, yrange=(100, 250),
             min_y_shifted_to_zero=False)
    ax.set_xlabel("height\n(a)", fontsize=10)
    ax.set_ylabel("weight", fontsize=10)
    ax.set_title("FPD/ICE", fontsize=10)
    ax.set_xticks([60,65,70,75])
    savefig(f"height_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=(1.3,2))
    ice = predict_catice(rf, X, 'sex', 'weight')
    plot_catice(ice, 'sex', 'weight', catnames={0:'M',1:'F'}, ax=ax, yrange=(0, 35),
                pdp_marker_size=15)
    ax.set_title("FPD/ICE", fontsize=10)
    savefig(f"sex_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=(1.3,1.8))
    ice = predict_catice(rf, X, 'pregnant', 'weight', cats=df_raw['pregnant'].unique())
    plot_catice(ice, 'pregnant', 'weight', catnames={0:'M',1:'F'}, ax=ax,
                min_y_shifted_to_zero=True,
                yrange=(-5, 45), pdp_marker_size=20)
    ax.set_title("FPD/ICE", fontsize=10)
    savefig(f"pregnant_vs_weight_pdp")


def shap_pregnant():
    np.random.seed(1)  # pick seed for reproducible article images
    n = 2000
    shap_test_size = 300
    X, y, df_raw, eqn = toy_weight_data(n=n)
    df = df_raw.copy()
    df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    # parameters from tune_RF() called in weight()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1,
                               max_features=0.9,
                               oob_score=True)
    rf.fit(X, y)  # Use full data set for plotting
    print("RF OOB R^2", rf.oob_score_)

    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100),
                                   feature_perturbation='interventional')
    shap_sample = X.sample(shap_test_size, replace=False)
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    GREY = '#444443'
    fig, ax = plt.subplots(1, 1, figsize=(1.3,1.8))

    preg_shap_values = shap_values[:, 1]
    avg_not_preg_weight = np.mean(preg_shap_values[np.where(shap_sample['pregnant']==0)])
    avg_preg_weight = np.mean(preg_shap_values[np.where(shap_sample['pregnant']==1)])
    ax.bar([0, 1], [avg_not_preg_weight-avg_not_preg_weight, avg_preg_weight-avg_not_preg_weight],
           color='#1E88E5')
    ax.set_title("SHAP", fontsize=10)
    ax.set_xlabel("pregnant")
    ax.set_xticks([0,1])
    ax.set_xticklabels(['False','True'])
    ax.set_ylabel("weight")
    ax.set_ylim(-1,45)
    ax.set_yticks([0,10,20,30,40])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    savefig('pregnant_vs_weight_shap')


def shap_weight(feature_perturbation, twin=False):
    np.random.seed(1)  # pick seed for reproducible article images
    n = 2000
    shap_test_size = 2000
    X, y, df_raw, eqn = toy_weight_data(n=n)
    df = df_raw.copy()
    df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    # parameters from tune_RF() called in weight()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1,
                               max_features=0.9,
                               oob_score=True)
    rf.fit(X, y)  # Use full data set for plotting
    print("RF OOB R^2", rf.oob_score_)

    if feature_perturbation=='interventional':
        explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100), feature_perturbation='interventional')
        xlabel = "height\n(c)"
        ylabel = None
        yticks = []
        figsize = (2.2, 2.2)
    else:
        explainer = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
        xlabel = "height\n(b)"
        ylabel = "SHAP height"
        yticks = [-75, -60, -40, -20, 0, 20, 40, 60, 75]
        figsize = (2.6, 2.2)
    shap_sample = X.sample(shap_test_size, replace=False)
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    df_shap = pd.DataFrame()
    df_shap['weight'] = shap_values[:, 2]
    df_shap['height'] = shap_sample.iloc[:, 2]

    # pdpy = df_shap.groupby('height').mean().reset_index()
    # print("len pdpy", len(pdpy))

    GREY = '#444443'
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    shap.dependence_plot("height", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=1)
    # ax.plot(pdpy['height'], pdpy['weight'], '.', c='k', markersize=.5, alpha=.5)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['top'].set_linewidth(.5)

    ax.set_ylabel(ylabel, fontsize=10, labelpad=0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.plot([70,70], [-75,75], '--', lw=.6, color=GREY)
    ax.text(69.8,60, "Max female", horizontalalignment='right',
            fontsize=9)

    leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
        partial_dependence(X=X, y=y, colname='height')

    ax.set_ylim(-77,75)
    # ax.set_xlim(min(pdpx), max(pdpx))
    ax.set_xticks([60,65,70,75])
    ax.set_yticks(yticks)

    ax.set_title(f"SHAP {feature_perturbation}", fontsize=10)
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
        # ax2.set_ylabel("weight", fontsize=12)

        ax2.plot(pdpx, pdpy, '.', markersize=1, c='k')
        # ax2.text(65,25, f"StratPD slope = {slope:.1f}")
        ax2.annotate(f"StratPD", (64.65,39), xytext=(66,18),
                     horizontalalignment='left',
                     arrowprops=dict(facecolor='black', width=.5, headwidth=5, headlength=5),
                     fontsize=9)

    savefig(f"weight_{feature_perturbation}_shap")


def saledayofweek():
    np.random.seed(1)  # pick seed for reproducible article images
    n = 10_000
    shap_test_size = 1000
    TUNE_RF = False
    X, y = load_bulldozer(n=n)

    avgprice = pd.concat([X,y], axis=1).groupby('saledayofweek')[['SalePrice']].mean()
    avgprice = avgprice.reset_index()['SalePrice']
    print(avgprice)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.scatter(range(0,7), avgprice, s=20, c='k')
    ax.scatter(X['saledayofweek'], y, s=3, alpha=.1, c='#1E88E5')
    # ax.set_xlim(1960,2010)
    ax.set_xlabel("saledayofweek\n(a)", fontsize=11)
    ax.set_ylabel("SalePrice ($)", fontsize=11)
    ax.set_title("Marginal plot", fontsize=13)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    savefig(f"bulldozer_saledayofweek_marginal")

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
    shap_sample = X.sample(shap_test_size, replace=False)
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    shap.dependence_plot("saledayofweek", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=.5)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_title("SHAP", fontsize=13)
    ax.set_ylabel("Impact on SalePrice\n(saledayofweek SHAP)", fontsize=11)
    ax.set_xlabel("saledayofweek\n(b)", fontsize=11)
    # ax.set_xlim(1960, 2010)
    ax.tick_params(axis='both', which='major', labelsize=10)

    savefig(f"bulldozer_saledayofweek_shap")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    plot_catstratpd(X, y, colname='saledayofweek', targetname='SalePrice',
                    catnames={0:'M',1:'T',2:'W',3:'R',4:'F',5:'S',6:'S'},
                 n_trials=1,
                 bootstrap=True,
                 show_x_counts=True,
                 show_xlabel=False,
                 show_impact=False,
                 pdp_marker_size=4,
                 pdp_marker_alpha=1,
                 ax=ax
                 )
    ax.set_title("StratPD", fontsize=13)
    ax.set_xlabel("saledayofweek\n(d)", fontsize=11)
    # ax.set_xlim(1960,2010)
    # ax.set_ylim(-10000,30_000)
    savefig(f"bulldozer_saledayofweek_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ice = predict_ice(rf, X, "saledayofweek", 'SalePrice', numx=30, nlines=100)
    plot_ice(ice, "saledayofweek", 'SalePrice', alpha=.3, ax=ax, show_ylabel=True,
#             yrange=(-10000,30_000),
             min_y_shifted_to_zero=True)
    # ax.set_xlim(1960, 2010)
    savefig(f"bulldozer_saledayofweek_pdp")


def productsize():
    np.random.seed(1)  # pick seed for reproducible article images
    shap_test_size = 1000
    TUNE_RF = False

    # reuse same data generated by gencsv.py for bulldozer to
    # make same comparison.
    df = pd.read_csv("bulldozer20k.csv")
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.scatter(X['ProductSize'], y, s=3, alpha=.1, c='#1E88E5')
    # ax.set_xlim(1960,2010)
    ax.set_xlabel("ProductSize\n(a)", fontsize=11)
    ax.set_ylabel("SalePrice ($)", fontsize=11)
    ax.set_title("Marginal plot", fontsize=13)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    savefig(f"bulldozer_ProductSize_marginal")

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

    # SHAP
    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100),
                                   feature_perturbation='interventional')
    shap_sample = X.sample(shap_test_size, replace=False)
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    shap.dependence_plot("ProductSize", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.set_title("(b) SHAP", fontsize=13)
    ax.set_ylabel("Impact on SalePrice\n(ProductSize SHAP)", fontsize=11)
    ax.set_xlabel("ProductSize", fontsize=11)
    # ax.set_xlim(1960, 2010)
    ax.set_ylim(-15000,40_000)
    ax.tick_params(axis='both', which='major', labelsize=10)
    savefig(f"bulldozer_ProductSize_shap")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    plot_stratpd(X, y, colname='ProductSize', targetname='SalePrice',
                 n_trials=10,
                 bootstrap=True,
                 show_slope_lines=False,
                 show_x_counts=False,
                 show_xlabel=False,
                 show_impact=False,
                 show_all_pdp=False,
                 pdp_marker_size=10,
                 pdp_marker_alpha=1,
                 ax=ax
                 )
    ax.set_title("(d) StratPD", fontsize=13)
    ax.set_xlabel("ProductSize", fontsize=11)
    ax.set_xlim(0, 5)
    ax.set_ylim(-15000,40_000)
    savefig(f"bulldozer_ProductSize_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ice = predict_ice(rf, X, "ProductSize", 'SalePrice', numx=30, nlines=100)
    plot_ice(ice, "ProductSize", 'SalePrice', alpha=.3, ax=ax, show_ylabel=True,
#             yrange=(-10000,30_000),
             min_y_shifted_to_zero=True)
    # ax.set_xlim(1960, 2010)
    ax.set_ylim(-15000,40_000)
    ax.set_title("(a) FPD/ICE plot", fontsize=13)
    savefig(f"bulldozer_ProductSize_pdp")


def saledayofyear():
    np.random.seed(1)  # pick seed for reproducible article images
    n = 10_000
    shap_test_size = 1000
    TUNE_RF = False
    X, y = load_bulldozer(n=n)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.scatter(X['saledayofyear'], y, s=3, alpha=.1, c='#1E88E5')
    # ax.set_xlim(1960,2010)
    ax.set_xlabel("saledayofyear\n(a)", fontsize=11)
    ax.set_ylabel("SalePrice ($)", fontsize=11)
    ax.set_title("Marginal plot", fontsize=13)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    savefig(f"bulldozer_saledayofyear_marginal")

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
    shap_sample = X.sample(shap_test_size, replace=False)
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    shap.dependence_plot("saledayofyear", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=.5)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_title("SHAP", fontsize=13)
    ax.set_ylabel("Impact on SalePrice\n(saledayofyear SHAP)", fontsize=11)
    ax.set_xlabel("saledayofyear\n(b)", fontsize=11)
    # ax.set_xlim(1960, 2010)
    ax.tick_params(axis='both', which='major', labelsize=10)

    savefig(f"bulldozer_saledayofyear_shap")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    plot_stratpd(X, y, colname='saledayofyear', targetname='SalePrice',
                 n_trials=10,
                 bootstrap=True,
                 show_all_pdp=False,
                 show_slope_lines=False,
                 show_x_counts=True,
                 show_xlabel=False,
                 show_impact=False,
                 pdp_marker_size=4,
                 pdp_marker_alpha=1,
                 ax=ax
                 )
    ax.set_title("StratPD", fontsize=13)
    ax.set_xlabel("saledayofyear\n(d)", fontsize=11)
    # ax.set_xlim(1960,2010)
    # ax.set_ylim(-10000,30_000)
    savefig(f"bulldozer_saledayofyear_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ice = predict_ice(rf, X, "saledayofyear", 'SalePrice', numx=30, nlines=100)
    plot_ice(ice, "saledayofyear", 'SalePrice', alpha=.3, ax=ax, show_ylabel=True,
#             yrange=(-10000,30_000),
             min_y_shifted_to_zero=True)
    # ax.set_xlim(1960, 2010)
    savefig(f"bulldozer_saledayofyear_pdp")


def yearmade():
    np.random.seed(1)  # pick seed for reproducible article images
    n = 20_000
    shap_test_size = 1000
    TUNE_RF = False

    # X, y = load_bulldozer(n=n)

    # reuse same data generated by gencsv.py for bulldozer to
    # make same comparison.
    df = pd.read_csv("bulldozer20k.csv")
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

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

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.scatter(X['YearMade'], y, s=3, alpha=.1, c='#1E88E5')
    ax.set_xlim(1960,2010)
    ax.set_xlabel("YearMade", fontsize=11)
    ax.set_ylabel("SalePrice ($)", fontsize=11)
    ax.set_title("(a) Marginal plot", fontsize=13)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    savefig(f"bulldozer_YearMade_marginal")

    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100),
                                   feature_perturbation='interventional')
    shap_sample = X.sample(shap_test_size, replace=False)
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    shap.dependence_plot("YearMade", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=.5)
    ax.yaxis.label.set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_title("(b) SHAP", fontsize=13)
    ax.set_ylabel("Impact on SalePrice\n(YearMade SHAP)", fontsize=11)
    ax.set_xlabel("YearMade", fontsize=11)
    ax.set_xlim(1960, 2010)
    ax.tick_params(axis='both', which='major', labelsize=10)

    savefig(f"bulldozer_YearMade_shap")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    plot_stratpd(X, y, colname='YearMade', targetname='SalePrice',
                 n_trials=10,
                 bootstrap=True,
                 show_slope_lines=False,
                 show_x_counts=True,
                 show_ylabel=False,
                 show_xlabel=False,
                 show_impact=False,
                 pdp_marker_size=4,
                 pdp_marker_alpha=1,
                 ax=ax
                 )
    ax.set_title("(d) StratPD", fontsize=13)
    ax.set_xlabel("YearMade", fontsize=11)
    ax.set_xlim(1960,2010)
    ax.set_ylim(-5000,30_000)
    savefig(f"bulldozer_YearMade_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ice = predict_ice(rf, X, "YearMade", 'SalePrice', numx=30, nlines=100)
    plot_ice(ice, "YearMade", 'SalePrice', alpha=.3, ax=ax, show_ylabel=True,
             yrange=(20_000,55_000))
    ax.set_xlabel("YearMade", fontsize=11)
    ax.set_xlim(1960, 2010)
    ax.set_title("(a) FPD/ICE plot", fontsize=13)
    savefig(f"bulldozer_YearMade_pdp")


def MachineHours():
    np.random.seed(1)  # pick seed for reproducible article images
    shap_test_size = 1000
    TUNE_RF = False

    # reuse same data generated by gencsv.py for bulldozer to
    # make same comparison.
    df = pd.read_csv("bulldozer20k.csv")

    # DROP RECORDS WITH MISSING MachineHours VALUES
    # df = df[df['MachineHours']!=3138]

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

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

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.scatter(X['MachineHours'], y, s=3, alpha=.1, c='#1E88E5')
    ax.set_xlim(0,30_000)
    ax.set_xlabel("MachineHours\n(a)", fontsize=11)
    ax.set_ylabel("SalePrice ($)", fontsize=11)
    ax.set_title("Marginal plot", fontsize=13)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    savefig(f"bulldozer_MachineHours_marginal")

    # SHAP
    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100),
                                   feature_perturbation='interventional')
    shap_sample = X.sample(shap_test_size, replace=False)
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    shap.dependence_plot("MachineHours", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=.5)
    ax.yaxis.label.set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_title("SHAP", fontsize=13)
    ax.set_ylabel("SHAP MachineHours)", fontsize=11)
    ax.set_xlabel("MachineHours\n(b)", fontsize=11)
    ax.set_xlim(0,30_000)
    ax.set_ylim(-3000,5000)
    ax.tick_params(axis='both', which='major', labelsize=10)
    savefig(f"bulldozer_MachineHours_shap")

    # STRATPD
    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    plot_stratpd(X, y, colname='MachineHours', targetname='SalePrice',
                 n_trials=10,
                 bootstrap=True,
                 show_all_pdp=False,
                 show_slope_lines=False,
                 show_x_counts=True,
                 barchar_alpha=1.0,
                 barchar_color='k',
                 show_ylabel=False,
                 show_xlabel=False,
                 show_impact=False,
                 pdp_marker_size=1,
                 pdp_marker_alpha=.3,
                 ax=ax
                 )
    # ax.annotate("Imputed median value", xytext=(10000,-5300),
    #             xy=(3138,-5200), fontsize=9,
    #             arrowprops={'arrowstyle':"->"})
    ax.yaxis.label.set_visible(False)
    ax.set_title("StratPD", fontsize=13)
    ax.set_xlim(0,30_000)
    ax.set_xlabel("MachineHours\n(d)", fontsize=11)
    ax.set_ylim(-6500,2_000)
    savefig(f"bulldozer_MachineHours_stratpd")

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ice = predict_ice(rf, X, "MachineHours", 'SalePrice', numx=300, nlines=200)
    plot_ice(ice, "MachineHours", 'SalePrice', alpha=.5, ax=ax,
             show_ylabel=True,
             yrange=(33_000,38_000)
             )
    ax.set_xlabel("MachineHours\n(a)", fontsize=11)
    ax.set_title("FPD/ICE plot", fontsize=13)
    ax.set_xlim(0,30_000)
    savefig(f"bulldozer_MachineHours_pdp")


def unsup_yearmade():
    np.random.seed(1)  # pick seed for reproducible article images
    n = 10_000
    X, y = load_bulldozer(n=n)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    plot_stratpd(X, y, colname='YearMade', targetname='SalePrice',
                 n_trials=1,
                 bootstrap=True,
                 show_slope_lines=False,
                 show_x_counts=True,
                 show_xlabel=False,
                 show_impact=False,
                 pdp_marker_size=4,
                 pdp_marker_alpha=1,
                 ax=ax,
                 supervised=False
                 )
    ax.set_title("Unsupervised StratPD", fontsize=13)
    ax.set_xlabel("YearMade", fontsize=11)
    ax.set_xlim(1960,2010)
    ax.set_ylim(-10000,30_000)
    savefig(f"bulldozer_YearMade_stratpd_unsup")


def unsup_weight():
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X, y, df_raw, eqn = toy_weight_data(2000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 0],
                 show_x_counts=False,
                 yrange=(-13, 0), slope_line_alpha=.1, supervised=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 1],
                 show_x_counts=False,
                 yrange=(-13, 0), slope_line_alpha=.1, supervised=True)

    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 0],
                    show_x_counts=False,
                    catnames=df_raw['pregnant'].unique(),
                    yrange=(-5, 45))
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 1],
                    show_x_counts=False,
                    catnames=df_raw['pregnant'].unique(),
                    yrange=(-5, 45))

    axes[0, 0].set_title("Unsupervised")
    axes[0, 1].set_title("Supervised")

    axes[0, 1].get_yaxis().set_visible(False)
    axes[1, 1].get_yaxis().set_visible(False)

    savefig(f"weight_unsup")
    plt.close()


def weight_ntrees():
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X, y, df_raw, eqn = toy_weight_data(1000)
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
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    X, y, df_raw, eqn = toy_weight_data(1000)
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


def noisy_poly_data(n, sd=1.0):
    x1 = np.random.uniform(-2, 2, size=n)
    x2 = np.random.uniform(-2, 2, size=n)

    y = x1 ** 2 + x2 + 10 + np.random.normal(0, sd, size=n)
    df = pd.DataFrame()
    df['x1'] = x1
    df['x2'] = x2
    df['y'] = y
    return df


def noise():
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000

    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

    sds = [0,.5,1,2]

    for i,sd in enumerate(sds):
        df = noisy_poly_data(n=n, sd=sd)
        X = df.drop('y', axis=1)
        y = df['y']
        plot_stratpd(X, y, 'x1', 'y',
                     show_ylabel=False,
                     pdp_marker_size=1,
                     show_x_counts=False,
                     ax=axes[i], yrange=(-4, .5))
    axes[0].set_ylabel("y", fontsize=12)

    for i,(ax,which) in enumerate(zip(axes,['(a)','(b)','(c)','(d)'])):
        ax.text(0, -1, f"{which}\n$\sigma = {sds[i]}$", horizontalalignment='center')
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_xticks([-2,-1,0,1,2])

    savefig(f"noise")


def meta_noise():
    np.random.seed(1)  # pick seed for reproducible article images
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    noises = [0, .5, .8, 1.0]
    sizes = [2, 10, 30, 50]

    fig, axes = plt.subplots(len(noises), len(sizes), figsize=(7, 6), sharey=True,
                             sharex=True)

    row = 0
    for sd in noises:
        df = noisy_poly_data(n=n, sd=sd)
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
                         show_x_counts=False,
                         min_samples_leaf=s,
                         yrange=(-3.5, .5),
                         pdp_marker_size=1,
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

    # row = 0
    # for sd in noises:
    #     axes[row, 0].scatter(X['x1'], y, slope_line_alpha=.12, label=None)
    #     axes[row, 0].set_xlabel("x1")
    #     axes[row, 0].set_ylabel("y")
    #     axes[row, 0].set_ylim(-5, 5)
    #     axes[row, 0].set_title(f"$y = x_1^2 + x_2 + \epsilon$, $\epsilon \sim N(0,{sd:.2f})$")
    #     row += 1

    # axes[lastrow, 0].set_ylabel(f'$y$ vs $x_c$ partition')
    # col = 0
    # for s in sizes:
    #     rtreeviz_univar(axes[lastrow, col],
    #                     X['x2'], y,
    #                     min_samples_leaf=s,
    #                     feature_name='x2',
    #                     target_name='y',
    #                     fontsize=10, show={'splits'},
    #                     split_linewidth=.5,
    #                     markersize=5)
    #     axes[lastrow, col].set_xlabel("x2")
    #     col += 1

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
    np.random.seed(1)  # pick seed for reproducible article images
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
    axes[0, 1].set_title("FPD/ICE", fontsize=10)

    savefig(f"bigx")
    plt.close()


def unsup_boston():
    np.random.seed(1)  # pick seed for reproducible article images
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
    axes[3].set_title("FPD/ICE")

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
    np.random.seed(1)  # pick seed for reproducible article images
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
    np.random.seed(1)  # pick seed for reproducible article images
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


def multi_joint_distr():
    np.random.seed(1)  # pick seed for reproducible article images
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
            rname = "SVM FPD/ICE"
        if rname == 'RandomForestRegressor':
            rname = "RF FPD/ICE"
        if rname == 'LinearRegression':
            rname = 'Linear FPD/ICE'
        if rname == 'KNeighborsRegressor':
            rname = 'kNN FPD/ICE'

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


def interactions():
    np.random.seed(1)  # pick seed for reproducible article images
    n = 2000
    df = synthetic_interaction_data(n)

    X, y = df[['x1', 'x2', 'x3']].copy(), df['y'].copy()
    X1 = X.iloc[:, 0]
    X2 = X.iloc[:, 1]
    X3 = X.iloc[:, 2] # UNUSED in y

    rf = RandomForestRegressor(n_estimators=10, oob_score=True)
    rf.fit(X, y)
    print("R^2 training", rf.score(X, y))
    print("R^2 OOB", rf.oob_score_)

    print("mean(y) =", np.mean(y))
    print("mean(X_1), mean(X_2) =", np.mean(X1), np.mean(X2))

    pdp_x1 = friedman_partial_dependence(rf, X, 'x1', numx=None, mean_centered=False)
    pdp_x2 = friedman_partial_dependence(rf, X, 'x2', numx=None, mean_centered=False)
    pdp_x3 = friedman_partial_dependence(rf, X, 'x3', numx=None, mean_centered=False)
    m1 = np.mean(pdp_x1[1])
    m2 = np.mean(pdp_x2[1])
    m3 = np.mean(pdp_x3[1])
    print("mean(PDP_1) =", np.mean(pdp_x1[1]))
    print("mean(PDP_2) =", np.mean(pdp_x2[1]))
    print("mean(PDP_2) =", np.mean(pdp_x3[1]))

    print("mean abs PDP_1-ybar", np.mean(np.abs(pdp_x1[1] - m1)))
    print("mean abs PDP_2-ybar", np.mean(np.abs(pdp_x2[1] - m2)))
    print("mean abs PDP_3-ybar", np.mean(np.abs(pdp_x3[1] - m3)))

    explainer = shap.TreeExplainer(rf, data=X,
                                   feature_perturbation='interventional')
    shap_values = explainer.shap_values(X, check_additivity=False)
    shapavg = np.mean(shap_values, axis=0)
    print("SHAP avg x1,x2,x3 =", shapavg)
    shapimp = np.mean(np.abs(shap_values), axis=0)
    print("SHAP avg |x1|,|x2|,|x3| =", shapimp)

    fig, axes = plt.subplots(1,4,figsize=(11.33,2.8))

    x1_color = '#1E88E5'
    x2_color = 'orange'
    x3_color = '#A22396'

    axes[0].plot(pdp_x1[0], pdp_x1[1], '.', markersize=1, c=x1_color, label='$FPD_1$', alpha=1)
    axes[0].plot(pdp_x2[0], pdp_x2[1], '.', markersize=1, c=x2_color, label='$FPD_2$', alpha=1)
    axes[0].plot(pdp_x3[0], pdp_x3[1], '.', markersize=1, c=x3_color, label='$FPD_3$', alpha=1)

    axes[0].text(0, 75, f"$\\bar{{y}}={np.mean(y):.1f}$", fontsize=13)
    axes[0].set_xticks([0,2,4,6,8,10])
    axes[0].set_xlabel("$x_1, x_2, x_3$", fontsize=10)
    axes[0].set_ylabel("y")
    axes[0].set_yticks([0, 25, 50, 75, 100, 125, 150])
    axes[0].set_ylim(-10,160)
    axes[0].set_title(f"(a) Friedman FPD")

    axes[0].spines['top'].set_linewidth(.5)
    axes[0].spines['right'].set_linewidth(.5)
    axes[0].spines['left'].set_linewidth(.5)
    axes[0].spines['bottom'].set_linewidth(.5)
    axes[0].spines['top'].set_color('none')
    axes[0].spines['right'].set_color('none')

    x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
    x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
    x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
    axes[0].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=10)

    # axes[0].legend(fontsize=10)

    #axes[1].plot(shap_values)
    shap.dependence_plot("x1", shap_values, X,
                         interaction_index=None, ax=axes[1], dot_size=4,
                         show=False, alpha=.5, color=x1_color)
    shap.dependence_plot("x2", shap_values, X,
                         interaction_index=None, ax=axes[1], dot_size=4,
                         show=False, alpha=.5, color=x2_color)
    shap.dependence_plot("x3", shap_values, X,
                         interaction_index=None, ax=axes[1], dot_size=4,
                         show=False, alpha=.5, color=x3_color)
    axes[1].set_xticks([0,2,4,6,8,10])
    axes[1].set_xlabel("$x_1, x_2, x_3$", fontsize=12)
    axes[1].set_ylim(-95,110)
    axes[1].set_title("(b) SHAP")
    axes[1].set_ylabel("SHAP values", fontsize=11)
    x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
    x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
    x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
    axes[1].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=12)

    df_x1 = pd.read_csv("images/x1_ale.csv")
    df_x2 = pd.read_csv("images/x2_ale.csv")
    df_x3 = pd.read_csv("images/x3_ale.csv")
    axes[2].plot(df_x1['x.values'],df_x1['f.values'],'.',color=x1_color,markersize=2)
    axes[2].plot(df_x2['x.values'],df_x2['f.values'],'.',color=x2_color,markersize=2)
    axes[2].plot(df_x3['x.values'],df_x3['f.values'],'.',color=x3_color,markersize=2)
    axes[2].set_title("(c) ALE")
    # axes[2].set_ylabel("y", fontsize=12)
    axes[2].set_xlabel("$x_1, x_2, x_3$", fontsize=12)
    axes[2].set_ylim(-95,110)
    # axes[2].tick_params(axis='both', which='major', labelsize=10)
    axes[2].set_xticks([0,2,4,6,8,10])
    axes[2].spines['top'].set_linewidth(.5)
    axes[2].spines['right'].set_linewidth(.5)
    axes[2].spines['left'].set_linewidth(.5)
    axes[2].spines['bottom'].set_linewidth(.5)
    axes[2].spines['top'].set_color('none')
    axes[2].spines['right'].set_color('none')
    x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
    x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
    x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
    axes[2].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=12)

    plot_stratpd(X, y, "x1", "y", ax=axes[3], pdp_marker_size=1,
                 pdp_marker_color=x1_color,
                 show_x_counts=False, n_trials=1, show_slope_lines=False)
    plot_stratpd(X, y, "x2", "y", ax=axes[3], pdp_marker_size=1,
                 pdp_marker_color=x2_color,
                 show_x_counts=False, n_trials=1, show_slope_lines=False)
    plot_stratpd(X, y, "x3", "y", ax=axes[3], pdp_marker_size=1,
                 pdp_marker_color=x3_color,
                 show_x_counts=False, n_trials=1, show_slope_lines=False)
    axes[3].set_xticks([0,2,4,6,8,10])
    axes[3].set_ylim(-20,160)
    axes[3].set_yticks([0, 25, 50, 75, 100, 125, 150])
    axes[3].set_xlabel("$x_1, x_2, x_3$", fontsize=12)
    # axes[3].set_ylabel("y", fontsize=12)
    axes[3].set_title("(d) StratPD")
    axes[3].spines['top'].set_linewidth(.5)
    axes[3].spines['right'].set_linewidth(.5)
    axes[3].spines['left'].set_linewidth(.5)
    axes[3].spines['bottom'].set_linewidth(.5)
    axes[3].spines['top'].set_color('none')
    axes[3].spines['right'].set_color('none')
    x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
    x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
    x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
    axes[3].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=12)

    savefig("interactions")


def gen_ale_plot_data_in_R():
    "Exec R and generate images/*.csv files.  Then plot with Python"
    os.system("R CMD BATCH ale_plots_bulldozer.R")
    os.system("R CMD BATCH ale_plots_rent.R")
    os.system("R CMD BATCH ale_plots_weather.R")
    os.system("R CMD BATCH ale_plots_weight.R")


def ale_yearmade():
    df = pd.read_csv("images/YearMade_ale.csv")
    # df['f.values'] -= np.min(df['f.values'])
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.plot(df['x.values'],df['f.values'],'.',color='k',markersize=4)
    ax.set_title("(c) ALE", fontsize=13)
    ax.set_xlabel("YearMade", fontsize=11)
    ax.set_xlim(1960, 2010)
    ax.set_ylim(-25000,30000)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    savefig('bulldozer_YearMade_ale')


def ale_MachineHours():
    df = pd.read_csv("images/MachineHours_ale.csv")
    # df['f.values'] -= np.min(df['f.values'])
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.plot(df['x.values'],df['f.values'],'.',color='k',markersize=4)
    ax.set_title("ALE", fontsize=13)
    # ax.set_ylabel("SalePrice", fontsize=11)
    ax.set_xlabel("(c) MachineHours", fontsize=11)
    ax.set_xlim(0, 30_000)
    ax.set_ylim(-3000,5000)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    savefig('bulldozer_MachineHours_ale')


def ale_productsize():
    df = pd.read_csv("images/ProductSize_ale.csv")
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=figsize2)
    ax.plot(df['x.values'],df['f.values'],'.',color='k',markersize=10)
    ax.set_title("(c) ALE", fontsize=13)
    # ax.set_ylabel("SalePrice", fontsize=11)
    ax.set_xlabel("ProductSize", fontsize=11)
    # ax.set_xlim(0, 30_000)
    ax.set_ylim(-15000,40000)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    savefig('bulldozer_ProductSize_ale')


def ale_height():
    df = pd.read_csv("images/height_ale.csv")
    # df['f.values'] -= np.min(df['f.values'])
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=(2.4, 2.2))
    ax.plot(df['x.values'],df['f.values'],'.',color='k',markersize=1)
    # ax.set_ylim(-5,150)
    ax.set_ylim(-65,90)
    # ax.set_yticks([0,20,40,60,80,100,120,140,150])
    ax.set_title("ALE", fontsize=10)
    ax.set_ylabel("Weight", fontsize=10, labelpad=0)
    ax.set_xlabel("Height\n(d)", fontsize=10)
    ax.set_xticks([60,65,70,75])
    ax.set_yticks([-75, -60, -40, -20, 0, 20, 40, 60, 75])

    ax.tick_params(axis='both', which='major', labelsize=10)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    savefig('height_ale')


def ale_state():
    df = pd.read_csv("images/state_ale.csv")
    df = df.sort_values(by="x.values")
    df['f.values'] -= np.min(df['f.values'])
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(df['x.values'],df['f.values'],color='#BEBEBE')
    ax.set_title("ALE", fontsize=13)
    ax.set_xlabel("state")
    ax.set_ylabel("temperature")
    ax.set_title("(c) ALE")
    ax.set_ylim(0,55)
    ax.set_yticks([0,10,20,30,40,50])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    savefig('state_ale')


def ale_pregnant():
    df = pd.read_csv("images/pregnant_ale.csv")
    df['x.values'] = df['x.values'].map({0:"False",1:"True"})
    df['f.values'] -= np.min(df['f.values'])
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=(1.3,1.8))
    ax.bar(df['x.values'],df['f.values'],color='#BEBEBE')
    ax.set_title("ALE", fontsize=10)
    ax.set_xlabel("pregnant")
    ax.set_ylabel("weight")
    ax.set_title("ALE")
    ax.set_ylim(-1,45)
    ax.set_yticks([0,10,20,30,40])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    savefig('pregnant_2_ale')


def rent_deep_learning_model(X_train, y_train, X_test, y_test):
    np.random.seed(1)  # pick seed for reproducible article images
    from tensorflow.keras import models, layers, callbacks, optimizers

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    # for colname in X.columns:
    #     m = np.mean(X[colname])
    #     sd = np.std(X[colname])
    #     X[colname] = (X[colname]-m)/sd

    #y = (y - np.mean(y))/np.std(y)

    model = models.Sequential()
    layer1 = 100
    batch_size = 1000
    dropout = 0.3
    model.add(layers.Dense(layer1, input_dim=X_train.shape[1], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='linear'))

    # learning_rate=1e-2 #DEFAULT
    opt = optimizers.SGD()  # SGB gets NaNs?
    # opt = optimizers.RMSprop(lr=0.1)
    opt = optimizers.Adam(lr=0.3)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

    callback = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train,
                        # epochs=1000,
                        epochs=500,
                        # validation_split=0.2,
                        validation_data=(X_test, y_test),
                        batch_size=batch_size,
                        # callbacks=[tensorboard_callback],
                        verbose=1
                        )

    y_pred = model.predict(X_train)
    # y_pred *= np.std(y_raw)  # undo normalization on y
    # y_pred += np.mean(y_raw)
    r2 = r2_score(y_train, y_pred)
    print("Keras training R^2", r2)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Keras validation R^2", r2)

    if False: # Show training results
        y_pred = model.predict(X)
        y_pred *= np.std(y_raw)     # undo normalization on y
        y_pred += np.mean(y_raw)
        r2 = r2_score(y_raw, y_pred)
        print("Keras training R^2", r2)
        plt.ylabel("MAE")
        plt.xlabel("epochs")

        plt.plot(history.history['val_mae'], label='val_mae')
        plt.plot(history.history['mae'], label='train_mae')
        plt.title(f"batch_size {batch_size}, Layer1 {layer1}, Layer2 {layer2}, R^2 {r2:.3f}")
        plt.legend()
        plt.show()

    return model, r2


def partitioning():
    np.random.seed(1)  # pick seed for reproducible article images
    # np.random.seed(2)
    n = 200
    x = np.random.uniform(0, 1, size=n)
    x1 = x + np.random.normal(0, 0.1, n)
    # x2 = x + np.random.normal(0, 0.03, n)
    x2 = (x*4).astype(int) + np.random.randint(0, 5, n)
    X = np.vstack([x1, x2]).T

    y = X[:, 0] + X[:, 1] ** 2

    regr = tree.DecisionTreeRegressor(max_leaf_nodes=8,
                                      min_samples_leaf=1)  # limit depth of tree
    regr.fit(X[:,1].reshape(-1,1), y)

    shadow_tree = ShadowDecTree(regr, X[:,1].reshape(-1,1), y, feature_names=['x1', 'x2'])
    splits = []
    print("splits")
    for node in shadow_tree.internal:
        splits.append(node.split())
        print("\t",node.split())
    splits = sorted(splits)

    fig, ax = plt.subplots(1, 1, figsize=(3,2.5))

    color_map_min = '#c7e9b4'
    color_map_max = '#081d58'

    y_lim = np.min(y), np.max(y)
    y_range = y_lim[1] - y_lim[0]
    n_colors_in_map = 100
    markersize = 5
    scatter_edge=GREY
    color_map = [rgb2hex(c.rgb, force_long=True)
                 for c in Color(color_map_min).range_to(Color(color_map_max), n_colors_in_map)]
    color_map = [color_map[int(((y_-y_lim[0])/y_range)*(n_colors_in_map-1))] for y_ in y]
    # ax.scatter(x, y, marker='o', c=color_map, edgecolor=scatter_edge, lw=.3, s=markersize)

    ax.scatter(X[:,0], X[:,1], marker='o', c=color_map, alpha=.7, edgecolor=scatter_edge, lw=.3, s=markersize)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    a = -.08
    b = 1.05
    ax.set_xlim(a, b)
    # ax.set_ylim(a, b+0.02)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    for s in splits:
        ax.plot([a,b], [s,s], '-', c='grey', lw=.5)

    # savefig("partitioning_background")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(f"images/partitioning_background.svg", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    productsize()
    interactions()
    yearmade()
    rent()
    rent_ntrees()
    weight()
    shap_pregnant()
    shap_weight(feature_perturbation='tree_path_dependent', twin=True) # more biased but faster
    shap_weight(feature_perturbation='interventional', twin=True)      # takes 04:45 minutes
    weather()
    noise()

    # Invoke R to generate csv files then load with python to plot

    gen_ale_plot_data_in_R()

    ale_MachineHours()
    ale_yearmade()
    ale_height()
    ale_pregnant()
    ale_state()
    ale_productsize()
