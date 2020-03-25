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
from stratx.support import *
import inspect
import statsmodels.api as sm
from dtreeviz.trees import *

def df_string_to_cat(df: pd.DataFrame) -> dict:
    catencoders = {}
    for colname in df.columns:
        if is_string_dtype(df[colname]) or is_object_dtype(df[colname]):
            df[colname] = df[colname].astype('category').cat.as_ordered()
            catencoders[colname] = df[colname].cat.categories
    return catencoders


def weather():
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

    plot_catstratpd(X, y, 'state', 'temperature', catnames=catnames,
                    # min_samples_leaf=30,
                    n_trials=1,
                    min_y_shifted_to_zero=True,
                    # leftmost_shifted_to_zero=True,
                    show_x_counts=False,
                    bootstrap=True,
                    yrange=(-1, 55),
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

def bulldozer():
    n = 10_000
    X, y = load_bulldozer(n=n)
    # cat_partial_dependence(X, y, 'ModelID')
    # plot_catstratpd(X, y, 'saledayofweek', 'SalePrice', show_x_counts=False)
    # ux, cx = np.unique(X['MachineHours'], return_counts=True)
    # print(np.argmax(cx))
    # print(ux[np.argmax(cx)])
    plot_stratpd(X, y, 'MachineHours', 'SalePrice', n_trials=10, show_all_pdp=False)
    plt.show()


def toy_weight_data_no_edu(n):
    df = pd.DataFrame()
    nmen = n // 2 # 50/50 men/women
    nwomen = n // 2
    df['sex'] = ['M'] * nmen + ['F'] * nwomen
    df.loc[df['sex'] == 'F', 'pregnant'] = np.random.randint(0, 2, size=(nwomen,))
    df.loc[df['sex'] == 'M', 'pregnant'] = 0
    df.loc[df['sex'] == 'M', 'height'] = 5 * 12 + 8 + np.random.uniform(-7, +8,
                                                                        size=(nmen,))
    df.loc[df['sex'] == 'F', 'height'] = 5 * 12 + 5 + np.random.uniform(-4.5, +5,
                                                                        size=(nwomen,))
    df['weight'] = 120 \
                   + (df['height'] - df['height'].min()) * 10 \
                   + df['pregnant'] * 40
    df['pregnant'] = df['pregnant'].astype(bool)
    eqn = "y = 120 + 10(x_{height} - min(x_{height})) + 30x_{pregnant}"

    df['pregnant'] = df['pregnant'].astype(int)
    df['sex'] = df['sex'].map({'M': 0, 'F': 1}).astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    return X, y, df, eqn


def weight_partitioning():
    X, y, df_raw, eqn = toy_weight_data_no_edu(1000)
    print(df_raw.head(3))

    regr = tree.DecisionTreeRegressor(min_samples_leaf=10)  # limit depth of tree
    X_no_height = X.drop('height', axis=1)
    regr.fit(X_no_height, y)

    feature_names = ['sex', 'pregnant']
    shadow_tree = ShadowDecTree(regr, X_no_height, y, feature_names=feature_names)
    splits = []
    for node in shadow_tree.internal:
        print(node)
        splits.append(node.split())
    splits = sorted(splits)
    print(splits)
    t = dtreeviz(regr, X_no_height, y, feature_names=feature_names, target_name='weight')
    # t.view()

    rtreeviz_bivar_heatmap(None,
                           X_no_height, y,  # partition just x2
                           min_samples_leaf=10,
                           feature_names=feature_names,
                           n_colors_in_map=3,
                           fontsize=14)
    plt.savefig("/tmp/foo.pdf")

    plot_stratpd(X, y, 'height', 'weight', min_slopes_per_x=2, n_trials=1, show_all_pdp=False)
    plt.show()


def weight():
    X, y, df_raw, eqn = toy_weight_data(2000)
    # df = df_raw.copy()
    # df_string_to_cat(df)
    # df_cat_to_catcode(df)
    # df['pregnant'] = df['pregnant'].astype(int)

    print()
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, max_features=0.9, oob_score=True)
    rf.fit(X, y) # Use full data set for plotting
    print("RF OOB R^2", rf.oob_score_)

    print(X.head(5))

    # show pregnant female at max range drops going taller
    X_test = [[1, 1, 70, 10]]
    X_test = np.array(X_test)
    y_pred = rf.predict(X_test)
    print("pregnant female at max range", y_pred)

    X_test = [[1, 1, 72, 10]] # make them taller
    y_pred = rf.predict(X_test)
    print("pregnant female in male height range", y_pred)

    X_test = [[1, 0, 72, 10]]
    y_pred = rf.predict(X_test)
    print("nonpregnant female in male height range", y_pred)

    X_test = [[0, 0, 72, 10]]
    y_pred = rf.predict(X_test)
    print("male in male height range", y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    ice = predict_ice(rf, X[X['sex']==0], 'height', 'weight')
    plot_ice(ice, 'height', 'weight', ax=axes[0], pdp_linewidth=2, yrange=(100, 250),
             min_y_shifted_to_zero=False)
    axes[0].set_xlabel("height\n(a)", fontsize=12)
    axes[0].set_title("Male FPD/ICE", fontsize=10)
    axes[0].set_xticks([60,65,70,75])

    ice = predict_ice(rf, X[X['sex']==1], 'height', 'weight')
    plot_ice(ice, 'height', 'weight', ax=axes[1], pdp_linewidth=2, yrange=(100, 250),
             min_y_shifted_to_zero=False)
    axes[1].set_xlabel("height\n(a)", fontsize=12)
    axes[1].set_title("Female FPD/ICE", fontsize=10)
    axes[1].set_xticks([60,65,70,75])

    plt.show()

def sample_x1_equals_x2_pic():
    np.random.seed(1)
    n = 100
    x = np.random.uniform(0, 1, size=n)
    x1 = x + np.random.normal(0, 0.05, n)
    # x2 = x + np.random.normal(0, 0.05, n)
    x2 = (x*4).astype(int) + np.random.randint(0, 5, n)
    X = np.vstack([x1, x2]).T

    y = X[:, 0] + X[:, 1] ** 2

    regr = tree.DecisionTreeRegressor(min_samples_leaf=5)  # limit depth of tree
    regr.fit(X[:,1].reshape(-1,1), y)

    print(np.unique(X[:,1]))

    shadow_tree = ShadowDecTree(regr, X[:,1].reshape(-1,1), y, feature_names=['x1', 'x2'])
    splits = []
    for node in shadow_tree.internal:
        print(node)
        splits.append(node.split())
    splits = sorted(splits)
    print(splits)

    fig, ax = plt.subplots(1, 1, figsize=(3,2.5))
    ax.scatter(X[:,0], X[:,1], c='k', alpha=.7, s=10)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_yticks(np.unique(X[:,1]))
    a = -.08
    b = 1.05
    ax.set_xlim(a, b)
    ax.set_ylim(-.5,8)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    for s in splits:
        ax.plot([a,b], [s,s], ':', c='grey', lw=1.5)

    plt.tight_layout()
    plt.savefig
    # t = rtreeviz_bivar_heatmap(ax,
    #                            X, y, # partition just x2
    #                            min_samples_leaf=20,
    #                            feature_names=['$x_1$','$x_2$'],
    #                            fontsize=14)

    t = rtreeviz_univar(ax,
                               X[:,1], y, # partition just x2
                               min_samples_leaf=20,
                               fontsize=14,
                        show={})

    plt.show()
    plt.close()

    # viz = dtreeviz(regr,
    #          X[:,1].reshape(-1,1),
    #          y,
    #          target_name='y',
    #          feature_names=['x2'])
    # viz.view()
    # plt.show()


def rent():
    # X,y = load_rent(n=10_000)
    # df_rent = X.copy()
    # df_rent['price'] = y

    df = pd.read_csv("rent10k.csv")
    X = df.drop('price', axis=1)
    y = df['price']

    # plot_stratpd(X, y, 'longitude', 'price', n_trials=10, show_all_pdp=True)
    plot_stratpd(X, y, 'latitude', 'price', n_trials=10,
                 min_slopes_per_x=20,
                 show_all_pdp=True,
                 yrange=(-300,300))
    plt.show()


rent()
# sample_x1_equals_x2_pic()
# weight_partitioning()
# weight()
# bulldozer()
# weather()
#bigX()