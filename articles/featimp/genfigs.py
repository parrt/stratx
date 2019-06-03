import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata
from  matplotlib.collections import LineCollection
import time
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from pdpbox import pdp
from rfpimp import *
from scipy.integrate import cumtrapz
from stratx.featimp import *
from stratx.partdep import *
from stratx.ice import *
import inspect
import shap
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor

def df_string_to_cat(df:pd.DataFrame) -> dict:
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


def addnoise(df, n=1, c=0.5):
    if n==1:
        df['noise'] = np.random.random(len(df)) * c
        return
    for i in range(n):
        df[f'noise{i+1}'] = np.random.random(len(df)) * c


def savefig(filename):
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(f"images/{filename}.pdf")
    plt.savefig(f"images/{filename}.png", dpi=300)
    plt.close()

def boston():
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target
    addnoise(df, 3)

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    plot_all_imp(X, y)

    savefig("boston")

def cars():
    df_cars = pd.read_csv("../../notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)
    df_cars.head(5)

    catencoders = df_string_to_cat(df_cars)
    df_cat_to_catcode(df_cars)

    addnoise(df_cars, 3)

    # X = df_cars[['horsepower', 'weight', 'noise']]
    X = df_cars.drop(['mpg','name'], axis=1)
    y = df_cars['mpg']
    plot_all_imp(X, y)

    savefig("cars")

def rent():
    df = pd.read_json('../../notebooks/data/train.json')

    # Create ideal numeric data set w/o outliers etc...
    # Create ideal numeric data set w/o outliers etc...
    df = df[(df.price > 1_000) & (df.price < 10_000)]
    df = df[df.bathrooms <= 6]  # There's almost no data for above
    df = df[(df.longitude != 0) | (df.latitude != 0)]
    df = df[(df['latitude'] > 40.55) & (df['latitude'] < 40.94) &
            (df['longitude'] > -74.1) & (df['longitude'] < -73.67)]
    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price']]
    df_rent.head()

    df_rent = df_rent.sample(n=2000)  # get a small subsample
    addnoise(df_rent, 3)

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    plot_all_imp(X, y)

    savefig("rent")

def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n // 2
    nwomen = n // 2
    # df['ID'] = range(100,100+n)
    df['sex'] = ['M'] * nmen + ['F'] * nwomen
    df.loc[df['sex'] == 'F', 'pregnant'] = np.random.randint(0, 2, size=(nwomen,))
    df.loc[df['sex'] == 'M', 'pregnant'] = 0
    df.loc[df['sex'] == 'M', 'height'] = 5 * 12 + 8 + np.random.uniform(-7, +8,
                                                                        size=(nmen,))
    df.loc[df['sex'] == 'F', 'height'] = 5 * 12 + 5 + np.random.uniform(-4.5, +5,
                                                                        size=(
                                                                        nwomen,))
    df.loc[df['sex'] == 'M', 'education'] = 10 + np.random.randint(0, 8, size=nmen)
    df.loc[df['sex'] == 'F', 'education'] = 12 + np.random.randint(0, 8, size=nwomen)
    df['weight'] = 120 \
                   + (df['height'] - df['height'].min()) * 10 \
                   + df['pregnant'] * 30 \
                   - df['education'] * 1.2
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    return df

def weight():
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()

    addnoise(df, 3)

    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']
    plot_all_imp(X, y)
    savefig("weight")

    plot_all_imp_and_models(X, y)
    savefig("weight_models")

# def meta_weight():
#     df_raw = toy_weight_data(2000)
#     df = df_raw.copy()
#
#     addnoise(df, 3)
#
#     df_string_to_cat(df)
#     df_cat_to_catcode(df)
#     df['pregnant'] = df['pregnant'].astype(int)
#     X = df.drop('weight', axis=1)
#     y = df['weight']
#
#     ncols = len(X.columns)
#     fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(15, 0.22 * ncols))
#
#     i = 0
#     for n in [2, 5, 10, 20, 50, 100, 200]:
#         I = strat_importances(X, y, min_samples_leaf=n, hires_threshold=1000)
#         plot_importances(I, ax=axes[i], color='#fee090')
#         axes[i].set_title(f"{n} samples / leaf")
#         i += 1
#
#     savefig("weight_leaf_size")


def plot_all_imp(X, y, model=None, figsize=None):
    if figsize is None:
        ncols = len(X.columns)
        if ncols<=5:
            figsize = (10, 0.3 * ncols)
        else:
            figsize = (10, 0.22 * ncols)

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=figsize)

    I = strat_importances(X, y, min_samples_leaf=50, hires_threshold=1000)
    plot_importances(I, ax=axes[0], color='#fee090')

    if model is None:
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=False)
        model.fit(X, y)

    start = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_values = np.mean(np.abs(shap_values), axis=0) # measure avg magnitude
    stop = time.time()
    print(f"SHAP time {(stop-start):.1f}s")
    I = pd.DataFrame(data={'Feature':X.columns, 'Importance':shap_values})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    plot_importances(I, ax=axes[1])

    I = importances(model, X, y)
    plot_importances(I, ax=axes[2])

    I = dropcol_importances(model, X, y)
    plot_importances(I, ax=axes[3])

    I = pd.DataFrame(data={'Feature':X.columns, 'Importance':model.feature_importances_})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    plot_importances(I, ax=axes[4])

    axes[0].set_title('StratIm')
    axes[1].set_title('SHAP')
    axes[2].set_title('Permutation')
    axes[3].set_title('Dropcol')
    axes[4].set_title('Gini')


def plot_all_imp_and_models(X, y, figsize=None):
    if figsize is None:
        ncols = len(X.columns)
        if ncols<=5:
            figsize = (11, 0.3 * ncols * 3)
        else:
            figsize = (11, 0.22 * ncols * 3)

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=figsize)

    I = strat_importances(X, y, min_samples_leaf=50, hires_threshold=1000)
    plot_importances(I, ax=axes[0,0], color='#fee090')

    regrs = [
        RandomForestRegressor(n_estimators=100, min_samples_leaf=1),
#        svm.SVR(gamma='scale'),
        KNeighborsRegressor(5),
        LinearRegression()]

    row = 0
    for regr in regrs:
        regr.fit(X, y)
        rname = regr.__class__.__name__
        if rname=='SVR':
            rname = "SVM"
        if rname=='RandomForestRegressor':
            rname = "RF"
        if rname=='LinearRegression':
            rname = 'Linear'
        axes[row,1].set_title(rname)
        axes[row,2].set_title(rname)
        axes[row,3].set_title(rname)
        axes[row,4].set_title(rname)

        start = time.time()
        if isinstance(regr, RandomForestRegressor):
            explainer = shap.TreeExplainer(regr)
        elif isinstance(regr, LinearRegression):
            explainer = shap.LinearExplainer(regr, X, feature_dependence="correlation", nsamples=100)
        else:
            # way too slow to be useful
            explainer = shap.KernelExplainer(regr.predict, X.sample(n=25), link='identity')

        shap_values = explainer.shap_values(X)
        shap_values = np.mean(np.abs(shap_values), axis=0) # measure avg magnitude
        stop = time.time()
        print(f"SHAP time {(stop-start):.1f}s")
        I = pd.DataFrame(data={'Feature':X.columns, 'Importance':shap_values})
        I = I.set_index('Feature')
        I = I.sort_values('Importance', ascending=False)
        plot_importances(I, ax=axes[row,1])

        I = importances(regr, X, y)
        plot_importances(I, ax=axes[row,2])

        I = dropcol_importances(regr, X, y)
        plot_importances(I, ax=axes[row,3])

        if isinstance(regr, RandomForestRegressor):
            I = pd.DataFrame(data={'Feature':X.columns, 'Importance':regr.feature_importances_})
            I = I.set_index('Feature')
            I = I.sort_values('Importance', ascending=False)
            plot_importances(I, ax=axes[row,4])

        row += 1

    axes[0,0].set_title('StratIm')
    axes[0,1].set_title('SHAP')
    axes[0,2].set_title('Permutation')
    axes[0,3].set_title('Dropcol')
    axes[0,4].set_title('Gini')


if __name__ == '__main__':
    # rent()
    # boston()
    # cars()
    # weight()
    # meta_weight()