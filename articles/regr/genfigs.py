import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
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
from mine.plot import *
from mine.ice import *

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


def savefig(filename):
    plt.savefig(f"images/{filename}.pdf")
    plt.savefig(f"images/{filename}.png")


def load_rent():
    """
    *Data use rules prevent us from storing this data in this repo*. Download the data
    set from Kaggle. (You must be a registered Kaggle user and must be logged in.)
    Go to the Kaggle [data page](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data)
    and save `train.json`
    :return:
    """
    df = pd.read_json('train.json')

    # Create ideal numeric data set w/o outliers etc...
    df = df[(df.price > 1_000) & (df.price < 10_000)]
    df = df[df.bathrooms <= 3]  # There's almost no data for 3.5 and above with small sample
    df = df[(df.longitude != 0) | (df.latitude != 0)]
    df = df[(df['latitude'] > 40.55) & (df['latitude'] < 40.94) &
            (df['longitude'] > -74.1) & (df['longitude'] < -73.67)]
    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price']]

    return df_rent


def rent():
    df_rent = load_rent()
    df_rent = df_rent.sample(n=6000)  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    avg_per_baths = df_rent.groupby('bathrooms').mean()['price']

    fig, ax = plt.subplots(1,1, figsize=(3,3))
    ax.scatter(df_rent['bathrooms'], df_rent['price'], alpha=0.07, s=6)#, label="observation")
    ax.scatter(np.unique(df_rent['bathrooms']), avg_per_baths, c='black', label="average price/baths")
    ax.set_xlabel("bathrooms")#, fontsize=12)
    ax.set_ylabel("Rent price")#, fontsize=12)
    ax.set_ylim(0,10_000)
    # ax.legend()
    plt.tight_layout()
    savefig("baths_vs_price")
    plt.close()

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1,1, figsize=(3,3))
    ax.set_ylim(-1000,5000)
    ice = ice_predict(rf, X, 'bathrooms', 'price', nlines=700)
    ice_plot(ice, 'bathrooms', 'price', alpha=.05, ax=ax)
    plt.tight_layout()
    savefig("baths_vs_price_pdp")
    plt.close()

    fig, ax = plt.subplots(1,1, figsize=(3,3))
    ax.set_ylim(-1000,5000)
    mine_plot(X, y, 'bathrooms', 'price', ax=ax, alpha=.2, nlines=700)
    plt.tight_layout()
    savefig("baths_vs_price_mipd")
    plt.close()

    lm = Lasso()
    lm.fit(X, y)

    fig, ax = plt.subplots(1,1, figsize=(3,3))
    ax.set_ylim(-1000,5000)
    ice = ice_predict(lm, X, 'bathrooms', 'price', nlines=700)
    ice_plot(ice, 'bathrooms', 'price', alpha=.05, ax=ax)
    plt.tight_layout()
    savefig("baths_vs_price_pdp_lm")
    plt.close()


if __name__ == '__main__':
    rent()
