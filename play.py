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
from stratpd.plot import *
from stratpd.ice import *

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

def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n//2
    nwomen = n//2
    df['ID'] = range(100,100+n)
    df['sex'] = ['M']*nmen + ['F']*nwomen
    df.loc[df['sex']=='F','pregnant'] = np.random.randint(0,2,size=(nwomen,))
    df.loc[df['sex']=='M','pregnant'] = 0
    df.loc[df['sex']=='M','height'] = 5*12+8 + np.random.uniform(-7, +8, size=(nmen,))
    df.loc[df['sex']=='F','height'] = 5*12+5 + np.random.uniform(-4.5, +5, size=(nwomen,))
    df.loc[df['sex']=='M','education'] = 10 + np.random.randint(0,8,size=nmen)
    df.loc[df['sex']=='F','education'] = 12 + np.random.randint(0,8,size=nwomen)
    df['weight'] = 120 \
                   + (df['height']-df['height'].min()) * 10 \
                   + df['pregnant']*30 \
                   - df['education']*1.2
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    return df

def weight():
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(5, 2, figsize=(8,16), gridspec_kw = {'height_ratios':[.2,3,3,3,3]})

    axes[0,0].get_xaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,0].axis('off')
    axes[0,1].axis('off')

    # mine_plot(X, y, 'education', 'weight', ax=axes[1][0],
    #              yrange=(-12,0),
    #           nlines = 500
    #              )
    # mine_plot(X, y, 'height', 'weight', ax=axes[2][0],
    #              yrange=(0,160),
    # nlines = 1000
    # )
    catstratpd_plot(X, y, 'sex', 'weight', ax=axes[3][0],
                     alpha=.2,
                     cats=df_raw['sex'].unique(),
                     yrange=(0,5)
                     )
    catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[4][0],
                     alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                     yrange=(0,35)
                     )

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    # ice = ice_predict(rf, X, 'education', 'weight')
    # ice_plot(ice, 'education', 'weight', ax=axes[1, 1], yrange=(-12, 0))
    # ice = ice_predict(rf, X, 'height', 'weight')
    # ice_plot(ice, 'height', 'weight', ax=axes[2, 1], yrange=(0, 160))
    # ice = ice_predict(rf, X, 'sex', 'weight')
    # ice_plot(ice, 'sex', 'weight', ax=axes[3,1], yrange=(0,5), cats=df_raw['sex'].unique())
    # ice = ice_predict(rf, X, 'pregnant', 'weight')
    # ice_plot(ice, 'pregnant', 'weight', ax=axes[4,1], yrange=(0,10), cats=df_raw['pregnant'].unique())

    fig.suptitle("weight = 120 + 10*(height-min(height)) + 10*pregnant - 1.2*education", size=14)

    plt.tight_layout()

    plt.show()


def rent():
    df = pd.read_json('notebooks/data/train.json')

    # Create ideal numeric data set w/o outliers etc...
    # Create ideal numeric data set w/o outliers etc...
    df = df[(df.price > 1_000) & (df.price < 10_000)]
    df = df[df.bathrooms <= 6]  # There's almost no data for above
    df = df[(df.longitude != 0) | (df.latitude != 0)]
    df = df[(df['latitude'] > 40.55) & (df['latitude'] < 40.94) &
            (df['longitude'] > -74.1) & (df['longitude'] < -73.67)]
    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price']]
    df_rent.head()

    df_rent = df_rent.sample(n=4000)  # get a small subsample

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    fig, axes = plt.subplots(4, 2, figsize=(8,16))
    stratpd_plot(X, y, 'bedrooms', 'price', ax=axes[0,0], alpha=.2, yrange=(0,3000), nlines=1000)
    #     stratpd_plot(X, y, 'bedrooms', 'price', ax=axes[0,0], alpha=.2, yrange=(0,3000), nlines=1000)(X, y, 'bathrooms', 'price', ax=axes[1,0], alpha=.2, yrange=(0,5000), nlines=1000)
    #     stratpd_plot(X, y, 'bedrooms', 'price', ax=axes[0,0], alpha=.2, yrange=(0,3000), nlines=1000)(X, y, 'latitude', 'price', ax=axes[2,0], alpha=.2, yrange=(0,1700), nlines=1000)
    #     stratpd_plot(X, y, 'bedrooms', 'price', ax=axes[0,0], alpha=.2, yrange=(0,3000), nlines=1000)(X, y, 'longitude', 'price', ax=axes[3,0], alpha=.2, yrange=(-3000,250), nlines=1000)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    # rf = Lasso()
    # rf.fit(X, y)

    # ice = ice_predict(rf, X, 'bedrooms', 'price')
    # ice_plot(ice, 'bedrooms', 'price', ax=axes[0, 1], alpha=.05, yrange=(0,3000))
    # ice = ice_predict(rf, X, 'bathrooms', 'price')
    # ice_plot(ice, 'bathrooms', 'price', alpha=.05, ax=axes[1, 1])
    # ice = ice_predict(rf, X, 'latitude', 'price')
    # ice_plot(ice, 'latitude', 'price', ax=axes[2, 1], alpha=.05, yrange=(0,1700))
    # ice = ice_predict(rf, X, 'longitude', 'price')
    # ice_plot(ice, 'longitude', 'price', ax=axes[3, 1], alpha=.05, yrange=(-750,3000))

    plt.tight_layout()
    plt.show()


def toy_weather_data():
    def temp(x): return np.sin((x+365/2)*(2*np.pi)/365)
    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df = pd.DataFrame()
    df['dayofyear'] = range(1,365+1)
    df['state'] = np.random.choice(['CA','CO','AZ','WA'], len(df))
    df['temperature'] = temp(df['dayofyear'])
    df.loc[df['state']=='CA','temperature'] = 70 + df.loc[df['state']=='CA','temperature'] * noise('CA')
    df.loc[df['state']=='CO','temperature'] = 40 + df.loc[df['state']=='CO','temperature'] * noise('CO')
    df.loc[df['state']=='AZ','temperature'] = 90 + df.loc[df['state']=='AZ','temperature'] * noise('AZ')
    df.loc[df['state']=='WA','temperature'] = 60 + df.loc[df['state']=='WA','temperature'] * noise('WA')
    return df


def weather():
    df_raw = toy_weather_data()
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    print(catencoders)
    df_cat_to_catcode(df)
    X = df.drop('temperature', axis=1)
    y = df['temperature']

    fig, axes = plt.subplots(4, 2, figsize=(8, 8),
                             gridspec_kw={'height_ratios': [.2, 3, 3, 3]})

    axes[0, 0].get_xaxis().set_visible(False)
    axes[0, 1].get_xaxis().set_visible(False)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')

    """
    The scale diff between states, obscures the sinusoidal nature of the
    dayofyear vs temp plot. With noise N(0,5) gotta zoom in -3,3 on mine too.
    otherwise, smooth quasilinear plot with lots of bristles showing volatility.
    Flip to N(-5,5) which is more realistic and we see sinusoid for both, even at
    scale. yep, the N(0,5) was obscuring sine for both. 
    """
    stratpd_plot(X, y, 'dayofyear', 'temperature', ax=axes[1][0],
                 # ntrees=10, min_samples_leaf=2,
                 hires_min_samples_leaf=11
                 )  # , yrange=(-10,10))
    # catstratpd_plot(X, y, 'state', 'temperature', cats=catencoders['state'],
    #                 ax=axes[2][0])  # , yrange=(0,160))
    #
    # rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
    # rf.fit(X, y)
    #
    # ice = ice_predict(rf, X, 'dayofyear', 'temperature')
    # ice_plot(ice, 'dayofyear', 'temperature', ax=axes[1, 1])  # , yrange=(-12,0))
    #
    # ice = ice_predict(rf, X, 'state', 'temperature')
    # ice_plot(ice, 'state', 'temperature', cats=catencoders['state'],
    #          ax=axes[2, 1])  # , yrange=(-12,0))
    #
    df = df_raw.copy()
    axes[3, 0].plot(df.loc[df['state'] == 'CA', 'dayofyear'],
                    df.loc[df['state'] == 'CA', 'temperature'], label="CA")
    axes[3, 0].plot(df.loc[df['state'] == 'CO', 'dayofyear'],
                    df.loc[df['state'] == 'CO', 'temperature'], label="CO")
    axes[3, 0].plot(df.loc[df['state'] == 'AZ', 'dayofyear'],
                    df.loc[df['state'] == 'AZ', 'temperature'], label="AZ")
    axes[3, 0].plot(df.loc[df['state'] == 'WA', 'dayofyear'],
                    df.loc[df['state'] == 'WA', 'temperature'], label="WA")
    axes[3, 0].legend()
    axes[3, 0].set_title('Raw data')
    axes[3, 0].set_ylabel('Temperature')
    axes[3, 0].set_xlabel('Dataframe row index')

    rtreeviz_univar(axes[3, 1],
                    X['state'], y,
                    feature_name='state',
                    target_name='y',
                    min_samples_leaf=2,
                    fontsize=10)
    axes[3, 1].set_title(f'state space partition with min_samples_leaf={2}')
    axes[3, 1].set_xlabel("state")
    axes[3, 1].set_ylabel("y")

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    # rent()
    # weight()
    weather()