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
from stratpd.plot import *
from stratpd.ice import *
import inspect

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
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(f"images/{filename}.pdf")
    plt.savefig(f"images/{filename}.png")


def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n//2
    nwomen = n//2
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


def load_rent():
    """
    *Data use rules prevent us from storing this data in this repo*. Download the data
    set from Kaggle. (You must be a registered Kaggle user and must be logged in.)
    Go to the Kaggle [data page](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data)
    and save `train.json`
    :return:
    """
    df = pd.read_json('../../notebooks/data/train.json')

    # Create ideal numeric data set w/o outliers etc...
    df = df[(df.price > 1_000) & (df.price < 10_000)]
    df = df[df.bathrooms <= 4]  # There's almost no data for above with small sample
    df = df[(df.longitude != 0) | (df.latitude != 0)]
    df = df[(df['latitude'] > 40.55) & (df['latitude'] < 40.94) &
            (df['longitude'] > -74.1) & (df['longitude'] < -73.67)]
    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price']]

    return df_rent


def rent():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent.sample(n=9000)  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    figsize = (5,5)
    colname='bathrooms'

    fig, axes = plt.subplots(2,2, figsize=figsize)
    avg_per_baths = df_rent.groupby(colname).mean()['price']
    axes[0,0].scatter(df_rent[colname], df_rent['price'], alpha=0.07, s=5)#, label="observation")
    axes[0,0].scatter(np.unique(df_rent[colname]), avg_per_baths, s=6, c='black', label="average price/{colname}")
    axes[0,0].set_ylabel("price")#, fontsize=12)
    axes[0,0].set_ylim(0,10_000)
    # ax.legend()

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    ice = ice_predict(rf, X, colname, 'price', nlines=1000)
    ice_plot(ice, colname, 'price', alpha=.05, ax=axes[0,1], show_xlabel=False)
    axes[0,1].set_ylim(-1000,5000)

    lm = Lasso()
    lm.fit(X, y)

    ice = ice_predict(lm, X, colname, 'price', nlines=1000)
    ice_plot(ice, colname, 'price', alpha=.05, ax=axes[1,0], show_ylabel=False)
    axes[1,0].set_ylim(-1000,5000)

    stratpd_plot(X, y, colname, 'price', ax=axes[1,1], alpha=.2)
    axes[1,1].set_ylim(-1000,5000)

    #axes[1,1].get_yaxis().set_visible(False)

    savefig(f"{colname}_vs_price")
    plt.close()


def meta_rent():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent.sample(n=9000, random_state=111)  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    supervised = True

    def onevar(colname, row, yrange=None):
        alpha = 0.08
        for i, t in enumerate([1, 5, 10, 30]):
            stratpd_plot(X, y, colname, 'price', ax=axes[row,i], alpha=alpha,
                         yrange=yrange,
                         supervised=supervised,
                         show_ylabel = t==1,
                         ntrees=t)

    fig, axes = plt.subplots(3, 4, figsize=(8,6), sharey=True)
    for i in range(1,4):
        axes[0, i].get_yaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
        axes[2, i].get_yaxis().set_visible(False)

    onevar('bedrooms', row=0, yrange=(0,3000))
    onevar('bathrooms', row=1, yrange=(0,3000))
    onevar('latitude', row=2, yrange=(0,3000))
    
    savefig(f"rent_ntrees")
    plt.close()

    # fig, axes = plt.subplots(1, 4, figsize=(8,2), sharey=True)
    # onevar('longitude', row=3, yrange=(-4000,4000))
    # 
    # savefig(f"longitude_vs_price_ntrees")
    # plt.close()


def unsup_rent():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent.sample(n=9000, random_state=111)  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    supervised = False

    fig, axes = plt.subplots(3, 2, figsize=(4,6), sharey=True)

    stratpd_plot(X, y, 'bedrooms', 'price', ax=axes[0,0], alpha=.2, supervised=False)
    stratpd_plot(X, y, 'bedrooms', 'price', ax=axes[0,1], alpha=.2, supervised=True)

    stratpd_plot(X, y, 'bathrooms', 'price', ax=axes[1,0], alpha=.2, supervised=False)
    stratpd_plot(X, y, 'bathrooms', 'price', ax=axes[1,1], alpha=.2, supervised=True)

    stratpd_plot(X, y, 'latitude', 'price', ax=axes[2,0], alpha=.2, supervised=False)
    stratpd_plot(X, y, 'latitude', 'price', ax=axes[2,1], alpha=.2, supervised=True)

    for i in range(3):
        axes[i,1].get_yaxis().set_visible(False)

    
    savefig(f"rent_unsup")
    plt.close()


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
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_yr1 = toy_weather_data()
    df_yr1['year']=1980
    df_yr2 = toy_weather_data()
    df_yr2['year']=1981
    df_yr3 = toy_weather_data()
    df_yr3['year']=1982
    df_raw = pd.concat([df_yr1, df_yr2, df_yr3], axis=0)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    print(catencoders)
    df_cat_to_catcode(df)
    X = df.drop('temperature', axis=1)
    y = df['temperature']
    figsize=(2.5,2.5)
    """
    The scale diff between states, obscures the sinusoidal nature of the
    dayofyear vs temp plot. With noise N(0,5) gotta zoom in -3,3 on mine too.
    otherwise, smooth quasilinear plot with lots of bristles showing volatility.
    Flip to N(-5,5) which is more realistic and we see sinusoid for both, even at
    scale. yep, the N(0,5) was obscuring sine for both. 
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    stratpd_plot(X, y, 'dayofyear', 'temperature', ax=ax,
                 hires_min_samples_leaf=13,
                 yrange=(-15,15),
                 pdp_dot_size=2, alpha=.5)
    
    savefig(f"dayofyear_vs_temp_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    catstratpd_plot(X, y, 'state', 'temperature', cats=catencoders['state'],
                    sort=None,
                    alpha=.3,
                    min_samples_leaf=11,
                 ax=ax)  # , yrange=(0,160))
    
    savefig(f"state_vs_temp_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = ice_predict(rf, X, 'dayofyear', 'temperature')
    ice_plot(ice, 'dayofyear', 'temperature', ax=ax, yrange=(-15,15))
    
    savefig(f"dayofyear_vs_temp_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = ice_predict(rf, X, 'state', 'temperature')
    ice_plot(ice, 'state', 'temperature', cats=catencoders['state'], ax=ax)
    
    savefig(f"state_vs_temp_pdp")
    plt.close()

    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    # rtreeviz_univar(ax,
    #                 X['state'], y,
    #                 feature_name='state',
    #                 target_name='y',
    #                 fontsize=10, show={'splits'})
    # 
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(X['state'], y, alpha=.05, s=20)
    ax.set_xticklabels(np.concatenate([[''], catencoders['state'].values]))
    ax.set_xlabel("state")
    ax.set_ylabel("temperature")
    
    savefig(f"state_vs_temp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df = df_raw.copy()
    avgtmp = df.groupby(['state','dayofyear'])[['temperature']].mean()
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
    
    savefig(f"dayofyear_vs_temp")
    plt.close()


def weight():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']
    figsize=(2.5,2.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    stratpd_plot(X, y, 'education', 'weight', ax=ax, yrange=(-12, 0), alpha=.1, nlines=700, show_ylabel=False)
#    ax.get_yaxis().set_visible(False)
    savefig(f"education_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    stratpd_plot(X, y, 'height', 'weight', ax=ax, yrange=(0, 160), alpha=.1, nlines=700, show_ylabel=False)
#    ax.get_yaxis().set_visible(False)
    savefig(f"height_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    catstratpd_plot(X, y, 'sex', 'weight', ax=ax,
                    alpha=.2,
                    cats=df_raw['sex'].unique(),
                    yrange=(0, 5)
                    )
    savefig(f"sex_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    catstratpd_plot(X, y, 'pregnant', 'weight', ax=ax,
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(-5,35)
                    )
    savefig(f"pregnant_vs_weight_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = ice_predict(rf, X, 'education', 'weight')
    ice_plot(ice, 'education', 'weight', ax=ax, yrange=(-12, 0))
    savefig(f"education_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = ice_predict(rf, X, 'height', 'weight')
    ice_plot(ice, 'height', 'weight', ax=ax, yrange=(0, 160))
    savefig(f"height_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = ice_predict(rf, X, 'sex', 'weight')
    ice_plot(ice, 'sex', 'weight', ax=ax, yrange=(0, 5),
             cats=df_raw['sex'].unique())
    savefig(f"sex_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = ice_predict(rf, X, 'pregnant', 'weight')
    ice_plot(ice, 'pregnant', 'weight', ax=ax, yrange=(-5, 35),
             cats=df_raw['pregnant'].unique())
    savefig(f"pregnant_vs_weight_pdp")
    plt.close()


def unsup_weight():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    stratpd_plot(X, y, 'education', 'weight', ax=axes[0,0], yrange=(-12, 0), alpha=.1, nlines=700, supervised=False)
    stratpd_plot(X, y, 'education', 'weight', ax=axes[0,1], yrange=(-12, 0), alpha=.1, nlines=700, supervised=True)

    catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[1,0],
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(-5,35), supervised=False)
    catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[1,1],
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(-5,35), supervised=True)

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)

    
    savefig(f"weight_unsup")
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

    fig, axes = plt.subplots(2, 4, figsize=(8,4))
    for i in range(1,4):
        axes[0,i].get_yaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)
    stratpd_plot(X, y, 'education', 'weight', ax=axes[0,0], yrange=(-12,0), alpha=.05, pdp_dot_size=10, show_ylabel=True,
                 ntrees=1, max_features=1.0, bootstrap=False)
    stratpd_plot(X, y, 'education', 'weight', ax=axes[0,1], yrange=(-12,0), alpha=.05, pdp_dot_size=10, show_ylabel=False,
                 ntrees=5, max_features='auto', bootstrap=True)
    stratpd_plot(X, y, 'education', 'weight', ax=axes[0,2], yrange=(-12,0), alpha=.05, pdp_dot_size=10, show_ylabel=False,
                 ntrees=10, max_features = 'auto', bootstrap = True)
    stratpd_plot(X, y, 'education', 'weight', ax=axes[0,3], yrange=(-12,0), alpha=.05, pdp_dot_size=10, show_ylabel=False,
                 ntrees=30, max_features='auto', bootstrap=True)

    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,0], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=True,
    #              ntrees=1, max_features=1.0, bootstrap=False)
    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,1], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=False,
    #              ntrees=5, max_features='auto', bootstrap=True)
    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,2], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=False,
    #              ntrees=10, max_features = 'auto', bootstrap = True)
    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,3], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=False,
    #              ntrees=30, max_features='auto', bootstrap=True)

    catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[1,0], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=True,
                     yrange=(0,35),
                     ntrees=1, max_features=1.0, bootstrap=False)
    catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[1,1], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
                     yrange=(0,35),
                     ntrees=5, max_features='auto', bootstrap=True)
    catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[1,2], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
                     yrange=(0,35),
                     ntrees=10, max_features='auto', bootstrap=True)
    catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[1,3], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
                     yrange=(0,35),
                     ntrees=30, max_features='auto', bootstrap=True)
    
    savefig(f"height_pregnant_vs_weight_ntrees")
    plt.close()

    # fig, axes = plt.subplots(1, 4, figsize=(8,2))
    # catstratpd_plot(X, y, 'sex', 'weight', ax=axes[0], alpha=.2, cats=df_raw['sex'].unique(), show_ylabel=True,
    #                  yrange=(0,5),
    #                  ntrees=1, max_features=1.0, bootstrap=False)
    # catstratpd_plot(X, y, 'sex', 'weight', ax=axes[1], alpha=.2, cats=df_raw['sex'].unique(), show_ylabel=False,
    #                  yrange=(0,5),
    #                  ntrees=5, max_features='auto', bootstrap=True)
    # catstratpd_plot(X, y, 'sex', 'weight', ax=axes[2], alpha=.2, cats=df_raw['sex'].unique(), show_ylabel=False,
    #                  yrange=(0,5),
    #                  ntrees=10, max_features='auto', bootstrap=True)
    # catstratpd_plot(X, y, 'sex', 'weight', ax=axes[3], alpha=.2, cats=df_raw['sex'].unique(), show_ylabel=False,
    #                  yrange=(0,5),
    #                  ntrees=30, max_features='auto', bootstrap=True)
    # 
    # savefig(f"sex_vs_weight_ntrees")
    # plt.close()
    #
    # fig, axes = plt.subplots(1, 4, figsize=(8,2))
    # catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[0], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=True,
    #                  yrange=(0,35),
    #                  ntrees=1, max_features=1.0, bootstrap=False)
    # catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[1], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
    #                  yrange=(0,35),
    #                  ntrees=5, max_features='auto', bootstrap=True)
    # catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[2], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
    #                  yrange=(0,35),
    #                  ntrees=10, max_features='auto', bootstrap=True)
    # catstratpd_plot(X, y, 'pregnant', 'weight', ax=axes[3], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
    #                  yrange=(0,35),
    #                  ntrees=30, max_features='auto', bootstrap=True)
    # 
    # savefig(f"pregnant_vs_weight_ntrees")
    # plt.close()


def additivity_data(n):
    x1 = np.random.uniform(-1, 1, size=n)
    x2 = np.random.uniform(-1, 1, size=n)

    y = x1*x1 + x2 + np.random.normal(0, 1, size=n)
    df = pd.DataFrame()
    df['x1'] = x1
    df['x2'] = x2
    df['y'] = y
    return df

def additivity():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    df = additivity_data(n=n)
    X = df.drop('y', axis=1)
    y = df['y']

    fig, axes = plt.subplots(2, 2, figsize=(4,4), sharey=True)
    stratpd_plot(X, y, 'x1', 'y', ax=axes[0,0],
              hires_threshold=10, yrange=(-1, 1), pdp_dot_size=3, alpha=.1, nlines=700)
    
    stratpd_plot(X, y, 'x2', 'y', ax=axes[1,0],
                 hires_threshold=10, pdp_dot_size=3, alpha=.1, nlines=700)
    
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = ice_predict(rf, X, 'x1', 'y', numx=20, nlines=700)
    ice_plot(ice, 'x1', 'y', ax=axes[0,1], yrange=(-1, 1), show_ylabel=False)
    
    ice = ice_predict(rf, X, 'x2', 'y', numx=20, nlines=700)
    ice_plot(ice, 'x2', 'y', ax=axes[1,1], yrange=(-2, 2), show_ylabel=False)

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)

    savefig(f"additivity")
    plt.close()


def bigX_data(n):
    x1 = np.random.uniform(-1, 1, size=n)
    x2 = np.random.uniform(-1, 1, size=n)
    x3 = np.random.uniform(-1, 1, size=n)

    y = 0.2 * x1 - 5 * x2 + 10 * x2 * np.where(x3 >= 0, 1, 0) + np.random.normal(0, 1, size=n)
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
    
    # Partial deriv is just 0.2 so this is correct. flat deriv curve, net effect line at slope .2
    # ICE is way too shallow and not line at n=1000 even
    fig, axes = plt.subplots(3, 2, figsize=(4, 6), sharey=True)
    stratpd_plot(X, y, 'x1', 'y', ax=axes[0,0], yrange=(-4,4), alpha=.1, nlines=700, pdp_dot_size=2)
    
    # Partial deriv wrt x2 is -5 plus 10 about half the time so about 0
    # Should not expect a criss-cross like ICE since deriv of 1_x3>=0 is 0 everywhere
    # wrt to any x, even x3. x2 *is* affecting y BUT the net effect at any spot
    # is what we care about and that's 0. Just because marginal x2 vs y shows non-
    # random plot doesn't mean that x2's net effect is nonzero. We are trying to
    # strip away x1/x3's effect upon y. When we do, x2 has no effect on y.
    # Key is asking right question. Don't look at marginal plot and say obvious.
    # Ask what is net effect at every x2? 0.
    stratpd_plot(X, y, 'x2', 'y', ax=axes[1,0], yrange=(-4,4), alpha=.1, nlines=700, pdp_dot_size=2)
    
    # Partial deriv wrt x3 of 1_x3>=0 is 0 everywhere so result must be 0
    stratpd_plot(X, y, 'x3', 'y', ax=axes[2,0], yrange=(-4,4), alpha=.1, nlines=700, pdp_dot_size=2)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")
    
    ice = ice_predict(rf, X, 'x1', 'y', numx=10)
    ice_plot(ice, 'x1', 'y', ax=axes[0,1], yrange=(-4,4))
    
    ice = ice_predict(rf, X, 'x2', 'y', numx=10)
    ice_plot(ice, 'x2', 'y', ax=axes[1,1], yrange=(-4,4))

    ice = ice_predict(rf, X, 'x3', 'y', numx=10)
    ice_plot(ice, 'x3', 'y', ax=axes[2,1], yrange=(-4,4))

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)
    axes[2,1].get_yaxis().set_visible(False)

    savefig(f"bigx")
    plt.close()


def unsup_boston():
    boston = load_boston()
    print(len(boston.data))
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    fig, axes = plt.subplots(2, 2, figsize=(4, 4))

    stratpd_plot(X, y, 'AGE', 'MEDV', ax=axes[0,0], yrange=(-20,20), supervised=False, show_xlabel=False)
    stratpd_plot(X, y, 'AGE', 'MEDV', ax=axes[0,1], yrange=(-20,20), supervised=True, show_xlabel=False)

    rf = RandomForestRegressor(n_estimators=100, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    axes[1,0].scatter(df['AGE'], y, s=5, alpha=.7)
    axes[1,0].set_ylabel('MEDV')
    axes[1,0].set_xlabel('AGE')

    ice = ice_predict(rf, X, 'AGE', 'MEDV', numx=10)
    ice_plot(ice, 'AGE', 'MEDV', ax=axes[1, 1], yrange=(-20,20))

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)

    
    savefig(f"boston_unsup")
    plt.close()


if __name__ == '__main__':
    # unsup_boston()
    # rent()
    # meta_rent()
    # unsup_rent()
    weight()
    # meta_weight()
    # unsup_weight()
    # weather()
    # additivity()
    # bigX()