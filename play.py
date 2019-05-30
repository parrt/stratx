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
from stratpd.featimp import *
from scipy.stats import spearmanr
from sklearn import svm

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
    # df['ID'] = range(100,100+n)
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

def dep_cars():
    df_cars = pd.read_csv("notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)
    df_cars.head(5)

    X = df_cars[['horsepower', 'weight']]
    y = df_cars['mpg']
    print(feature_corr_matrix(X))
    C = np.corrcoef(X.values.T)
    print(C)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    print(feature_dependence_matrix(X))

def dep_weight():
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']
    print(feature_corr_matrix(X))
    C = np.corrcoef(X.values.T)
    print(C)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    print(feature_dependence_matrix(X))

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

    plot_stratpd(X, y, 'education', 'weight', ax=axes[1][0],
                 yrange=(-12,0),
                 nlines = 500,
                 hires_threshold=100
                 )
    plot_stratpd(X, y, 'height', 'weight', ax=axes[2][0],
                 yrange=(0,160),
                 nlines = 1000,
                 hires_threshold=100
                 )
    plot_catstratpd(X, y, 'sex', 'weight', ax=axes[3][0],
                    alpha=.2,
                    cats=df_raw['sex'].unique(),
                    yrange=(0,5)
                    )
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[4][0],
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


def meta_weight():
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, axes = plt.subplots(4, 4, figsize=(8,8))

    plot_stratpd(X, y, 'education', 'weight', ax=axes[0][0], yrange=(-12, 0), alpha=.05, pdp_dot_size=10,
                 ntrees=1, max_features=1.0, bootstrap=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0][1], yrange=(-12, 0), alpha=.05, pdp_dot_size=10,
                 ntrees=5, max_features='auto', bootstrap=True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0][2], yrange=(-12, 0), alpha=.05, pdp_dot_size=10,
                 ntrees=20, max_features = 'auto', bootstrap = True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0][3], yrange=(-12, 0), alpha=.05, pdp_dot_size=10,
                 ntrees=50, max_features='auto', bootstrap=True)

    plot_stratpd(X, y, 'height', 'weight', ax=axes[1][0], yrange=(0, 160), alpha=.05, pdp_dot_size=10,
                 ntrees=1, max_features=1.0, bootstrap=False)
    plot_stratpd(X, y, 'height', 'weight', ax=axes[1][1], yrange=(0, 160), alpha=.05, pdp_dot_size=10,
                 ntrees=5, max_features='auto', bootstrap=True)
    plot_stratpd(X, y, 'height', 'weight', ax=axes[1][2], yrange=(0, 160), alpha=.05, pdp_dot_size=10,
                 ntrees=20, max_features = 'auto', bootstrap = True)
    plot_stratpd(X, y, 'height', 'weight', ax=axes[1][3], yrange=(0, 160), alpha=.05, pdp_dot_size=10,
                 ntrees=50, max_features='auto', bootstrap=True)


    plot_catstratpd(X, y, 'sex', 'weight', ax=axes[2][0], alpha=.2, cats=df_raw['sex'].unique(),
                    yrange=(0,5),
                    ntrees=1, max_features=1.0, bootstrap=False)
    plot_catstratpd(X, y, 'sex', 'weight', ax=axes[2][1], alpha=.2, cats=df_raw['sex'].unique(),
                    yrange=(0,5),
                    ntrees=5, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'sex', 'weight', ax=axes[2][2], alpha=.2, cats=df_raw['sex'].unique(),
                    yrange=(0,5),
                    ntrees=20, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'sex', 'weight', ax=axes[2][3], alpha=.2, cats=df_raw['sex'].unique(),
                    yrange=(0,5),
                    ntrees=50, max_features='auto', bootstrap=True)

    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[3][0], alpha=.2, cats=df_raw['pregnant'].unique(),
                    yrange=(0,35),
                    ntrees=1, max_features=1.0, bootstrap=False)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[3][1], alpha=.2, cats=df_raw['pregnant'].unique(),
                    yrange=(0,35),
                    ntrees=5, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[3][2], alpha=.2, cats=df_raw['pregnant'].unique(),
                    yrange=(0,35),
                    ntrees=20, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[3][3], alpha=.2, cats=df_raw['pregnant'].unique(),
                    yrange=(0,35),
                    ntrees=50, max_features='auto', bootstrap=True)


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

    # df_rent = df_rent.sample(n=8000)  # get a small subsample

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    supervised = True

    fig, axes = plt.subplots(4, 2, figsize=(8,16))
    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 0], alpha=.2, yrange=(0, 3000), nlines=1000, supervised=supervised)
    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 0], alpha=.2, yrange=(0, 3000), nlines=1000, supervised=supervised)
    plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 0], alpha=.2, yrange=(0, 3000), nlines=1000, supervised=supervised)
    plot_stratpd(X, y, 'longitude', 'price', ax=axes[3, 0], alpha=.2, nlines=1000, supervised=supervised)

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


def meta_rent():
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

    df_rent = df_rent.sample(n=8000, random_state=111)  # get a small subsample

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    supervised = True

    def onevar(colname, row, yrange):
        for i, t in enumerate([1, 1, 1, 1]):
            plot_stratpd(X, y, colname, 'price', ax=axes[row, i], alpha=.05,
                         yrange=yrange,
                         supervised=supervised,
                         ntrees=t)

    fig, axes = plt.subplots(4, 4, figsize=(8,8), sharey=True)
    onevar('bedrooms', row=0, yrange=(0,3000))
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(8,8), sharey=True)
    onevar('bathrooms', row=1, yrange=(0,3000))
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(8,8), sharey=True)
    onevar('latitude', row=2, yrange=(0,3000))
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(8,8), sharey=True)
    onevar('longitude', row=3)
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
    plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[1][0],
                 ntrees=1,
                 # min_samples_leaf=2,
                 hires_min_samples_leaf=13
                 , yrange=(-10,10))
    # plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[1][1],
    #              ntrees=5,
    #              # min_samples_leaf=5,
    #              hires_min_samples_leaf=13
    #              , yrange=(-10,10))
    # plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[2][0],
    #              ntrees=10,
    #              # min_samples_leaf=7,
    #              hires_min_samples_leaf=13
    #              , yrange=(-10,10))
    # plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[2][1],
    #              ntrees=20,
    #              # min_samples_leaf=20,
    #              hires_min_samples_leaf=13
    #              , yrange=(-10,10))

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
    avgtmp = df.groupby(['state','dayofyear'])[['temperature']].mean()
    # avgtmp.sort_values('dayofyear')
    avgtmp = avgtmp.reset_index()
    print(avgtmp.reset_index().head(10))
    ca = avgtmp.query('state=="CA"')
    co = avgtmp.query('state=="CO"')
    az = avgtmp.query('state=="AZ"')
    wa = avgtmp.query('state=="WA"')
    axes[3, 0].plot(ca['dayofyear'], ca['temperature'], label="CA")
    axes[3, 0].plot(co['dayofyear'], co['temperature'], label="CO")
    axes[3, 0].plot(az['dayofyear'], az['temperature'], label="AZ")
    axes[3, 0].plot(wa['dayofyear'], wa['temperature'], label="WA")
    # axes[3, 0].plot(df.loc[df['state'] == 'CA', 'dayofyear'],
    #                 df.loc[df['state'] == 'CA', 'temperature'], label="CA")
    # axes[3, 0].plot(df.loc[df['state'] == 'CO', 'dayofyear'],
    #                 df.loc[df['state'] == 'CO', 'temperature'], label="CO")
    # axes[3, 0].plot(df.loc[df['state'] == 'AZ', 'dayofyear'],
    #                 df.loc[df['state'] == 'AZ', 'temperature'], label="AZ")
    # axes[3, 0].plot(df.loc[df['state'] == 'WA', 'dayofyear'],
    #                 df.loc[df['state'] == 'WA', 'temperature'], label="WA")
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


def multi_joint_distr():
    np.random.seed(42)
    df = pd.DataFrame(np.random.multivariate_normal([6, 6, 6, 6],
                                                    [
                                                        [1,  4, .5,  3],
                                                        [4,  1,  2,  .3],
                                                        [.5, 2,  1,  .8],
                                                        [3,  .3, .8,  1]
                                                    ],
                                                    1000),
                      columns=['x1','x2','x3','x4'])
    df['y'] = df['x1'] + df['x2'] + df['x3'] + df['x4']
    X = df.drop('y', axis=1)
    y = df['y']

    r = LinearRegression()
    r.fit(X, y)
    print(r.coef_) # should be all 1s

    yrange = (-2, 15)
    min_samples_leaf = 60

    fig, axes = plt.subplots(5, 4, figsize=(8,8))

    axes[0,0].scatter(X['x1'],y,s=5, alpha=.3)
    axes[0, 0].set_xlim(0,12)
    axes[0, 0].set_ylim(0,45)
    axes[0,1].scatter(X['x2'],y,s=5, alpha=.3)
    axes[0, 1].set_xlim(0,12)
    axes[0, 1].set_ylim(3,45)
    axes[0, 2].scatter(X['x3'],y,s=5, alpha=.3)
    axes[0, 2].set_xlim(0,12)
    axes[0, 2].set_ylim(3,45)
    axes[0, 3].scatter(X['x4'],y,s=5, alpha=.3)
    axes[0, 3].set_xlim(0,12)
    axes[0, 3].set_ylim(3,45)

    # axes[0, 0].set_xlabel("x1")
    # axes[0, 1].set_xlabel("x2")
    # axes[0, 2].set_xlabel("x3")
    axes[0, 0].set_ylabel("y")

    for i in range(5):
        for j in range(1,4):
            axes[i,j].get_yaxis().set_visible(False)

    uniqx, pdp = \
        plot_stratpd(X, y, 'x1', 'y', ax=axes[1,0], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf=min_samples_leaf,
                     hires_threshold=200,
                     yrange=yrange, show_xlabel=False, show_ylabel=True)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,0].text(3,7,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp = \
        plot_stratpd(X, y, 'x2', 'y', ax=axes[1,1], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf=min_samples_leaf,
                     hires_threshold=200,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,1].text(3,7,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp = \
        plot_stratpd(X, y, 'x3', 'y', ax=axes[1,2], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf=min_samples_leaf,
                     hires_threshold=200,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,2].text(3,7,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp = \
        plot_stratpd(X, y, 'x4', 'y', ax=axes[1,3], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf=min_samples_leaf,
                     hires_threshold=200,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,3].text(3,7,f"Slope={r.coef_[0]:.2f}")

    axes[1,0].text(3,10,'StratPD', horizontalalignment='left')
    axes[1,1].text(3,10,'StratPD', horizontalalignment='left')
    axes[1,2].text(3,10,'StratPD', horizontalalignment='left')
    axes[1,3].text(3,10,'StratPD', horizontalalignment='left')

    regrs = [
        RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True),
        svm.SVR(gamma='scale'),
        LinearRegression()]
    row = 2
    for regr in regrs:
        regr.fit(X, y)
        rname = regr.__class__.__name__
        if rname=='SVR':
            rname = "SVM"
        if rname=='RandomForestRegressor':
            rname = "RF"
        if rname=='LinearRegression':
            rname = 'Linear'

        show_xlabel = True if row==4 else False

        axes[row,0].text(3, 10, rname, horizontalalignment='left')
        axes[row,1].text(3, 10, rname, horizontalalignment='left')
        axes[row,2].text(3, 10, rname, horizontalalignment='left')
        axes[row,3].text(3, 10, rname, horizontalalignment='left')
        ice = ice_predict(regr, X, 'x1', 'y')
        ice_plot(ice, 'x1', 'y', ax=axes[row, 0], xrange=(0,12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=True)
        ice = ice_predict(regr, X, 'x2', 'y')
        ice_plot(ice, 'x2', 'y', ax=axes[row, 1], xrange=(0,12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=False)
        ice = ice_predict(regr, X, 'x3', 'y')
        ice_plot(ice, 'x3', 'y', ax=axes[row, 2], xrange=(0,12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=False)
        ice = ice_predict(regr, X, 'x4', 'y')
        ice_plot(ice, 'x4', 'y', ax=axes[row, 3], xrange=(0,12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=False)
        row += 1

    plt.tight_layout()
    plt.show()

def imp_boston():
    boston = load_boston()
    print(len(boston.data))
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    colnames = X.columns.values
    ncols = len(colnames)
    # fig, axes = plt.subplots(1, 1)

    plot_strat_importances(X, y, 'MEDV')

    plt.tight_layout()
    plt.show()

def imp_cars():
    df_cars = pd.read_csv("notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)
    df_cars.head(5)

    catencoders = df_string_to_cat(df_cars)
    df_cat_to_catcode(df_cars)

    c = .5
    df_cars['noise'] = np.random.random(len(df_cars)) * c

    # X = df_cars[['horsepower', 'weight', 'noise']]
    X = df_cars.drop(['mpg','name'], axis=1)
    y = df_cars['mpg']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,4))
    plot_strat_importances(X, y, ax=axes[0], min_samples_leaf=20)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    I = importances(rf, X, y)
    plot_importances(I, ax=axes[1])

    I = dropcol_importances(rf, X, y)
    plot_importances(I, ax=axes[2])

    axes[0].set_title('StratIm')
    axes[1].set_title('Permutation')
    axes[2].set_title('Dropcol')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # imp_boston()
    imp_cars()
    # multi_joint_distr()
    # rent()
    # meta_rent()
    # weight()
    # dep_weight()
    # dep_cars()
    # meta_weight()
    # weather()