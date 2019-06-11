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
from stratx.partdep import *
from stratx.ice import *
from stratx.featimp import *
from scipy.stats import spearmanr
from sklearn import svm
import shap

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
    # np.random.seed(42)
    df_raw = toy_weight_data(3000)
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
                 alpha=.1
                 )
    plot_stratpd(X, y, 'height', 'weight', ax=axes[2][0],
                 yrange=(0,160),
                 nlines = 1000,
                 )
    plot_catstratpd(X, y, 'sex', 'weight', ax=axes[3][0],
                    alpha=1,
                    min_samples_leaf_partition=5,
                    cats=df_raw['sex'].unique(),
                    # zero_center=True
                    yrange=(0,5)
                    )
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[4][0],
                    min_samples_leaf_partition=5,
                    alpha=1,
                    cats=df_raw['pregnant'].unique(),
                    # zero_center=True
                    yrange=(0,35)
                    )

    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    ice = predict_ice(rf, X, 'education', 'weight')
    plot_ice(ice, 'education', 'weight', ax=axes[1,1], yrange=(-12, 0))

    ice = predict_ice(rf, X, 'height', 'weight')
    plot_ice(ice, 'height', 'weight', ax=axes[2,1], yrange=(0, 160))

    ice = predict_catice(rf, X, 'sex', 'weight')
    plot_catice(ice, 'sex', 'weight', cats=df_raw['sex'].unique(), ax=axes[3,1], yrange=(0, 5), pdp_marker_width=15)

    ice = predict_catice(rf, X, 'pregnant', 'weight', cats=df_raw['pregnant'].unique())
    plot_catice(ice, 'pregnant', 'weight', cats=df_raw['pregnant'].unique(), ax=axes[4,1], yrange=(-5, 35), pdp_marker_width=15)

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

    df_rent = df_rent.sample(n=10000)  # get a small subsample

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    supervised = True

    fig, axes = plt.subplots(4, 2, figsize=(8,16))
    min_samples_leaf = 10#0.0001
    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 0],
                 min_samples_leaf_partition=min_samples_leaf,
                 alpha=.2, yrange=(0, 3000), nlines=1000, supervised=supervised)
    # plot_catstratpd(X, y, 'bedrooms', 'price', cats=np.unique(X['bedrooms']),
    #                 ax=axes[0, 1],
    #                 min_samples_leaf=min_samples_leaf,
    #                 alpha=1,
    #                 # yrange=(0, 3000),
    #                 sort=None)

    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 0],
                 min_samples_leaf_partition=min_samples_leaf,
                 alpha=.2, yrange=(-500, 3000), nlines=1000, supervised=supervised)
    # plot_catstratpd(X, y, 'bathrooms', 'price', cats=np.unique(X['bedrooms']),
    #                 ax=axes[1, 1],
    #                 min_samples_leaf=min_samples_leaf,
    #                 alpha=1,
    #                 # yrange=(0, 3000),
    #                 sort=None)

    # plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 0],
    #              min_samples_leaf=min_samples_leaf,
    #              alpha=.2, yrange=(0, 3000), nlines=1000, supervised=supervised)
    # plot_stratpd(X, y, 'longitude', 'price', ax=axes[3, 0],
    #              min_samples_leaf=min_samples_leaf,
    #              alpha=.2, nlines=1000, supervised=supervised)

    # rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    # rf.fit(X, y)

    # rf = Lasso()
    # rf.fit(X, y)

    # ice = predict_ice(rf, X, 'bedrooms', 'price')
    # plot_ice(ice, 'bedrooms', 'price', ax=axes[0, 1], alpha=.05, yrange=(0,3000))
    # ice = predict_ice(rf, X, 'bathrooms', 'price')
    # plot_ice(ice, 'bathrooms', 'price', alpha=.05, ax=axes[1, 1])
    # ice = predict_ice(rf, X, 'latitude', 'price')
    # plot_ice(ice, 'latitude', 'price', ax=axes[2, 1], alpha=.05, yrange=(0,1700))
    # ice = predict_ice(rf, X, 'longitude', 'price')
    # plot_ice(ice, 'longitude', 'price', ax=axes[3, 1], alpha=.05, yrange=(-750,3000))

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

    # df_rent = df_rent.sample(n=8000, random_state=111)  # get a small subsample

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    plot_meta(X, y, colnames=['bedrooms','bathrooms','latitude','longitude'])


def plot_meta(X, y, colnames, yrange=None):
    min_samples_leaf_values = [2, 5, 10, 30, 50, 100, 200]

    min_r2_hires = .2

    nrows = len(colnames)
    ncols = len(min_samples_leaf_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    row = 0
    for colname in colnames:
        col = 0
        for meta in min_samples_leaf_values:
            print(f"---------- min_samples_leaf={meta} ----------- ")
            plot_stratpd(X, y, colname, 'price', ax=axes[row, col],
                         min_r2_hires=min_r2_hires,
                         min_samples_leaf_partition=meta,
                         yrange=yrange,
                         ntrees=1)
            axes[row, col].set_title(f"samples/leaf={meta}, min R^2={min_r2_hires}")
            col += 1
        row += 1

    plt.tight_layout()
    plt.show()


def toy_weather_data():
    def temp(x): return np.sin((x+365/2)*(2*np.pi)/365)*10
    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))
    states = ['CA', 'CO', 'AZ', 'WA']
    bases = [70, 40, 90, 60]
    # np.random.choice(states, len(df))
    states_dfs = []
    for base,state in zip(bases,states):
        df = pd.DataFrame()
        df['dayofyear'] = range(1,365+1)
        df['state'] = state
        df['temperature'] = temp(df['dayofyear']) + base + noise(state)
        states_dfs.append(df)

    return pd.concat(states_dfs, axis=0)


def weather():
    # np.random.seed(66)
    nyears = 5
    years = []
    for y in range(1980,1980+nyears):
        df_ = toy_weather_data()
        df_['year']=y
        years.append(df_)

    df_raw = pd.concat(years, axis=0)

    # df_raw.drop('year', axis=1, inplace=True)
    df = df_raw.copy()
    print(df.head(5))

    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    X = df.drop('temperature', axis=1)
    y = df['temperature']

    fig, axes = plt.subplots(4, 5, figsize=(22, 10))

    min_samples_leaf = 2

    j = 0
    for sz in [0.05, .10, .20, .30, .40]:
        print(f"---------- {sz} samples/leaf -------------")

        if False:
            all_x = sorted(np.unique(X['dayofyear']))
            combined_curve = defaultdict(float)
            nboots = 1
            for i in range(nboots): # bootstrap
                df_ = df.sample(n=len(X), replace=True)
                X_ = df_.drop('temperature', axis=1)
                y_ = df_['temperature']
                uniq_x, curve, r2_at_x = \
                    plot_stratpd(X_, y_, 'dayofyear', 'temperature', ax=axes[0][j],
                                 min_samples_leaf_partition=2,
                                 min_samples_leaf_piecewise=sz,
                                 min_r2_hires=0.35,
                                 alpha=.5,
                                 pdp_dot_size=0,
                                 yrange=(-10,10))

                for x_,y_ in zip(uniq_x,curve):
                    combined_curve[x_] += y_
            all_x = np.array(list(combined_curve.keys()))
            combined_curve = np.array(list(combined_curve.values()))
            combined_curve /= nboots
            axes[0,j].scatter(all_x, combined_curve, c='red', s=3)

        plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[0][j],
                     min_samples_leaf_partition=min_samples_leaf,
                     min_samples_leaf_piecewise=sz,
                     # hires_nbins=sz,
                     alpha=.3,
                     yrange=(-20,20))

        axes[0][j].set_title(f"{min_samples_leaf} samples/leaf\n{sz} $x_c$ samples/leaf")
        j += 1

    plt.title(f"{nyears} years")


    # plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[1][1],
    #              ntrees=5,
    #              # min_samples_leaf=5,
    #              min_samples_leaf_hires=13
    #              , yrange=(-10,10))
    # plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[2][0],
    #              ntrees=10,
    #              # min_samples_leaf=7,
    #              min_samples_leaf_hires=13
    #              , yrange=(-10,10))
    # plot_stratpd(X, y, 'dayofyear', 'temperature', ax=axes[2][1],
    #              ntrees=20,
    #              # min_samples_leaf=20,
    #              min_samples_leaf_hires=13
    #              , yrange=(-10,10))

    # catstratpd_plot(X, y, 'state', 'temperature', cats=catencoders['state'],
    #                 ax=axes[2][0])  # , yrange=(0,160))
    #
    # rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
    # rf.fit(X, y)
    # #
    # ice = predict_ice(rf, X, 'dayofyear', 'temperature')
    # plot_ice(ice, 'dayofyear', 'temperature', ax=axes[3, 2])  # , yrange=(-12,0))
    #
    # ice = predict_ice(rf, X, 'state', 'temperature')
    # plot_ice(ice, 'state', 'temperature', cats=catencoders['state'],
    #          ax=axes[3, 3])  # , yrange=(-12,0))
    #
    if True:
        df = df_raw.copy()
        avgtmp = df.groupby(['state','dayofyear'])[['temperature']].mean()
        # avgtmp.sort_values('dayofyear')
        avgtmp = avgtmp.reset_index()
        ca = avgtmp.query('state=="CA"')
        co = avgtmp.query('state=="CO"')
        az = avgtmp.query('state=="AZ"')
        wa = avgtmp.query('state=="WA"')
        axes[3, 0].plot(ca['dayofyear'], ca['temperature'], label="CA", lw=.5)
        axes[3, 0].plot(co['dayofyear'], co['temperature'], label="CO", lw=.5)
        axes[3, 0].plot(az['dayofyear'], az['temperature'], label="AZ", lw=.5)
        axes[3, 0].plot(wa['dayofyear'], wa['temperature'], label="WA", lw=.5)
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
    # np.random.seed(42)
    n = 1000
    df = pd.DataFrame(np.random.multivariate_normal([6, 6, 6, 6],
                                                    [
                                                        [1,  5, .5,  3],
                                                        [5,  1,  2,  .3],
                                                        [.5, 2,  1,  .8],
                                                        [3,  .3, .8,  1]
                                                    ],
                                                    n),
                      columns=['x1','x2','x3','x4'])
    df['y'] = df['x1'] + df['x2'] + df['x3'] + df['x4']
    X = df.drop('y', axis=1)
    y = df['y']

    r = LinearRegression()
    r.fit(X, y)
    print(r.coef_) # should be all 1s

    yrange = (-2, 15)

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

    min_samples_leaf = .005
    min_r2_hires = .3
    hires_window_width = .3
    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x1', 'y', ax=axes[1,0], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf,
                     yrange=yrange, show_xlabel=False, show_ylabel=True)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,0].text(3,7,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x2', 'y', ax=axes[1,1], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,1].text(3,7,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x3', 'y', ax=axes[1,2], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,2].text(3,7,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x4', 'y', ax=axes[1,3], xrange=(0,12),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf,
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
        ice = predict_ice(regr, X, 'x1', 'y')
        plot_ice(ice, 'x1', 'y', ax=axes[row, 0], xrange=(0, 12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=True)
        ice = predict_ice(regr, X, 'x2', 'y')
        plot_ice(ice, 'x2', 'y', ax=axes[row, 1], xrange=(0, 12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=False)
        ice = predict_ice(regr, X, 'x3', 'y')
        plot_ice(ice, 'x3', 'y', ax=axes[row, 2], xrange=(0, 12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=False)
        ice = predict_ice(regr, X, 'x4', 'y')
        plot_ice(ice, 'x4', 'y', ax=axes[row, 3], xrange=(0, 12), yrange=yrange, show_xlabel=show_xlabel, show_ylabel=False)
        row += 1

    plt.tight_layout()
    plt.show()

def imp_boston():
    boston = load_boston()
    print(len(boston.data))
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target
    c = .5
    df['noise'] = np.random.random(len(df)) * c

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    plot_all_imp(X, y)


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
    plot_all_imp(X, y)


def plot_all_imp(X, y):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12,4))
    plot_strat_importances(X, y, ax=axes[0], min_samples_leaf=20)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=False)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    shap_values = np.mean(np.abs(shap_values), axis=0) # measure avg magnitude
    I = pd.DataFrame(data={'Feature':X.columns, 'Importance':shap_values})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    plot_importances(I, ax=axes[1])

    I = importances(rf, X, y)
    plot_importances(I, ax=axes[2])

    I = dropcol_importances(rf, X, y)
    plot_importances(I, ax=axes[3])

    I = pd.DataFrame(data={'Feature':X.columns, 'Importance':rf.feature_importances_})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    plot_importances(I, ax=axes[4])

    axes[0].set_title('StratIm')
    axes[1].set_title('SHAP')
    axes[2].set_title('Permutation')
    axes[3].set_title('Dropcol')
    axes[4].set_title('Gini')
    plt.tight_layout()
    plt.show()


def additivity_data(n, sd=1.0):
    x1 = np.random.uniform(-1, 1, size=n)
    x2 = np.random.uniform(-1, 1, size=n)

    y = x1**2 + x2 + np.random.normal(0, sd, size=n)
    # y = x1**2 #+ np.random.normal(0, 1, size=n)
    df = pd.DataFrame()
    df['x1'] = x1
    df['x2'] = x2
    df['y'] = y
    return df

def meta_additivity():
    # np.random.seed(99)
    n = 1000
    df = additivity_data(n=n, sd=1)
    X = df.drop('y', axis=1)
    y = df['y']

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=False)

    axes[0,0].scatter(X['x1'], y, alpha=.12, label=None)
    axes[0,0].set_xlabel("x1")
    axes[0,0].set_ylabel("y")
    axes[1,0].scatter(X['x2'], y, alpha=.12, label=None)
    axes[1,0].set_xlabel("x2")
    axes[1,0].set_ylabel("y")

    # min_samples_leaf_hires = 50//3
    min_samples_leaf = 20
    plot_stratpd(X, y, 'x1', 'y', ax=axes[0, 1],
                 min_samples_leaf_partition=min_samples_leaf,
                 min_samples_leaf_piecewise=.4,
                 yrange=(-1, 1),
                 pdp_dot_size=2, alpha=.4)

    rtreeviz_univar(axes[1, 1],
                    X['x2'], y,
                    min_samples_leaf=min_samples_leaf,
                    feature_name='x2',
                    target_name='y',
                    fontsize=10, show={'splits'})
    axes[1,1].set_xlabel("x2")
    axes[1,1].set_ylabel("y")

    axes[0,0].set_title("$y = x_1^2 + x_2 + \epsilon$")
    # axes[0,1].set_title(f"leaf sz {min_samples_leaf}, hires {min_r2_hires}\nhires h {hires_window_width}")

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    # axes[0, 1].get_yaxis().set_visible(False)
    # axes[1, 1].get_yaxis().set_visible(False)


    plt.show()
    plt.close()

if __name__ == '__main__':
    # meta_additivity()
    # imp_cars()
    multi_joint_distr()
    # rent()
    # meta_rent()
    # weight()
    # dep_weight()
    # dep_cars()
    # meta_weight()
    weather()