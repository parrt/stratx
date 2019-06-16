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
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from pdpbox import pdp
from rfpimp import *
from scipy.integrate import cumtrapz
from stratx.partdep import *
from stratx.ice import *
import inspect

# This genfigs.py code is just demonstration code to generate figures for the paper.
# There are lots of programming sins committed here; to not take this to be
# our idea of good code. ;)

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

def addnoise(df, n=1, c=0.5, prefix=''):
    if n==1:
        df[f'{prefix}noise'] = np.random.random(len(df)) * c
        return
    for i in range(n):
        df[f'{prefix}noise{i+1}'] = np.random.random(len(df)) * c

def fix_missing_num(df, colname):
    df[colname+'_na'] = pd.isnull(df[colname])
    df[colname].fillna(df[colname].median(), inplace=True)

def savefig(filename, pad=0):
    plt.tight_layout(pad=pad, w_pad=0, h_pad=0)
    plt.savefig(f"images/{filename}.pdf")
    plt.savefig(f"images/{filename}.png", dpi=300)
    plt.close()


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
                   - df['education']*1.5
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
    df_rent = df_rent.sample(n=10_000)  # get a small subsample since SVM is slowwww
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    figsize = (5,4)
    colname='bedrooms'

    fig, axes = plt.subplots(2,2, figsize=figsize)
    avg_per_baths = df_rent.groupby(colname).mean()['price']
    axes[0,0].scatter(df_rent[colname], df_rent['price'], alpha=0.07, s=5)#, label="observation")
    axes[0,0].scatter(np.unique(df_rent[colname]), avg_per_baths, s=6, c='black', label="average price/{colname}")
    axes[0,0].set_ylabel("price")#, fontsize=12)
    axes[0,0].set_ylim(0,10_000)
    # ax.legend()

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    plot_ice(ice, colname, 'price', alpha=.05, ax=axes[0, 1], show_xlabel=False, show_ylabel=False)
    axes[0,1].set_ylim(-1000,5000)

    nfeatures = 4
    m = svm.SVR(gamma=1 / nfeatures)
    # m = KNeighborsRegressor()
    # m = Lasso()
    m.fit(X, y)

    ice = predict_ice(m, X, colname, 'price', nlines=1000)
    plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 0], show_ylabel=True)
    axes[1,0].set_ylim(-1000,5000)

    plot_stratpd(X, y, colname, 'price', ax=axes[1, 1], alpha=.2, show_ylabel=False)
    axes[1,1].set_ylim(-1000,5000)

    #axes[1,1].get_yaxis().set_visible(False)

    savefig(f"{colname}_vs_price")
    plt.close()


def rent_int():
    # np.random.seed(42)
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent.sample(n=10_000)  # get a small subsample since SVM is slowwww
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    figsize = (5,4)

    fig, axes = plt.subplots(2,3, figsize=(6,4))

    avg_per_baths = df_rent.groupby('bedrooms').mean()['price']
    axes[0,0].scatter(df_rent['bedrooms'], df_rent['price'], alpha=0.07, s=5)#, label="observation")
    axes[0,0].scatter(np.unique(df_rent['bedrooms']), avg_per_baths, s=6, c='black', label="average price/bedrooms")
    axes[0,0].set_xlabel("bedrooms")#, fontsize=12)
    axes[0,0].set_ylabel("price")#, fontsize=12)
    axes[0,0].set_ylim(0,10_000)
    avg_per_baths = df_rent.groupby('bathrooms').mean()['price']
    axes[1,0].scatter(df_rent['bathrooms'], df_rent['price'], alpha=0.07, s=5)#, label="observation")
    axes[1,0].scatter(np.unique(df_rent['bathrooms']), avg_per_baths, s=6, c='black', label="average price/bathrooms")
    axes[1,0].set_xlabel("bathrooms")#, fontsize=12)
    axes[1,0].set_ylabel("price")#, fontsize=12)
    axes[1,0].set_ylim(0,10_000)

    stratpd_min_samples_leaf_partition = 10 #  the default
    catstratpd_min_samples_leaf_partition = 10

    plot_stratpd(X, y, 'bedrooms', 'price',
                 min_samples_leaf_partition=stratpd_min_samples_leaf_partition,
                 ax=axes[0, 1], alpha=.2, show_ylabel=False)
    axes[0,1].set_ylim(-500,5000)

    plot_catstratpd(X, y, 'bedrooms', 'price', cats=np.unique(X['bedrooms']),
                    min_samples_leaf_partition=catstratpd_min_samples_leaf_partition,
                    ax=axes[0, 2], alpha=.2, show_ylabel=False,
                    sort=None)
    axes[0,2].set_ylim(-500,5000)

    plot_stratpd(X, y, 'bathrooms', 'price',
                 min_samples_leaf_partition=stratpd_min_samples_leaf_partition,
                 ax=axes[1, 1], alpha=.2, show_ylabel=False)
    axes[1,1].set_ylim(-500,5000)

    X['bathrooms'] = X['bathrooms'].astype(str)
    baths = np.unique(X['bathrooms'])
    plot_catstratpd(X, y, 'bathrooms', 'price', cats=baths,
                    min_samples_leaf_partition=catstratpd_min_samples_leaf_partition,
                    ax=axes[1, 2], alpha=.2, show_ylabel=False,
                    sort=None)
    axes[1,2].set_ylim(-500,5000)

    axes[0,0].set_title("Scatter")#, fontsize=12)
    axes[0,1].set_title("StratPD")#, fontsize=12)
    axes[0,2].set_title("CatStratPD")#, fontsize=12)

    savefig(f"rent_intcat")
    plt.close()

def plot_with_noise_col(df, colname):
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
    features_with_noise = ['bedrooms', 'bathrooms', 'latitude', 'longitude', colname + '_noise']

    type = "noise"

    fig, axes = plt.subplots(2,2, figsize=(5,5), sharey=True, sharex=True)

    df = df.copy()
    addnoise(df, n=1, c=50, prefix=colname + '_')

    X = df[features]
    y = df['price']

    # STRATPD ON ROW 1
    X = df[features]
    y = df['price']
    uniq_x, curve, r2_at_x = \
        plot_stratpd(X, y, colname, 'price', ax=axes[0, 0], alpha=.15, show_xlabel=True, show_ylabel=False)
    axes[0, 0].set_ylim(-1000,5000)
    axes[0, 0].set_title(f"StratPD")

    X = df[features_with_noise]
    y = df['price']
    plot_stratpd(X, y, colname, 'price', ax=axes[0, 1], alpha=.15, show_ylabel=False)
    axes[0, 1].set_ylim(-1000,5000)
    axes[0, 1].set_title(f"StratPD w/{type} col")

    # ICE ON ROW 2
    X = df[features]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    # do it w/o dup'd column
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    uniq_x, pdp_curve = \
        plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 0], show_xlabel=True)
    axes[1,0].set_ylim(-1000,5000)
    axes[1, 0].set_title(f"PD/ICE")

    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlim(0,6)

    X = df[features_with_noise]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    uniq_x_, pdp_curve_ = \
        plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1,1], show_xlabel=True, show_ylabel=False)
    axes[1,1].set_ylim(-1000,5000)
    axes[1, 1].set_title(f"PD/ICE w/{type} col")
    # print(f"max ICE curve {np.max(pdp_curve):.0f}, max curve with dup {np.max(pdp_curve_):.0f}")

    axes[0,0].get_xaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(False)


def plot_with_dup_col(df, colname):
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
    features_with_dup = ['bedrooms', 'bathrooms', 'latitude', 'longitude', colname + '_dup']

    fig, axes = plt.subplots(2,3, figsize=(7.5,5), sharey=True, sharex=True)

    type = "dup"
    verbose=False

    df = df.copy()
    df[colname + '_dup'] = df[colname]
    # df_rent[colname+'_dupdup'] = df_rent[colname]

    # STRATPD ON ROW 1
    X = df[features]
    y = df['price']
    print(f"shape is {X.shape}")
    uniq_x, curve, r2_at_x = \
        plot_stratpd(X, y, colname, 'price', ax=axes[0, 0], alpha=.15, show_xlabel=True, show_ylabel=True,
                     verbose=verbose)
    axes[0, 0].set_ylim(-1000,5000)
    axes[0, 0].set_title(f"StratPD")

    X = df[features_with_dup]
    y = df['price']
    print(f"shape with dup is {X.shape}")
    plot_stratpd(X, y, colname, 'price', ax=axes[0, 1], alpha=.15, show_ylabel=False,
                 verbose=verbose)
    axes[0, 1].set_ylim(-1000,5000)
    axes[0, 1].set_title(f"StratPD w/{type} col")

    uniq_x_, curve_, r2_at_x_ = \
        plot_stratpd(X, y, colname, 'price', ax=axes[0, 2], alpha=.15, show_xlabel=True, show_ylabel=False,
                 ntrees=15,
                 max_features=2,
                 bootstrap=False,
                 verbose=verbose
                 )
    axes[0, 2].set_ylim(-1000,5000)
    axes[0, 2].set_title(f"StratPD w/{type} col")
    axes[0, 2].text(.2,4000, "ntrees=15")
    axes[0, 2].text(.2,3500, "max features per split=2")

    # ICE ON ROW 2
    X = df[features]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)

    # do it w/o dup'd column
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    uniq_x, pdp_curve = \
        plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 0], show_xlabel=False)
    axes[1,0].set_ylim(-1000,5000)
    axes[1, 0].set_title(f"PD/ICE")

    for i in range(2):
        for j in range(3):
            axes[i,j].set_xlim(0,6)

    # with dup'd column
    X = df[features_with_dup]
    y = df['price']
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    uniq_x_, pdp_curve_ = \
        plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1,1], show_xlabel=True, show_ylabel=False)
    axes[1,1].set_ylim(-1000,5000)
    axes[1, 1].set_title(f"PD/ICE w/{type} col")
    # print(f"max ICE curve {np.max(pdp_curve):.0f}, max curve with dup {np.max(pdp_curve_):.0f}")

    axes[1, 2].set_title(f"PD/ICE w/{type} col")
    axes[1, 2].text(.2, 4000, "Cannot compensate")

    print(f"max curve {np.max(curve):.0f}, max curve with dup {np.max(curve_):.0f}")

    axes[1,2].get_xaxis().set_visible(False)
    axes[1,2].get_yaxis().set_visible(False)

    axes[0,0].get_xaxis().set_visible(False)
    axes[0,1].get_xaxis().set_visible(False)


def rent_extra_cols():
    print(f"----------- {inspect.stack()[0][3]} -----------")

    df_rent = load_rent()
    # df_rent = df_rent.sample(n=10_000)  # get a small subsample

    colname = 'bedrooms'
    plot_with_dup_col(df_rent, colname)
    savefig(f"{colname}_vs_price_dup")
    # plt.tight_layout()
    # plt.show()

    colname = 'bedrooms'
    plot_with_noise_col(df_rent, colname)
    savefig(f"{colname}_vs_price_noise")

    colname = 'bathrooms'
    plot_with_dup_col(df_rent, colname)
    savefig(f"{colname}_vs_price_dup")

    colname = 'bathrooms'
    plot_with_noise_col(df_rent, colname)
    savefig(f"{colname}_vs_price_noise")


def rent_ntrees():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent.sample(n=10_000)  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    supervised = True

    def onevar(colname, row, yrange=None):
        alpha = 0.05
        for i, t in enumerate([1, 5, 10, 30]):
            plot_stratpd(X, y, colname, 'price', ax=axes[row, i], alpha=alpha,
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


def meta_boston():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    boston = load_boston()
    print(len(boston.data))
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    yranges = [(-30,0), (0,30), (-8,8), (-11, 0)]

    for min_r2_hires in [.01,.1,.2,.5, 1.0]:
        for min_samples_leaf_piecewise in [.01, .1, .2, .3]:
            plot_meta(X, y, colnames=['LSTAT', 'RM', 'CRIM', 'DIS'], targetname='MEDV',
                      min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                      yranges=yranges)
            savefig(f"meta_boston_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_piecewise:.2f}")


def marginal_plot(X, y, colname, targetname, ax=None):
    ax.scatter(X[colname], y, alpha=.10, label=None)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    col = X[colname]

    r = LinearRegression()
    r.fit(X[[colname]], y)
    xcol = np.linspace(np.min(col), np.max(col), num=100)
    yhat = r.predict(xcol.reshape(-1, 1))
    ax.plot(xcol, yhat, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    ax.text(min(xcol)*1.02, max(y)*.95, f"$\\beta_{{{colname}}}$={r.coef_[0]:.3f}")


def plot_meta(X, y, colnames, targetname, min_samples_leaf_piecewise, yranges=None):
    min_samples_leaf_values = [2, 5, 10, 30, 50, 100, 200]

    nrows = len(colnames)
    ncols = len(min_samples_leaf_values)
    fig, axes = plt.subplots(nrows, ncols+2, figsize=((ncols+2)*2.5, nrows*2.5))

    if yranges is None:
        yranges = [None]*len(colnames)

    row = 0
    for i,colname in enumerate(colnames):
        marginal_plot(X, y, colname, targetname, ax=axes[row, 0])
        col = 2
        for meta in min_samples_leaf_values:
            print(f"---------- min_samples_leaf={meta}, hires leafsz={min_samples_leaf_piecewise:.2f} ----------- ")
            plot_stratpd(X, y, colname, targetname, ax=axes[row, col],
                         min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                         min_samples_leaf_partition=meta,
                         yrange=yranges[i],
                         ntrees=1)
            axes[row, col].set_title(f"leafsz={meta}, hires leafsz={min_samples_leaf_piecewise:.2f}",
                                     fontsize=9)
            col += 1
        row += 1

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    row = 0
    for i,colname in enumerate(colnames):
        ice = predict_ice(rf, X, colname, targetname)
        plot_ice(ice, colname, targetname, ax=axes[row, 1])
        row += 1


def unsup_rent():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_rent = load_rent()
    df_rent = df_rent.sample(n=9000, random_state=111)  # get a small subsample

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    fig, axes = plt.subplots(3, 2, figsize=(4,6))

    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 0], alpha=.2, supervised=False)
    plot_stratpd(X, y, 'bedrooms', 'price', ax=axes[0, 1], alpha=.2, supervised=True)

    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 0], alpha=.2, supervised=False)
    plot_stratpd(X, y, 'bathrooms', 'price', ax=axes[1, 1], alpha=.2, supervised=True)

    plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 0], alpha=.2, supervised=False)
    plot_stratpd(X, y, 'latitude', 'price', ax=axes[2, 1], alpha=.2, supervised=True)

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
    cats = np.unique(df['state'])

    figsize=(2.5,2.5)
    """
    The scale diff between states, obscures the sinusoidal nature of the
    dayofyear vs temp plot. With noise N(0,5) gotta zoom in -3,3 on mine too.
    otherwise, smooth quasilinear plot with lots of bristles showing volatility.
    Flip to N(-5,5) which is more realistic and we see sinusoid for both, even at
    scale. yep, the N(0,5) was obscuring sine for both. 
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_stratpd(X, y, 'dayofyear', 'temperature', ax=ax,
                 yrange=(-15,15),
                 pdp_marker_size=2, alpha=.5)

    savefig(f"dayofyear_vs_temp_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'state', 'temperature', cats=cats,
                    alpha=.3,
                    style='strip',
                    ax=ax,yrange=(-2,60))

    ax.set_title("StratPD")
    savefig(f"state_vs_temp_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'dayofyear', 'temperature')
    plot_ice(ice, 'dayofyear', 'temperature', ax=ax, yrange=(-15, 15))

    savefig(f"dayofyear_vs_temp_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_catice(rf, X, 'state', 'temperature')
    plot_catice(ice, 'state', 'temperature', cats=catencoders['state'], ax=ax, pdp_marker_size=10,
                yrange=(-2,60))
    ax.set_title("PD/ICE")

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
    ax.set_title("State/day vs temp")

    savefig(f"dayofyear_vs_temp")
    plt.close()


def meta_weather():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    # np.random.seed(66)

    min_r2_hires = 1.0
    # min_samples_leaf_piecewise = .2

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

    fig, axes = plt.subplots(4+1, 7, figsize=(22, (4+1)*2.8), sharey=False)
    row = 0
    for min_samples_leaf_piecewise in [.01, .1, .2, .3, .5]:
        col = 0
        for s in [2, 5, 10, 30, 50, 60, 100]:
            print(f"------------------- SIZE {s} --------------------")
            # if col>1: axes[row, col].get_yaxis().set_visible(False)
            plot_stratpd(X, y, 'dayofyear', 'temp', ax=axes[row, col],
                         min_samples_leaf_partition=s,
                         min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                         # show_importance=True,
                         yrange=(-10, 10),
                         pdp_marker_size=2, alpha=.4,
                         show_ylabel=False)

            axes[row,col].set_title(f"leaf sz {s}, hires leaf sz {min_samples_leaf_piecewise}",
                                    fontsize=9)
            col += 1
        row += 1

    savefig(f"meta_weather_r2_hires_{min_samples_leaf_piecewise}")
    plt.tight_layout()
    # plt.show()
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
    plot_stratpd(X, y, 'education', 'weight', ax=ax,
                 min_samples_leaf_partition=2,
                 # min_samples_leaf_piecewise=.02,
                 yrange=(-12, 0), alpha=.1, nlines=700, show_ylabel=False)
#    ax.get_yaxis().set_visible(False)
    savefig(f"education_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_stratpd(X, y, 'height', 'weight', ax=ax,
                 yrange=(0, 160), alpha=.1, nlines=700, show_ylabel=False)
#    ax.get_yaxis().set_visible(False)
    savefig(f"height_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'sex', 'weight', ax=ax,
                    alpha=.2,
                    cats=df_raw['sex'].unique(),
                    yrange=(0, 5),
                    )
    savefig(f"sex_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=ax,
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(-5,35),
                    )
    savefig(f"pregnant_vs_weight_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'education', 'weight')
    plot_ice(ice, 'education', 'weight', ax=ax, yrange=(-12, 0))
    savefig(f"education_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'height', 'weight')
    plot_ice(ice, 'height', 'weight', ax=ax, yrange=(0, 160))
    savefig(f"height_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_catice(rf, X, 'sex', 'weight')
    plot_catice(ice, 'sex', 'weight', cats=df_raw['sex'].unique(), ax=ax, yrange=(0, 5), pdp_marker_size=15)
    savefig(f"sex_vs_weight_pdp")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_catice(rf, X, 'pregnant', 'weight', cats=df_raw['pregnant'].unique())
    plot_catice(ice, 'pregnant', 'weight', cats=df_raw['pregnant'].unique(), ax=ax, yrange=(-5, 35), pdp_marker_size=15)
    savefig(f"pregnant_vs_weight_pdp")


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
                 min_samples_leaf_piecewise=.1,
                 yrange=(-12, 0), alpha=.1, nlines=700, supervised=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 1],
                 min_samples_leaf_piecewise=.1,
                 yrange=(-12, 0), alpha=.1, nlines=700, supervised=True)

    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 0],
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(-5,35), supervised=False)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 1],
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(-5,35), supervised=True)

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)

    
    savefig(f"weight_unsup")
    plt.close()


def weight_ntrees():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(2000)
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
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 0],
                 min_samples_leaf_piecewise=.1,
                 yrange=(-12, 0), alpha=.1, pdp_marker_size=10, show_ylabel=True,
                 ntrees=1, max_features=1.0, bootstrap=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 1],
                 min_samples_leaf_piecewise=.1,
                 yrange=(-12, 0), alpha=.1, pdp_marker_size=10, show_ylabel=False,
                 ntrees=5, max_features='auto', bootstrap=True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 2],
                 min_samples_leaf_piecewise=.1,
                 yrange=(-12, 0), alpha=.08, pdp_marker_size=10, show_ylabel=False,
                 ntrees=10, max_features = 'auto', bootstrap = True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 3],
                 min_samples_leaf_piecewise=.1,
                 yrange=(-12, 0), alpha=.05, pdp_marker_size=10, show_ylabel=False,
                 ntrees=30, max_features='auto', bootstrap=True)

    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,0], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=True,
    #              ntrees=1, max_features=1.0, bootstrap=False)
    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,1], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=False,
    #              ntrees=5, max_features='auto', bootstrap=True)
    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,2], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=False,
    #              ntrees=10, max_features = 'auto', bootstrap = True)
    # stratpd_plot(X, y, 'height', 'weight', ax=axes[1,3], yrange=(0,160), alpha=.05, pdp_dot_size=10, show_ylabel=False,
    #              ntrees=30, max_features='auto', bootstrap=True)

    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 0], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=True,
                    yrange=(0,35),
                    ntrees=1, max_features=1.0, bootstrap=False)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 1], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
                    yrange=(0,35),
                    ntrees=5, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 2], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
                    yrange=(0,35),
                    ntrees=10, max_features='auto', bootstrap=True)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=axes[1, 3], alpha=.2, cats=df_raw['pregnant'].unique(), show_ylabel=False,
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


def meta_weight():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_raw = toy_weight_data(1000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    yranges=[(0,150), (-10,0)]

    for min_samples_leaf_piecewise in [.01, .1, .2, .3]:
        plot_meta(X, y, colnames=['height', 'education'], targetname='weight',
                  min_samples_leaf_piecewise=min_samples_leaf_piecewise, yranges=yranges)
        savefig(f"meta_weight_hires_{min_samples_leaf_piecewise:.2f}")
        # plt.show()

def additivity_data(n, sd=1.0):
    x1 = np.random.uniform(-1, 1, size=n)
    x2 = np.random.uniform(-1, 1, size=n)

    y = x1**2 + x2 + np.random.normal(0, sd, size=n)
    df = pd.DataFrame()
    df['x1'] = x1
    df['x2'] = x2
    df['y'] = y
    return df

def additivity():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    df = additivity_data(n=n, sd=1) # quite noisy
    X = df.drop('y', axis=1)
    y = df['y']

    fig, axes = plt.subplots(2, 2, figsize=(4,4))#, sharey=True)
    plot_stratpd(X, y, 'x1', 'y', ax=axes[0, 0],
                 yrange=(-1, 1),
                 min_samples_leaf_piecewise=.5)

    plot_stratpd(X, y, 'x2', 'y', ax=axes[1, 0],
                 min_samples_leaf_piecewise=.5)

    axes[0, 0].set_ylim(-1,1)
    axes[1, 0].set_ylim(-2,2)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = predict_ice(rf, X, 'x1', 'y', numx=20, nlines=700)
    plot_ice(ice, 'x1', 'y', ax=axes[0, 1], yrange=(-1, 1), show_ylabel=False)
    
    ice = predict_ice(rf, X, 'x2', 'y', numx=20, nlines=700)
    plot_ice(ice, 'x2', 'y', ax=axes[1, 1], yrange=(-2, 2), show_ylabel=False)

    # axes[0,1].get_yaxis().set_visible(False)
    # axes[1,1].get_yaxis().set_visible(False)

    savefig(f"additivity")
    # plt.show()
    plt.close()


def meta_additivity():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    noises = [0, .2, .8, 1.0]
    sizes = [2, 5, 10, 40]
    min_samples_leaf_piecewise = .3

    fig, axes = plt.subplots(len(noises)+1, len(sizes), figsize=(7, 8), sharey=True, sharex=True)

    row = 0
    for sd in noises:
        df = additivity_data(n=n, sd=sd)
        X = df.drop('y', axis=1)
        y = df['y']
        col = 0
        for s in sizes:
            if row==3:
                show_xlabel = True
            else:
                show_xlabel = False
            print(f"------------------- noise {sd}, SIZE {s} --------------------")
            if col>1: axes[row, col].get_yaxis().set_visible(False)
            plot_stratpd(X, y, 'x1', 'y', ax=axes[row, col],
                         min_samples_leaf_partition=s,
                         # show_importance=True,
                         min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                         yrange=(-1, 1),
                         pdp_marker_size=2, alpha=.4,
                         show_ylabel=False,
                         show_xlabel=show_xlabel)
            if col==0:
                axes[row,col].set_ylabel(f'$y, \epsilon \sim N(0,{sd:.2f})$')

            if row==0:
                axes[row,col].set_title("Min $x_{\\overline{c}}$ leaf "+f"{s}",
                                      fontsize=12)
            col += 1
        row += 1

    lastrow = len(noises)

    axes[lastrow, 0].set_ylabel(f'$y$ vs $x_c$ partition')

    # row = 0
    # for sd in noises:
    #     axes[row, 0].scatter(X['x1'], y, alpha=.12, label=None)
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

    savefig(f"meta_additivity_noise", pad=.85)


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
    fig, axes = plt.subplots(2, 2, figsize=(4, 4), sharey=True)

    # Partial deriv wrt x2 is -5 plus 10 about half the time so about 0
    # Should not expect a criss-cross like ICE since deriv of 1_x3>=0 is 0 everywhere
    # wrt to any x, even x3. x2 *is* affecting y BUT the net effect at any spot
    # is what we care about and that's 0. Just because marginal x2 vs y shows non-
    # random plot doesn't mean that x2's net effect is nonzero. We are trying to
    # strip away x1/x3's effect upon y. When we do, x2 has no effect on y.
    # Ask what is net effect at every x2? 0.
    plot_stratpd(X, y, 'x2', 'y', ax=axes[0, 0], yrange=(-4, 4),
                 min_samples_leaf_piecewise=.3,
                 pdp_marker_size=2)
    
    # Partial deriv wrt x3 of 1_x3>=0 is 0 everywhere so result must be 0
    plot_stratpd(X, y, 'x3', 'y', ax=axes[1, 0], yrange=(-4, 4),
                 min_samples_leaf_piecewise=.3,
                 pdp_marker_size=2)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")
    
    ice = predict_ice(rf, X, 'x2', 'y', numx=100)
    plot_ice(ice, 'x2', 'y', ax=axes[0, 1], yrange=(-4, 4))

    ice = predict_ice(rf, X, 'x3', 'y', numx=100)
    plot_ice(ice, 'x3', 'y', ax=axes[1, 1], yrange=(-4, 4))

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)

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

    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2))

    axes[0].scatter(df['AGE'], y, s=5, alpha=.7)
    axes[0].set_ylabel('MEDV')
    axes[0].set_xlabel('AGE')

    axes[0].set_title("Scatter")
    axes[1].set_title("Unsupervised")
    axes[2].set_title("Supervised")
    axes[3].set_title("PD/ICE")

    plot_stratpd(X, y, 'AGE', 'MEDV', ax=axes[1], yrange=(-20, 20),
                 ntrees=20,
                 bootstrap=True,
                 supervised=False, show_ylabel=False,
                 verbose=True)
    plot_stratpd(X, y, 'AGE', 'MEDV', ax=axes[2], yrange=(-20, 20),
                 ntrees=1,
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


def lm_plot(X, y, colname, targetname,ax=None):
    ax.scatter(X[colname], y, alpha=.12, label=None)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    col = X[colname]
    # y_pred_hp = r_col.predict(col.values.reshape(-1, 1))
    # ax.plot(col, y_pred_hp, ":", linewidth=1, c='red', label='y ~ horsepower')

    r = LinearRegression()
    r.fit(X[['horsepower','weight']], y)

    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = 0 if colname=='horsepower' else 1
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
    df_cars = pd.read_csv("../../notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)

    X = df_cars[['horsepower', 'weight']]
    y = df_cars['mpg']

    fig, axes = plt.subplots(2, 3, figsize=(9,4))
    lm_plot(X, y, 'horsepower', 'mpg', ax=axes[0,0])

    lm_plot(X, y, 'weight', 'mpg', ax=axes[1,0])

    plot_stratpd(X, y, 'horsepower', 'mpg', ax=axes[0, 1],
                 xrange=(45, 235), yrange=(-20, 20), show_ylabel=False)
    plot_stratpd(X, y, 'weight', 'mpg', ax=axes[1, 1],
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
    beta0 = -r.coef_[ci]*min(col) # solved for beta0 to get y-intercept
    # axes[0,1].plot(xcol, xcol * r.coef_[ci], linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    # axes[0,2].plot(xcol, xcol * r.coef_[ci], linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")

    # draw regr line for weight
    colname = 'weight'
    col = X[colname]
    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = X.columns.get_loc(colname)
    beta0 = -r.coef_[ci]*min(col) # solved for beta0 to get y-intercept
    # axes[1,1].plot(xcol, xcol * r.coef_[ci]+11, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    # axes[1,2].plot(xcol, xcol * r.coef_[ci]+13, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    axes[1, 2].set_xlim(1600,5200)
    savefig("cars")

def meta_cars():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    df_cars = pd.read_csv("../../notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)

    X = df_cars[['horsepower', 'weight']]
    y = df_cars['mpg']

    yranges = None

    for min_samples_leaf_piecewise in [.01, .1, .2, .3]:
        plot_meta(X, y, colnames=['horsepower','weight'], targetname='mpg',
                  min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                  yranges=yranges)
        savefig(f"meta_cars_hires_{min_samples_leaf_piecewise:.2f}")
            # plt.show()


def bulldozer(): # warning: takes like 5 minutes to run
    print(f"----------- {inspect.stack()[0][3]} -----------")
    # np.random.seed(42)

    def onecol(_, X, y, colname, axes, row, xrange, yrange):
        avg_per_baths = df.groupby(colname).mean()['SalePrice']
        axes[row, 0].scatter(X[colname], y, alpha=0.07, s=1)  # , label="observation")
        axes[row, 0].set_ylabel("SalePrice")  # , fontsize=12)
        axes[row, 0].set_xlabel(colname)  # , fontsize=12)
        plot_stratpd(X, y, colname, 'SalePrice', ax=axes[row,1], xrange=xrange, yrange=yrange, show_ylabel=False,
                     verbose=False, alpha=.1)
        rf = RandomForestRegressor(n_estimators=20, min_samples_leaf=1, n_jobs=-1, oob_score=True)
        rf.fit(X, y)
        print(f"{colname} PD/ICE: RF OOB R^2 {rf.oob_score_:.3f}, training R^2 {rf.score(X,y)}")
        ice = predict_ice(rf, X, colname, 'SalePrice', numx=130, nlines=500)
        plot_ice(ice, colname, 'SalePrice', alpha=.05, ax=axes[row, 2], show_ylabel=False,
                 xrange=xrange, yrange=yrange)
        axes[row, 1].set_xlabel(colname)  # , fontsize=12)
        axes[row,1].set_ylim(*yrange)

    # to get that file, download Train.csv from
    # [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)
    # There are 401,126 records with 52 columns
    # (you must be logged in to get data)
    # The raw csv is superslow to load, but feather is fast so load then save as feather

    df = pd.read_feather("../../notebooks/data/bulldozer-train.feather")
    df['MachineHours'] = df['MachineHoursCurrentMeter'] # shorten name
    basefeatures = ['ModelID', 'YearMade', 'MachineHours']

    df = df[df['YearMade'] >= 1960]
    df = df[df['MachineHours']>0]

    df = df[basefeatures+['SalePrice']].reindex()
    df = df.dropna(axis='rows') # drop any rows with nan


    # Get subsample; it's a (sorted) timeseries so get last records not random
    df = df.iloc[-10_000:]  # take only last 10,000 records

    X, y = df[basefeatures], df['SalePrice']

    print(f"Avg bulldozer price is {np.mean(y):.2f}$")
    print(f"after cleanup {len(X)} bulldozer records")

    fig, axes = plt.subplots(3,3, figsize=(7,6))

    onecol(df, X, y, 'YearMade', axes, 0, xrange=(1960,2012), yrange=(-1000,50000))
    onecol(df, X, y, 'MachineHours', axes, 1, xrange=(0,35_000), yrange=(-35_000,35_000))

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

    xlocs = np.arange(1,ncats+1)
    axes[2, 0].scatter(xlocs, sorted_ys, alpha=0.2, s=2)  # , label="observation")
    axes[2, 0].set_ylabel("SalePrice")  # , fontsize=12)
    axes[2, 0].set_xlabel('ModelID')  # , fontsize=12)
    axes[2, 0].tick_params(axis='x', which='both', bottom=False)

    plot_catstratpd(X, y, 'ModelID', 'SalePrice', cats=np.unique(df['ModelID']), ax=axes[2,1], sort='ascending',
                    # min_samples_leaf_partition=5,
                    yrange=(0,130000), show_ylabel=False,
                    alpha=0.1,
                    style='scatter',
                    marker_size=3,
                    show_xticks=False,
                    verbose=False)

    rf = RandomForestRegressor(n_estimators=20, min_samples_leaf=1, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"ModelID PD/ICE: RF OOB R^2 {rf.oob_score_:.3f}, training R^2 {rf.score(X,y)}")

    # too slow to do all so get 1000
    ucats = np.unique(X['ModelID'])
    ucats = np.random.choice(ucats, size=1000, replace=False)
    ice = predict_catice(rf, X, 'ModelID', 'SalePrice', cats=ucats)
    plot_catice(ice, 'ModelID', targetname='SalePrice', cats=ucats,
                alpha=.05, ax=axes[2, 2], yrange=(0, 130000), show_ylabel=False,
                marker_size=3,
                sort='ascending',
                show_xticks=False)

    axes[0, 0].set_title("Scatter")
    axes[0, 1].set_title("StratPD")
    axes[0, 2].set_title("PD/ICE")

    savefig("bulldozer")
    # plt.tight_layout()
    # plt.show()


def multi_joint_distr():
    # np.random.seed(42)
    n = 1000
    min_samples_leaf_partition = 50
    min_samples_leaf_piecewise = .4
    df = pd.DataFrame(np.random.multivariate_normal([6, 6, 6, 6],
                                                    [
                                                        [1,  5, .7,  3],
                                                        [5, 1, 2, .5],
                                                        [.7, 2,  1,  1.5],
                                                        [3, .5, 1.5, 1]
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

    fig, axes = plt.subplots(6, 4, figsize=(7.5,8), sharey=False)#, sharex=True)

    axes[0,0].scatter(X['x1'], y, s=5, alpha=.08)
    axes[0, 0].set_xlim(0,13)
    axes[0, 0].set_ylim(0,45)
    axes[0,1].scatter(X['x2'], y, s=5, alpha=.08)
    axes[0, 1].set_xlim(0,13)
    axes[0, 1].set_ylim(3,45)
    axes[0, 2].scatter(X['x3'], y, s=5, alpha=.08)
    axes[0, 2].set_xlim(0,13)
    axes[0, 2].set_ylim(3,45)
    axes[0, 3].scatter(X['x4'], y, s=5, alpha=.08)
    axes[0, 3].set_xlim(0,13)
    axes[0, 3].set_ylim(3,45)

    # axes[0, 0].set_xlabel("x1")
    # axes[0, 1].set_xlabel("x2")
    # axes[0, 2].set_xlabel("x3")
    axes[0, 0].set_ylabel("y")

    for i in range(6):
        for j in range(1,4):
            axes[i,j].get_yaxis().set_visible(False)

    for i in range(6):
        for j in range(4):
            axes[i, j].set_xlim(0, 15)

    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x1', 'y', ax=axes[1,0], xrange=(0,13),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf_partition,
                     min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                     yrange=yrange, show_xlabel=False, show_ylabel=True)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,0].text(1,10,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x2', 'y', ax=axes[1,1], xrange=(0,13),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf_partition,
                     min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,1].text(1,10,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x3', 'y', ax=axes[1,2], xrange=(0,13),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf_partition,
                     min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,2].text(1,10,f"Slope={r.coef_[0]:.2f}")

    uniqx, pdp, r2_at_x = \
        plot_stratpd(X, y, 'x4', 'y', ax=axes[1,3], xrange=(0,13),
                     # show_dx_line=True,
                     min_samples_leaf_partition=min_samples_leaf_partition,
                     min_samples_leaf_piecewise=min_samples_leaf_piecewise,
                     yrange=yrange, show_xlabel=False, show_ylabel=False)
    r = LinearRegression()
    r.fit(uniqx.reshape(-1, 1), pdp)
    axes[1,3].text(1,10,f"Slope={r.coef_[0]:.2f}")

    axes[1,0].text(1,13,'StratPD', horizontalalignment='left')
    axes[1,1].text(1,13,'StratPD', horizontalalignment='left')
    axes[1,2].text(1,13,'StratPD', horizontalalignment='left')
    axes[1,3].text(1,13,'StratPD', horizontalalignment='left')

    nfeatures = 4
    regrs = [
        RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True),
        svm.SVR(gamma=1 / nfeatures),#gamma='scale'),
        LinearRegression(),
        KNeighborsRegressor(n_neighbors=5)]
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
        if rname=='KNeighborsRegressor':
            rname = 'kNN'

        show_xlabel = True if row==5 else False

        axes[row,0].text(3, 10, rname, horizontalalignment='left')
        axes[row,1].text(3, 10, rname, horizontalalignment='left')
        axes[row,2].text(3, 10, rname, horizontalalignment='left')
        axes[row,3].text(3, 10, rname, horizontalalignment='left')
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
    # multi_joint_distr()
    # rent_extra_cols()
    # bulldozer()
    # cars()
    # meta_cars()
    # unsup_boston()
    # rent()
    rent_int()
    # rent_ntrees()
    # meta_boston()
    # meta_weight()
    # unsup_rent()
    # weight()
    # unsup_weight()
    # meta_weather()
    # weight_ntrees()
    # weather()
    # meta_additivity()
    # additivity()
    # bigX()
