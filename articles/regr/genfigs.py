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
from stratx.partdep import *
from stratx.ice import *
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


def fix_missing_num(df, colname):
    df[colname+'_na'] = pd.isnull(df[colname])
    df[colname].fillna(df[colname].median(), inplace=True)

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

    ice = predict_ice(rf, X, colname, 'price', nlines=1000)
    plot_ice(ice, colname, 'price', alpha=.05, ax=axes[0, 1], show_xlabel=False)
    axes[0,1].set_ylim(-1000,5000)

    lm = Lasso()
    lm.fit(X, y)

    ice = predict_ice(lm, X, colname, 'price', nlines=1000)
    plot_ice(ice, colname, 'price', alpha=.05, ax=axes[1, 0], show_ylabel=False)
    axes[1,0].set_ylim(-1000,5000)

    plot_stratpd(X, y, colname, 'price', ax=axes[1, 1], alpha=.2)
    axes[1,1].set_ylim(-1000,5000)

    #axes[1,1].get_yaxis().set_visible(False)

    savefig(f"{colname}_vs_price")
    plt.close()


def rent_ntrees():
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
            plot_stratpd(X, y, colname, 'price', ax=axes[row, i], alpha=alpha,
                         yrange=yrange,
                         min_r2_hires=0.5,
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


def meta_rent():
    df_rent = load_rent()

    # df_rent = df_rent.sample(n=8000, random_state=111)  # get a small subsample

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    yranges = None

    min_r2_hires = 0.0
    min_samples_leaf_hires = .1
    plot_meta(X, y, colnames=['bedrooms','bathrooms','latitude','longitude'], targetname='rent',
              min_r2_hires=min_r2_hires,
              min_samples_leaf_hires=min_samples_leaf_hires)
    savefig(f"meta_rent_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")
    # plt.show()

    min_r2_hires = 0.0
    min_samples_leaf_hires = .2
    plot_meta(X, y, colnames=['bedrooms','bathrooms','latitude','longitude'], targetname='rent',
              min_r2_hires=min_r2_hires,
              min_samples_leaf_hires=min_samples_leaf_hires)
    savefig(f"meta_rent_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")
    # plt.show()

    min_r2_hires = .1
    min_samples_leaf_hires = .1
    plot_meta(X, y, colnames=['bedrooms','bathrooms','latitude','longitude'], targetname='rent',
              min_r2_hires=min_r2_hires,
              min_samples_leaf_hires=min_samples_leaf_hires, yranges=yranges)
    savefig(f"meta_rent_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")

    min_r2_hires = .1
    min_samples_leaf_hires = .2
    plot_meta(X, y, colnames=['bedrooms','bathrooms','latitude','longitude'], targetname='rent',
              min_r2_hires=min_r2_hires,
              min_samples_leaf_hires=min_samples_leaf_hires, yranges=yranges)
    savefig(f"meta_rent_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")

    min_r2_hires = .1
    min_samples_leaf_hires = .3
    plot_meta(X, y, colnames=['bedrooms','bathrooms','latitude','longitude'], targetname='rent',
              min_r2_hires=min_r2_hires,
              min_samples_leaf_hires=min_samples_leaf_hires, yranges=yranges)
    savefig(f"meta_rent_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")

    min_r2_hires = .5
    min_samples_leaf_hires = .1
    plot_meta(X, y, colnames=['bedrooms','bathrooms','latitude','longitude'], targetname='rent',
              min_r2_hires=min_r2_hires,
              min_samples_leaf_hires=min_samples_leaf_hires, yranges=yranges)
    savefig(f"meta_rent_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")

    # plt.show()


def meta_boston():
    boston = load_boston()
    print(len(boston.data))
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    yranges = [(-30,0), (0,30), (-8,8), (-11, 0)]

    for min_r2_hires in [.01,.1,.2,.5, 1.0]:
        for min_samples_leaf_hires in [.01, .1, .2, .3]:
            plot_meta(X, y, colnames=['LSTAT', 'RM', 'CRIM', 'DIS'], targetname='MEDV',
                      min_r2_hires=min_r2_hires,
                      min_samples_leaf_hires=min_samples_leaf_hires,
                      yranges=yranges)
            savefig(f"meta_boston_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")


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


def plot_meta(X, y, colnames, targetname, min_r2_hires, min_samples_leaf_hires, yranges=None):
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
            print(f"---------- min_samples_leaf={meta}, min R^2={min_r2_hires:.2f}, hires leafsz={min_samples_leaf_hires:.2f} ----------- ")
            plot_stratpd(X, y, colname, targetname, ax=axes[row, col],
                         min_r2_hires=min_r2_hires,
                         min_samples_leaf_hires=min_samples_leaf_hires,
                         min_samples_leaf=meta,
                         yrange=yranges[i],
                         ntrees=1)
            axes[row, col].set_title(f"leafsz={meta}, min R^2={min_r2_hires:.2f}\nhires leafsz={min_samples_leaf_hires:.2f}",
                                     fontsize=9)
            col += 1
        row += 1

    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
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
                 min_samples_leaf_hires=15,
                 yrange=(-15,15),
                 pdp_dot_size=2, alpha=.5)

    savefig(f"dayofyear_vs_temp_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'state', 'temperature', cats=catencoders['state'],
                    sort=None,
                    alpha=.3,
                    min_samples_leaf=11,
                    ax=ax)  # , yrange=(0,160))

    savefig(f"state_vs_temp_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'dayofyear', 'temperature')
    plot_ice(ice, 'dayofyear', 'temperature', ax=ax, yrange=(-15, 15))

    savefig(f"dayofyear_vs_temp_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'state', 'temperature')
    plot_ice(ice, 'state', 'temperature', cats=catencoders['state'], ax=ax)

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


def meta_weather():
    # np.random.seed(66)

    min_r2_hires = 1.0
    # min_samples_leaf_hires = .2

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
    for min_samples_leaf_hires in [.01, .1, .2, .3, .5]:
        col = 0
        for s in [2, 5, 10, 30, 50, 60, 100]:
            print(f"------------------- SIZE {s} --------------------")
            # if col>1: axes[row, col].get_yaxis().set_visible(False)
            plot_stratpd(X, y, 'dayofyear', 'temp', ax=axes[row, col],
                         min_samples_leaf=s,
                         min_samples_leaf_hires=min_samples_leaf_hires,
                         min_r2_hires=min_r2_hires,
                         min_samples_hires=10,
                         # show_importance=True,
                         yrange=(-10, 10),
                         pdp_dot_size=2, alpha=.4,
                         show_ylabel=False)

            axes[row,col].set_title(f"leaf sz {s}, hires {min_r2_hires}\nhires leaf sz {min_samples_leaf_hires}",
                                    fontsize=9)
            col += 1
        row += 1

    savefig(f"meta_weather_r2_{min_r2_hires}_hires_{min_samples_leaf_hires}")
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
    plot_stratpd(X, y, 'education', 'weight', ax=ax, yrange=(-12, 0), alpha=.1, nlines=700, show_ylabel=False)
#    ax.get_yaxis().set_visible(False)
    savefig(f"education_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_stratpd(X, y, 'height', 'weight', ax=ax, yrange=(0, 160), alpha=.1, nlines=700, show_ylabel=False)
#    ax.get_yaxis().set_visible(False)
    savefig(f"height_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'sex', 'weight', ax=ax,
                    alpha=.2,
                    cats=df_raw['sex'].unique(),
                    yrange=(0, 5)
                    )
    savefig(f"sex_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_catstratpd(X, y, 'pregnant', 'weight', ax=ax,
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(-5,35)
                    )
    savefig(f"pregnant_vs_weight_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'education', 'weight')
    plot_ice(ice, 'education', 'weight', ax=ax, yrange=(-12, 0))
    savefig(f"education_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'height', 'weight')
    plot_ice(ice, 'height', 'weight', ax=ax, yrange=(0, 160))
    savefig(f"height_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'sex', 'weight')
    plot_ice(ice, 'sex', 'weight', ax=ax, yrange=(0, 5),
             cats=df_raw['sex'].unique())
    savefig(f"sex_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ice = predict_ice(rf, X, 'pregnant', 'weight')
    plot_ice(ice, 'pregnant', 'weight', ax=ax, yrange=(-5, 35),
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
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 0], yrange=(-12, 0), alpha=.1, nlines=700, supervised=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 1], yrange=(-12, 0), alpha=.1, nlines=700, supervised=True)

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
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 0], yrange=(-12, 0), alpha=.05, pdp_dot_size=10, show_ylabel=True,
                 ntrees=1, max_features=1.0, bootstrap=False)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 1], yrange=(-12, 0), alpha=.05, pdp_dot_size=10, show_ylabel=False,
                 ntrees=5, max_features='auto', bootstrap=True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 2], yrange=(-12, 0), alpha=.05, pdp_dot_size=10, show_ylabel=False,
                 ntrees=10, max_features = 'auto', bootstrap = True)
    plot_stratpd(X, y, 'education', 'weight', ax=axes[0, 3], yrange=(-12, 0), alpha=.05, pdp_dot_size=10, show_ylabel=False,
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

    for min_r2_hires in [.1]:#[.01, .1, .2, .5]:
        for min_samples_leaf_hires in [.01, .1, .2, .3]:
            plot_meta(X, y, colnames=['height', 'education'], targetname='weight',
                      min_r2_hires=min_r2_hires,
                      min_samples_leaf_hires=min_samples_leaf_hires, yranges=yranges)
            savefig(f"meta_weight_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")
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
    df = additivity_data(n=n, sd=.8) # quite noisy
    X = df.drop('y', axis=1)
    y = df['y']

    fig, axes = plt.subplots(2, 2, figsize=(4,4), sharey=True)
    plot_stratpd(X, y, 'x1', 'y', ax=axes[0, 0],
                 yrange=(-1, 1),
                 min_samples_leaf=30,
                 min_samples_leaf_hires=.3,
                 pdp_dot_size=3, alpha=.5)
    
    plot_stratpd(X, y, 'x2', 'y', ax=axes[1, 0],
                 min_samples_leaf=30,
                 min_samples_leaf_hires=.3,
                 pdp_dot_size=3, alpha=.1, nlines=700)
    
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    ice = predict_ice(rf, X, 'x1', 'y', numx=20, nlines=700)
    plot_ice(ice, 'x1', 'y', ax=axes[0, 1], yrange=(-1, 1), show_ylabel=False)
    
    ice = predict_ice(rf, X, 'x2', 'y', numx=20, nlines=700)
    plot_ice(ice, 'x2', 'y', ax=axes[1, 1], yrange=(-2, 2), show_ylabel=False)

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)

    savefig(f"additivity")
    # plt.show()
    plt.close()


def meta_additivity():
    print(f"----------- {inspect.stack()[0][3]} -----------")
    n = 1000
    noises = [0, .2, .5, .8, 1.0]
    min_r2_hires = 1.0
    min_samples_leaf_hires = .2

    fig, axes = plt.subplots(len(noises)+1, 7, figsize=(22, (len(noises)+1)*2.8), sharey=False)

    row = 0
    for sd in noises:
        df = additivity_data(n=n, sd=sd)
        X = df.drop('y', axis=1)
        y = df['y']
        col = 1
        for s in [2, 5, 10, 30, 50, 60]:
            print(f"------------------- noise {sd}, SIZE {s} --------------------")
            if col>1: axes[row, col].get_yaxis().set_visible(False)
            plot_stratpd(X, y, 'x1', 'y', ax=axes[row, col],
                         min_samples_leaf=s,
                         min_samples_leaf_hires=min_samples_leaf_hires,
                         min_r2_hires=min_r2_hires,
                         min_samples_hires=10,
                         # show_importance=True,
                         yrange=(-1, 1),
                         pdp_dot_size=2, alpha=.4,
                         show_ylabel=False)

            axes[row,col].set_title(f"leaf sz {s}, hires {min_r2_hires}\nhires leaf sz {min_samples_leaf_hires}",
                                  fontsize=9)
            col += 1
        row += 1

    lastrow = len(noises)

    row = 0
    for sd in noises:
        axes[row, 0].scatter(X['x1'], y, alpha=.12, label=None)
        axes[row, 0].set_xlabel("x1")
        axes[row, 0].set_ylabel("y")
        axes[row, 0].set_ylim(-5, 5)
        axes[row, 0].set_title(f"$y = x_1^2 + x_2 + \epsilon$, $\epsilon \sim N(0,{sd:.2f})$")
        row += 1

    col = 1
    for s in [2, 5, 10, 30, 50, 60]:
        rtreeviz_univar(axes[lastrow, col],
                        X['x2'], y,
                        min_samples_leaf=s,
                        feature_name='x2',
                        target_name='y',
                        fontsize=10, show={'splits'})
        axes[lastrow, col].set_xlabel("x1")
        col += 1

    savefig(f"meta_additivity_r2_{min_r2_hires}_hires_{min_samples_leaf_hires}")
    plt.tight_layout()
    # plt.show()
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
    plot_stratpd(X, y, 'x1', 'y', ax=axes[0, 0], yrange=(-4, 4), alpha=.1, nlines=700, pdp_dot_size=2)
    
    # Partial deriv wrt x2 is -5 plus 10 about half the time so about 0
    # Should not expect a criss-cross like ICE since deriv of 1_x3>=0 is 0 everywhere
    # wrt to any x, even x3. x2 *is* affecting y BUT the net effect at any spot
    # is what we care about and that's 0. Just because marginal x2 vs y shows non-
    # random plot doesn't mean that x2's net effect is nonzero. We are trying to
    # strip away x1/x3's effect upon y. When we do, x2 has no effect on y.
    # Key is asking right question. Don't look at marginal plot and say obvious.
    # Ask what is net effect at every x2? 0.
    plot_stratpd(X, y, 'x2', 'y', ax=axes[1, 0], yrange=(-4, 4), alpha=.1, nlines=700, pdp_dot_size=2)
    
    # Partial deriv wrt x3 of 1_x3>=0 is 0 everywhere so result must be 0
    plot_stratpd(X, y, 'x3', 'y', ax=axes[2, 0], yrange=(-4, 4), alpha=.1, nlines=700, pdp_dot_size=2)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")
    
    ice = predict_ice(rf, X, 'x1', 'y', numx=10)
    plot_ice(ice, 'x1', 'y', ax=axes[0, 1], yrange=(-4, 4))
    
    ice = predict_ice(rf, X, 'x2', 'y', numx=10)
    plot_ice(ice, 'x2', 'y', ax=axes[1, 1], yrange=(-4, 4))

    ice = predict_ice(rf, X, 'x3', 'y', numx=10)
    plot_ice(ice, 'x3', 'y', ax=axes[2, 1], yrange=(-4, 4))

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

    plot_stratpd(X, y, 'AGE', 'MEDV', ax=axes[0, 0], yrange=(-20, 20), supervised=False, show_xlabel=False)
    plot_stratpd(X, y, 'AGE', 'MEDV', ax=axes[0, 1], yrange=(-20, 20), supervised=True, show_xlabel=False)

    rf = RandomForestRegressor(n_estimators=100, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    axes[1,0].scatter(df['AGE'], y, s=5, alpha=.7)
    axes[1,0].set_ylabel('MEDV')
    axes[1,0].set_xlabel('AGE')

    ice = predict_ice(rf, X, 'AGE', 'MEDV', numx=10)
    plot_ice(ice, 'AGE', 'MEDV', ax=axes[1, 1], yrange=(-20, 20))

    axes[0,1].get_yaxis().set_visible(False)
    axes[1,1].get_yaxis().set_visible(False)

    
    savefig(f"boston_unsup")
    plt.close()


def lm_plot(X, y, colname, targetname,ax=None):
    r_col = LinearRegression()
    r_col.fit(X[[colname]], y)
    ax.scatter(X[colname], y, alpha=.12, label=None)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    col = X[colname]
    # y_pred_hp = r_col.predict(col.values.reshape(-1, 1))
    # ax.plot(col, y_pred_hp, ":", linewidth=1, c='red', label='y ~ horsepower')
    r = LinearRegression()
    r.fit(X[['horsepower','weight']], y)
    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = X.columns.get_loc(colname)
    ax.plot(xcol, xcol * r.coef_[ci] + r_col.intercept_, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    left40 = xcol[int(len(xcol) * .4)]
    ax.text(min(xcol), max(y)*.94, f"$\hat{{y}} = \\beta_0 + \\beta_1 x_{{horsepower}} + \\beta_2 x_{{weight}}$")
    i = 1 if colname=='horsepower' else 2
    ax.text(left40, left40*r.coef_[ci] + r_col.intercept_, f"$\\beta_{i}$={r.coef_[ci]:.3f}")

def cars():
    df_cars = pd.read_csv("../../notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)

    X = df_cars[['horsepower', 'weight']]
    y = df_cars['mpg']

    fig, axes = plt.subplots(2, 3, figsize=(9,4))
    lm_plot(X, y, 'horsepower', 'mpg', ax=axes[0,0])

    lm_plot(X, y, 'weight', 'mpg', ax=axes[1,0])

    plot_stratpd(X, y, 'horsepower', 'mpg', ax=axes[0, 2], xrange=(45, 235), yrange=(-20, 20), show_ylabel=False)
    plot_stratpd(X, y, 'weight', 'mpg', ax=axes[1, 2], xrange=(1600, 5200), yrange=(-20, 20), show_ylabel=False)

    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    ice = predict_ice(rf, X, 'horsepower', 'mpg', numx=100)
    plot_ice(ice, 'horsepower', 'mpg', ax=axes[0, 1], yrange=(-20, 20), show_ylabel=False)
    ice = predict_ice(rf, X, 'weight', 'mpg', numx=100)
    plot_ice(ice, 'weight', 'mpg', ax=axes[1, 1], yrange=(-20, 20), show_ylabel=False)

    # draw regr line for horsepower
    r = LinearRegression()
    r.fit(X, y)
    colname = 'horsepower'
    col = X[colname]
    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = X.columns.get_loc(colname)
    beta0 = -r.coef_[ci]*min(col) # solved for beta0 to get y-intercept
    axes[0,1].plot(xcol, xcol * r.coef_[ci], linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    axes[0,2].plot(xcol, xcol * r.coef_[ci]+4, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")

    # draw regr line for weight
    colname = 'weight'
    col = X[colname]
    xcol = np.linspace(np.min(col), np.max(col), num=100)
    ci = X.columns.get_loc(colname)
    beta0 = -r.coef_[ci]*min(col) # solved for beta0 to get y-intercept
    axes[1,1].plot(xcol, xcol * r.coef_[ci]+11, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    axes[1,2].plot(xcol, xcol * r.coef_[ci]+13, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
    axes[1, 1].set_xlim(1600,5200)
    savefig("cars")

def meta_cars():
    df_cars = pd.read_csv("../../notebooks/data/auto-mpg.csv")
    df_cars = df_cars[df_cars['horsepower'] != '?']  # drop the few missing values
    df_cars['horsepower'] = df_cars['horsepower'].astype(float)

    X = df_cars[['horsepower', 'weight']]
    y = df_cars['mpg']

    yranges = None

    for min_r2_hires in [.01,.1,.2,.5,1.0]:
        for min_samples_leaf_hires in [.01, .1, .2, .3]:
            plot_meta(X, y, colnames=['horsepower','weight'], targetname='mpg',
                      min_r2_hires=min_r2_hires,
                      min_samples_leaf_hires=min_samples_leaf_hires,
                      yranges=yranges)
            savefig(f"meta_cars_r2_{min_r2_hires:.2f}_hires_{min_samples_leaf_hires:.2f}")
            # plt.show()


def bulldozer():
    # np.random.seed(42)

    def onecol(_, X, y, colname, axes, row, yrange):
        avg_per_baths = df.groupby(colname).mean()['SalePrice']
        axes[row, 0].scatter(X[colname], y, alpha=0.07, s=4)  # , label="observation")
        # axes[row, 0].scatter(np.unique(df[colname]), avg_per_baths, s=6, c='black',
        #                    label="average SalePrice/{colname}")
        axes[row, 0].set_ylabel("SalePrice")  # , fontsize=12)
        axes[row, 0].set_xlabel(colname)  # , fontsize=12)
        # ax.legend()
        plot_stratpd(X, y, colname, 'SalePrice', ax=axes[row,2], nlines=500, yrange=yrange, show_ylabel=False)
        rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=1, oob_score=True)
        rf.fit(X, y)
        ice = predict_ice(rf, X, colname, 'SalePrice', nlines=500)
        plot_ice(ice, colname, 'SalePrice', alpha=.05, ax=axes[row, 1], show_ylabel=False)
        axes[row, 1].set_xlabel(colname)  # , fontsize=12)
        axes[row,1].set_ylim(*yrange)

    # to get that file, download Train.csv from
    # [Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)
    # There are 401,126 records with 52 columns
    # (you must be logged in to get data)
    # The raw csv is superslow to load, but feather is fast so load then save as feather

    df = pd.read_feather("../../notebooks/data/bulldozer-train.feather")
    basefeatures = ['ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID', 'MachineHoursCurrentMeter']

    # Get subsample; it's a (sorted) timeseries so get last records not random
    df = df.iloc[-1000:]  # take only last 100,000 records

    df.loc[df.YearMade < 1950, 'YearMade'] = df['YearMade'].median()

    df['MachineHoursCurrentMeter'].fillna(df['MachineHoursCurrentMeter'].median(), inplace=True)

    df.loc[df.eval("MachineHoursCurrentMeter==0"), 'MachineHoursCurrentMeter'] = \
        df['MachineHoursCurrentMeter'].median()

    df = df.fillna(0)  # flip missing numeric values to zeros

    X, y = df[basefeatures], df['SalePrice']

    fig, axes = plt.subplots(3,3, figsize=(9,9))

    # onecol(df, X, y, 'YearMade', axes, 0, yrange=(-1000,40000))
    # onecol(df, X, y, 'MachineHoursCurrentMeter', axes, 1, yrange=(-1500,10000))

    axes[2, 0].scatter(X['ModelID'], y, alpha=0.07, s=3)  # , label="observation")
    axes[2, 0].set_ylabel("SalePrice")  # , fontsize=12)
    axes[2, 0].set_xlabel('ModelID')  # , fontsize=12)

    rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=1)
    rf.fit(X, y)

    ice = predict_catice(rf, X, 'ModelID', 'SalePrice', cats=np.unique(X['ModelID']))
    plot_catice(ice, 'ModelID', targetname='SalePrice', cats=np.unique(df['ModelID']),
                alpha=.05, ax=axes[2, 1], yrange=(-20000, 130000), show_ylabel=False, sort='ascending')

    plot_catstratpd(X, y, 'ModelID', 'SalePrice', cats=np.unique(df['ModelID']), ax=axes[2,2], sort='ascending',
                    yrange=(-20000,130000), show_ylabel=False)

    # savefig("bulldozer")
    plt.show()


if __name__ == '__main__':
    bulldozer()
    # cars()
    # meta_cars()
    # unsup_boston()
    # rent()
    # rent_ntrees()
    # meta_boston()
    # meta_weight()
    # meta_rent()
    # unsup_rent()
    # weight()
    # unsup_weight()
    # meta_weather()
    # weather()
    # meta_additivity()
    # additivity()
    # bigX()
