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


def savefig(filename):
    plt.savefig(f"images/{filename}.pdf")
    plt.savefig(f"images/{filename}.png")


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
    df = df[df.bathrooms <= 6]  # There's almost no data for above with small sample
    df = df[(df.longitude != 0) | (df.latitude != 0)]
    df = df[(df['latitude'] > 40.55) & (df['latitude'] < 40.94) &
            (df['longitude'] > -74.1) & (df['longitude'] < -73.67)]
    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price']]

    return df_rent


def rent():
    df_rent = load_rent()
    df_rent = df_rent.sample(n=8000)  # get a small subsample
    X = df_rent.drop('price', axis=1)
    y = df_rent['price']

    def showcol(colname):
        fig, ax = plt.subplots(1,1, figsize=(3,3))
        avg_per_baths = df_rent.groupby(colname).mean()['price']
        ax.scatter(df_rent[colname], df_rent['price'], alpha=0.07, s=6)#, label="observation")
        ax.scatter(np.unique(df_rent[colname]), avg_per_baths, c='black', label="average price/{colname}")
        ax.set_xlabel(colname)#, fontsize=12)
        ax.set_ylabel("Rent price")#, fontsize=12)
        ax.set_ylim(0,10_000)
        # ax.legend()
        plt.tight_layout()
        savefig(f"{colname}_vs_price")
        plt.close()

        fig, ax = plt.subplots(1,1, figsize=(3,3))
        stratpd_plot(X, y, colname, 'price', ax=ax, alpha=.2, nlines=700)
        ax.set_ylim(-1000,5000)
        plt.tight_layout()
        savefig(f"{colname}_vs_price_stratpd")
        plt.close()

        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
        rf.fit(X, y)

        fig, ax = plt.subplots(1,1, figsize=(3,3))
        ax.set_ylim(-1000,5000)
        ice = ice_predict(rf, X, colname, 'price', nlines=1000)
        ice_plot(ice, colname, 'price', alpha=.05, ax=ax)
        plt.tight_layout()
        savefig(f"{colname}_vs_price_pdp")
        plt.close()

        lm = Lasso()
        lm.fit(X, y)

        fig, ax = plt.subplots(1,1, figsize=(3,3))
        ice = ice_predict(lm, X, colname, 'price', nlines=1000)
        ice_plot(ice, colname, 'price', alpha=.05, ax=ax)
        ax.set_ylim(-1000,5000)
        plt.tight_layout()
        savefig(f"{colname}_vs_price_pdp_lm")
        plt.close()

    # showcol('bedrooms')
    showcol('bathrooms')


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

    """
    The scale diff between states, obscures the sinusoidal nature of the
    dayofyear vs temp plot. With noise N(0,5) gotta zoom in -3,3 on mine too.
    otherwise, smooth quasilinear plot with lots of bristles showing volatility.
    Flip to N(-5,5) which is more realistic and we see sinusoid for both, even at
    scale. yep, the N(0,5) was obscuring sine for both. 
    """
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'dayofyear', 'temperature', ax=ax,
              ntrees=30, min_samples_leaf=2, yrange=(-20,20),
                 pdp_dot_size=2, alpha=.1, nlines=900)
    plt.tight_layout()
    savefig(f"dayofyear_vs_temp_stratpd")
    plt.close()

    # catstratpd_plot(X, y, 'state', 'temperature', cats=catencoders['state'],
    #              ax=axes[2][0])  # , yrange=(0,160))

    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'dayofyear', 'temperature')
    ice_plot(ice, 'dayofyear', 'temperature', ax=ax, yrange=(-20,20))
    plt.tight_layout()
    savefig(f"dayofyear_vs_temp_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'state', 'temperature')
    ice_plot(ice, 'state', 'temperature', cats=catencoders['state'],
             ax=ax)
    plt.tight_layout()
    savefig(f"state_vs_temp_pdp")
    plt.close()

    df = df_raw.copy()
    # axes[3, 0].plot(df.loc[df['state'] == 'CA', 'dayofyear'],
    #                 df.loc[df['state'] == 'CA', 'temperature'], label="CA")
    # axes[3, 0].plot(df.loc[df['state'] == 'CO', 'dayofyear'],
    #                 df.loc[df['state'] == 'CO', 'temperature'], label="CO")
    # axes[3, 0].plot(df.loc[df['state'] == 'AZ', 'dayofyear'],
    #                 df.loc[df['state'] == 'AZ', 'temperature'], label="AZ")
    # axes[3, 0].plot(df.loc[df['state'] == 'WA', 'dayofyear'],
    #                 df.loc[df['state'] == 'WA', 'temperature'], label="WA")
    # axes[3, 0].legend()
    # axes[3, 0].set_title('Raw data')
    # axes[3, 0].set_ylabel('Temperature')
    # axes[3, 0].set_xlabel('Dataframe row index')

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    rtreeviz_univar(ax,
                    X['state'], y,
                    feature_name='state',
                    target_name='y',
                    min_samples_leaf=2,
                    fontsize=10, show={'splits'})
    ax.set_xlabel("state")
    ax.set_ylabel("y")
    plt.tight_layout()
    savefig(f"state_vs_temp_partition_pdp")
    plt.close()


def weight():
    df_raw = toy_weight_data(2000)
    df = df_raw.copy()
    catencoders = df_string_to_cat(df)
    df_cat_to_catcode(df)
    df['pregnant'] = df['pregnant'].astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'education', 'weight', ax=ax, yrange=(-12, 0), alpha=.1, nlines=700)
    plt.tight_layout()
    savefig(f"education_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'height', 'weight', ax=ax, yrange=(0, 160), alpha=.1, nlines=700)
    plt.tight_layout()
    savefig(f"height_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    catstratpd_plot(X, y, 'sex', 'weight', ax=ax, ntrees=30,
                    alpha=.2,
                    cats=df_raw['sex'].unique(),
                    yrange=(0, 5)
                    )
    plt.tight_layout()
    savefig(f"sex_vs_weight_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    catstratpd_plot(X, y, 'pregnant', 'weight', ax=ax, ntrees=30,
                    alpha=.2,
                    cats=df_raw['pregnant'].unique(),
                    yrange=(0, 30)
                    )
    plt.tight_layout()
    savefig(f"pregnant_vs_weight_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'education', 'weight')
    ice_plot(ice, 'education', 'weight', ax=ax, yrange=(-12, 0))
    plt.tight_layout()
    savefig(f"education_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'height', 'weight')
    ice_plot(ice, 'height', 'weight', ax=ax, yrange=(0, 160))
    plt.tight_layout()
    savefig(f"height_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'sex', 'weight')
    ice_plot(ice, 'sex', 'weight', ax=ax, yrange=(0, 5),
             cats=df_raw['sex'].unique())
    plt.tight_layout()
    savefig(f"sex_vs_weight_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'pregnant', 'weight')
    ice_plot(ice, 'pregnant', 'weight', ax=ax, yrange=(0, 30),
             cats=df_raw['pregnant'].unique())
    plt.tight_layout()
    savefig(f"pregnant_vs_weight_pdp")
    plt.close()
    #
    # fig.suptitle("weight = 120 + 10*(height-min(height)) + 10*pregnant - 1.2*education",
    #              size=14)
    #
    # plt.tight_layout()


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
    n = 1000
    df = additivity_data(n=n)
    X = df.drop('y', axis=1)
    y = df['y']

    min_samples_leaf = 11

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'x1', 'y', ax=ax, min_samples_leaf=min_samples_leaf,
                 ntrees=50,
              hires_threshold=10, yrange=(-1, 1), pdp_dot_size=3, alpha=.1, nlines=700)
    plt.tight_layout()
    savefig(f"add_x1_y_stratpd")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'x2', 'y', ax=ax, min_samples_leaf=min_samples_leaf,
                 ntrees=50,
                 hires_threshold=10, pdp_dot_size=3, alpha=.1, nlines=700)
    plt.tight_layout()
    savefig(f"add_x2_y_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'x1', 'y', numx=20, nlines=700)
    ice_plot(ice, 'x1', 'y', ax=ax, yrange=(-1, 1))
    plt.tight_layout()
    savefig(f"add_x1_y_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'x2', 'y', numx=20, nlines=700)
    ice_plot(ice, 'x2', 'y', ax=ax, yrange=(-2, 2))
    plt.tight_layout()
    savefig(f"add_x2_y_pdp")
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
    n = 1000
    df = bigX_data(n=n)
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Partial deriv is just 0.2 so this is correct. flat deriv curve, net effect line at slope .2
    # ICE is way too shallow and not line at n=1000 even
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'x1', 'y', ax=ax, yrange=(-.1,.5), alpha=.1, nlines=700, pdp_dot_size=2)
    plt.tight_layout()
    savefig(f"bigx_x1_y_stratpd")
    plt.close()

    # Partial deriv wrt x2 is -5 plus 10 about half the time so about 0
    # Should not expect a criss-cross like ICE since deriv of 1_x3>=0 is 0 everywhere
    # wrt to any x, even x3. x2 *is* affecting y BUT the net effect at any spot
    # is what we care about and that's 0. Just because marginal x2 vs y shows non-
    # random plot doesn't mean that x2's net effect is nonzero. We are trying to
    # strip away x1/x3's effect upon y. When we do, x2 has no effect on y.
    # Key is asking right question. Don't look at marginal plot and say obvious.
    # Ask what is net effect at every x2? 0.
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'x2', 'y', ax=ax, yrange=(-4,4), alpha=.1, nlines=700, pdp_dot_size=2)
    plt.tight_layout()
    savefig(f"bigx_x2_y_stratpd")
    plt.close()

    # Partial deriv wrt x3 of 1_x3>=0 is 0 everywhere so result must be 0
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    stratpd_plot(X, y, 'x3', 'y', ax=ax, yrange=(-4,4), alpha=.1, nlines=700, pdp_dot_size=2)
    plt.tight_layout()
    savefig(f"bigx_x3_y_stratpd")
    plt.close()

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, oob_score=True)
    rf.fit(X, y)
    print(f"RF OOB {rf.oob_score_}")
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'x1', 'y', numx=10)
    ice_plot(ice, 'x1', 'y', ax=ax, yrange=(-.1,.5))
    plt.tight_layout()
    savefig(f"bigx_x1_y_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'x2', 'y', numx=10)
    ice_plot(ice, 'x2', 'y', ax=ax)
    plt.tight_layout()
    savefig(f"bigx_x2_y_pdp")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ice = ice_predict(rf, X, 'x3', 'y', numx=10)
    ice_plot(ice, 'x3', 'y', ax=ax)
    plt.tight_layout()
    savefig(f"bigx_x3_y_pdp")
    plt.close()

if __name__ == '__main__':
    rent()
    weight()
    weather()
    additivity()
    bigX()