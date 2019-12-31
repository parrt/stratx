import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
import statsmodels.api as sm
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn import svm

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)

import rfpimp

from stratx.partdep import *
from stratx.featimp import *

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


def fix_missing_num(df, colname):
    df[colname+'_na'] = pd.isnull(df[colname]).astype(int)
    df[colname].fillna(df[colname].median(), inplace=True)


def df_split_dates(df,colname):
    df["saleyear"] = df[colname].dt.year
    df["salemonth"] = df[colname].dt.month
    df["saleday"] = df[colname].dt.day
    df["saledayofweek"] = df[colname].dt.dayofweek
    df["saledayofyear"] = df[colname].dt.dayofyear
    df[colname] = df[colname].astype(np.int64) # convert to seconds since 1970


def spearmans_importances(X, y):
    global I
    correlations = [spearmanr(X[colname], y)[0] for colname in X.columns]
    I = pd.DataFrame(data={'Feature': X.columns, 'Importance': np.abs(correlations)})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


def linear_model_importance(model, X, y):
    model.fit(X, y)
    score = model.score(X, y)

    imp = np.abs(model.coef_)

    # use statsmodels to get stderr for betas
    if isinstance(model, LinearRegression):
        # stderr for betas makes no sense in Lasso
        beta_stderr = sm.OLS(y.values, X).fit().bse
        imp /= beta_stderr

    imp /= np.sum(imp) # normalize
    I = pd.DataFrame(data={'Feature': X.columns, 'Importance': imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I, score


def shap_importances(model, X, n_shap, normalize=True, sort=True):
    start = timer()
    #X = shap.kmeans(X, k=n_shap)
    X = X.sample(n=n_shap, replace=False)
    if isinstance(model, RandomForestRegressor) or \
        isinstance(model, GradientBoostingRegressor) or \
        isinstance(model, xgb.XGBRegressor):
        explainer = shap.TreeExplainer(model, data=X, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X, check_additivity=False)
    elif isinstance(model, Lasso) or isinstance(model, LinearRegression):
        shap_values = shap.LinearExplainer(model, X, feature_dependence='independent').shap_values(X)
    else:
        explainer = shap.KernelExplainer(model.predict, X.sample(frac=.1))
        shap_values = explainer.shap_values(X, nsamples=100)
    shapimp = np.mean(np.abs(shap_values), axis=0)
    stop = timer()
    print(f"SHAP time for {len(X)} records using {model.__class__.__name__} = {(stop - start):.1f}s")

    total_imp = np.sum(shapimp)
    normalized_shap = shapimp
    if normalize:
        normalized_shap = shapimp / total_imp

    # print("SHAP", normalized_shap)
    shapI = pd.DataFrame(data={'Feature': X.columns, 'Importance': normalized_shap})
    shapI = shapI.set_index('Feature')
    if sort:
        shapI = shapI.sort_values('Importance', ascending=False)
    # plot_importances(shapI)
    return shapI


def compare_top_features(X, y, top_features_range=None,
                         n_shap=300,
                         metric = mean_absolute_error,
                         use_oob = False,
                         trials=1,
                         n_stratpd_trees=1,
                         bootstrap=False,
                         stratpd_min_samples_leaf=10,
                         min_slopes_per_x=15,
                         catcolnames=set(),
                         supervised=True):
    if use_oob and metric!=r2_score:
        #     print("Warning: use_oob can only give R^2; flipping metric to r2_score")
        metric=r2_score

    n_estimators = 40 # for both SHAP and testing purposes

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"Sanity check: R^2 OOB on {X.shape[0]} records: {rf.oob_score_:.3f}, training {metric.__name__}={metric(y, rf.predict(X))}")

    ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        get_multiple_imps(X, y,
                          n_stratpd_trees=n_stratpd_trees,
                          bootstrap=bootstrap,
                          stratpd_min_samples_leaf=stratpd_min_samples_leaf,
                          n_estimators=n_estimators,
                          n_shap=n_shap,
                          catcolnames=catcolnames,
                          min_slopes_per_x=min_slopes_per_x,
                          supervised=supervised)
    print("OLS\n", ols_I)
    print("OLS SHAP\n", shap_ols_I)
    print("RF SHAP\n", rf_I)
    print("RF perm\n", perm_I)
    print("Our importances\n",our_I)

    if top_features_range is None:
        top_features_range = (1, X.shape[1])

    features_names = ['OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact']

    print("Our FEATURES rank", our_I.index.values)

    print(f"n={len(X)}, n_top={top_features_range[1]}, n_estimators={n_estimators}, n_shap={n_shap}, min_samples_leaf={stratpd_min_samples_leaf}")
    topscores = []
    for top in range(top_features_range[0], top_features_range[1] + 1):
        ols_top = ols_I.iloc[:top, 0].index.values
        shap_ols_top = shap_ols_I.iloc[:top, 0].index.values
        rf_top = rf_I.iloc[:top, 0].index.values
        perm_top = perm_I.iloc[:top, 0].index.values
        our_top = our_I.iloc[:top, 0].index.values
        features_set = [ols_top, shap_ols_top, rf_top, perm_top, our_top]
        all = []
        for i in range(trials):
            # print(i, end=' ')
            results = []
            for name, features in zip(features_names, features_set):
                # print(f"Train with {features} from {name}")
                # Train RF model with top-k features
                rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=use_oob,
                                           min_samples_leaf=1, n_jobs=-1)
                rf.fit(X[features], y)
                if use_oob:
                    s = rf.oob_score_
                else:
                    y_pred = rf.predict(X[features])
                    s = metric(y, y_pred)

                results.append(s)
                # print(f"{name} valid R^2 {s:.3f}")
            all.append(results)
        # print(pd.DataFrame(data=all, columns=['OLS','RF','Ours']))
        # print()
        topscores.append( [round(m,3) for m in np.mean(all, axis=0)] )

        # avg = [f"{round(m,2):9.3f}" for m in np.mean(all, axis=0)]
        # print(f"Avg top-{top} valid {metric.__name__} {', '.join(avg)}")

    R = pd.DataFrame(data=topscores, columns=features_names)
    R.index = [f"top-{top} {'OOB' if use_oob else 'training'} {metric.__name__}" for top in range(top_features_range[0], top_features_range[1] + 1)]
    return R, ols_I, shap_ols_I, rf_I, perm_I, our_I


def get_multiple_imps(X, y, n_shap=300, n_estimators=50, stratpd_min_samples_leaf=10,
                      n_stratpd_trees=1,
                      bootstrap=False,
                      catcolnames=set(),
                      min_slopes_per_x=10,
                      supervised=True):

    lm = LinearRegression()
    lm.fit(X, y)
    X_ = StandardScaler().fit_transform(X)
    X_ = pd.DataFrame(X_, columns=X.columns)
    ols_I, score = linear_model_importance(lm, X_, y)

    ols_shap_I = shap_importances(lm, X, n_shap=len(X)) # fast enough so use all data

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    rf.fit(X, y)
    rf_I = shap_importances(rf, X, n_shap)

    perm_I = rfpimp.importances(rf, X, y) # permutation

    ours_I = importances(X, y, verbose=False,
                         min_samples_leaf=stratpd_min_samples_leaf,
                         n_trees = n_stratpd_trees,
                         bootstrap=bootstrap,
                         catcolnames=catcolnames,
                         min_slopes_per_x=min_slopes_per_x,
                         supervised=supervised)
    return ols_I, ols_shap_I, rf_I, perm_I, ours_I


def plot_topk(R, ax, k=None):
    GREY = '#444443'
    if k is None:
        k = R.shape[0]
    feature_counts = range(1, k + 1)
    fmts = ['o-','v-','s-','x-','-']
    for i,technique in enumerate(R.columns):
        fmt = fmts[i]
        ms = 8
        if fmt == 'x-': ms = 11
        if technique == 'StratImpact':
            color = '#A22396' #'#4574B4'  # '#415BA3'
            lw = 2
        else:
            color = GREY# '#415BA3'  # '#8E8E8F'
            lw = .5
        ax.plot(feature_counts, R[technique][:k], fmt, lw=lw, label=technique,
                c=color, alpha=.9, markersize=ms, fillstyle='none')

    plt.legend()
    ax.set_xlabel("Top $k$ most important features")
    ax.xaxis.set_ticks(feature_counts)


def plot_topk_bars(R, ax):
    feature_counts = np.array(range(1, R.shape[0] + 1))
    for i, technique in enumerate(R.columns):
        ax.bar(feature_counts + 0.15 * i, R[technique], width=0.15, label=technique)

    plt.legend()
    ax.set_xlabel("Top $k$ most important features")
    ax.xaxis.set_ticks(feature_counts)


def load_flights(n):
    """
    Download from https://www.kaggle.com/usdot/flight-delays/download and save
    flight-delays.zip; unzip to convenient data dir.  Save time by storing as
    feather.  5.8M records.
    """
    dir = "data/flight-delays"
    if os.path.exists(dir+"/flights.feather"):
        df_flights = pd.read_feather(dir + "/flights.feather")
    else:
        df_flights = pd.read_csv(dir+"/flights.csv", low_memory=False)
        df_flights.to_feather(dir+"/flights.feather")

    df_flights['dayofyear'] = pd.to_datetime(
        df_flights[['YEAR', 'MONTH', 'DAY']]).dt.dayofyear
    df_flights = df_flights[
        (df_flights['CANCELLED'] == 0) & (df_flights['DIVERTED'] == 0)]

    features = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'dayofyear',
                'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                'SCHEDULED_DEPARTURE', 'FLIGHT_NUMBER', 'TAIL_NUMBER',
                'AIR_TIME', 'DISTANCE',
                'TAXI_IN', 'TAXI_OUT',
                'DEPARTURE_TIME',
                'SCHEDULED_ARRIVAL',
                'SCHEDULED_TIME',
                'ARRIVAL_DELAY']  # target

    df_flights = df_flights[features]
    df_flights = df_flights.dropna()  # ignore missing stuff for ease and reduce size
    df_flights = df_flights.sample(n)
    df_string_to_cat(df_flights)
    df_cat_to_catcode(df_flights)

    X, y = df_flights.drop('ARRIVAL_DELAY', axis=1), df_flights['ARRIVAL_DELAY']

    return X, y, df_flights


def load_bulldozer():
    """
    Download Train.csv data from https://www.kaggle.com/c/bluebook-for-bulldozers/data
    and save in data subdir
    """
    if os.path.exists("data/bulldozer-train.feather"):
        print("Loading cached version")
        df = pd.read_feather("data/bulldozer-train.feather")
    else:
        dtypes = {col: str for col in
                  ['fiModelSeries', 'Coupler_System', 'Grouser_Tracks',
                   'Hydraulics_Flow']}
        df = pd.read_csv('data/Train.csv', dtype=dtypes, parse_dates=['saledate'])  # 35s load
        df = df.sort_values('saledate')
        df = df.reset_index(drop=True)
        df.to_feather("data/bulldozer-train.feather")

    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    df.loc[df.eval("MachineHours==0"),
           'MachineHours'] = np.nan
    fix_missing_num(df, 'MachineHours')

    df.loc[df.YearMade < 1950, 'YearMade'] = np.nan
    fix_missing_num(df, 'YearMade')
    df_split_dates(df, 'saledate')
    df['age'] = df['saleyear'] - df['YearMade']
    sizes = {None: 0, 'Mini': 1, 'Compact': 1, 'Small': 2, 'Medium': 3,
             'Large / Medium': 4, 'Large': 5}
    df['ProductSize'] = df['ProductSize'].map(sizes).values

    df['Enclosure'] = df['Enclosure'].replace('EROPS w AC', 'EROPS AC')
    df['Enclosure'] = df['Enclosure'].replace('None or Unspecified', np.nan)
    df['Enclosure'] = df['Enclosure'].replace('NO ROPS', np.nan)
    df['AC'] = df['Enclosure'].fillna('').str.contains('AC')
    df['AC'] = df['AC'].astype(int)
    # print(df.columns)

    # del df['SalesID']  # unique sales ID so not generalizer (OLS clearly overfits)
    # delete MachineID as it has inconsistencies and errors per Kaggle

    basefeatures = ['ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID',
                    'MachineHours'
                    ]
    X = df[basefeatures+
           ['age',
            'YearMade_na',
            'AC',
            'ProductSize',
            'MachineHours_na',
            'saleyear', 'salemonth', 'saleday', 'saledayofweek', 'saledayofyear']
           ]

    X = X.fillna(0)  # flip missing numeric values to zeros
    y = df['SalePrice']
    return X, y


def load_rent(n:int, clean_prices=True):
    """
    Download train.json from https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data
    and save into data subdir.
    """
    df = pd.read_json('data/train.json')

    # Create ideal numeric data set w/o outliers etc...

    if clean_prices:
        df = df[(df.price > 1_000) & (df.price < 10_000)]

    df = df[df.bathrooms <= 6]  # There's almost no data for 6 and above with small sample
    df = df[(df.longitude != 0) | (df.latitude != 0)]
    df = df[(df['latitude'] > 40.55) & (df['latitude'] < 40.94) &
            (df['longitude'] > -74.1) & (df['longitude'] < -73.67)]
    df['interest_level'] = df['interest_level'].map({'low': 1, 'medium': 2, 'high': 3})
    df["num_desc_words"] = df["description"].apply(lambda x: len(x.split()))
    df["num_features"] = df["features"].apply(lambda x: len(x))
    df["num_photos"] = df["photos"].apply(lambda x: len(x))

    hoods = {
        "hells": [40.7622, -73.9924],
        "astoria": [40.7796684, -73.9215888],
        "Evillage": [40.723163774, -73.984829394],
        "Wvillage": [40.73578, -74.00357],
        "LowerEast": [40.715033, -73.9842724],
        "UpperEast": [40.768163594, -73.959329496],
        "ParkSlope": [40.672404, -73.977063],
        "Prospect Park": [40.93704, -74.17431],
        "Crown Heights": [40.657830702, -73.940162906],
        "financial": [40.703830518, -74.005666644],
        "brooklynheights": [40.7022621909, -73.9871760513],
        "gowanus": [40.673, -73.997]
    }
    for hood, loc in hoods.items():
        # compute manhattan distance
        df[hood] = np.abs(df.latitude - loc[0]) + np.abs(df.longitude - loc[1])
        df[hood] *= 1000 # GPS range is very tight so distances are very small. bump up
    hoodfeatures = list(hoods.keys())

    df = df.sort_values(by='created').sample(min(n,len(df)), replace=False)  # get a small subsample
    # df = df.sort_values(by='created')  # time-sensitive dataset
    # df = df.iloc[-n:]

    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price',
                  'interest_level']+
                 hoodfeatures+
                 ['num_photos', 'num_desc_words', 'num_features']]
    # print(df_rent.head(3))

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    return X, y
