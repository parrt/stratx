import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from collections import OrderedDict

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
    correlations = [spearmanr(X[colname], y)[0] for colname in X.columns]
    I = pd.DataFrame(data={'Feature': X.columns, 'Importance': np.abs(correlations)})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


def pca_importances(X):
    """
    Get the first principle component and get "loading" from that as
    the feature importances.  First component won't explain everything but
    could still give useful importances in some cases.
    """
    X_ = StandardScaler().fit_transform(X)
    pca = PCA(svd_solver='full')
    pca.fit(X_)

    # print(list(pca.explained_variance_ratio_))
    # print("Explains this percentage:", pca.explained_variance_ratio_[0])

    # print( cross_val_score(pca, X) )

    correlations = np.abs(pca.components_[0, :])
    # print(correlations)
    I = pd.DataFrame(data={'Feature': X.columns, 'Importance': correlations})
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


def shap_importances(model, X_train, X_test, n_shap, normalize=True, sort=True):
    start = timer()
    #X = shap.kmeans(X, k=n_shap)
    X_test = X_test.sample(n=min(n_shap, len(X_test)), replace=False)
    if isinstance(model, RandomForestRegressor) or \
        isinstance(model, GradientBoostingRegressor) or \
        isinstance(model, xgb.XGBRegressor):
        """
        We get this warning for big X_train so choose smaller
        'Passing 20000 background samples may lead to slow runtimes. Consider using shap.sample(data, 100) to create a smaller background data set.'
        """
        explainer = shap.TreeExplainer(model, data=shap.sample(X_train, 100), feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    elif isinstance(model, Lasso) or isinstance(model, LinearRegression):
        shap_values = shap.LinearExplainer(model, X_train, feature_dependence='independent').shap_values(X_train)
    else:
        # gotta use really small sample; verrry slow
        explainer = shap.KernelExplainer(model.predict, X_train.sample(frac=.1))
        shap_values = explainer.shap_values(X_test, nsamples='auto')
    shapimp = np.mean(np.abs(shap_values), axis=0)
    stop = timer()
    print(f"SHAP time for {len(X_test)} test records using {model.__class__.__name__} = {(stop - start):.1f}s")

    total_imp = np.sum(shapimp)
    normalized_shap = shapimp
    if normalize:
        normalized_shap = shapimp / total_imp

    # print("SHAP", normalized_shap)
    shapI = pd.DataFrame(data={'Feature': X_test.columns, 'Importance': normalized_shap})
    shapI = shapI.set_index('Feature')
    if sort:
        shapI = shapI.sort_values('Importance', ascending=False)
    # plot_importances(shapI)
    return shapI


def compare_top_features(X, y, top_features_range=None,
                         n_shap=300,
                         metric = mean_absolute_error,
                         use_oob = False,
                         time_sensitive=False,
                         trials=1,
                         n_stratpd_trees=1,
                         bootstrap=False,
                         stratpd_min_samples_leaf=10,
                         min_slopes_per_x=15,
                         catcolnames=set(),
                         supervised=True,
                         include=['Spearman', 'PCA', 'OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact'],
                         drop=()):
    if use_oob and metric!=r2_score:
        #     print("Warning: use_oob can only give R^2; flipping metric to r2_score")
        metric=r2_score

    include = include.copy()
    for feature in drop:
        include.remove(feature)

    if time_sensitive:
        n_test = int(0.20 * len(X))
        n_train = len(X) - n_test
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_estimators = 40 # for both SHAP and testing purposes

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Sanity check: R^2 OOB on {X_train.shape[0]} training records: {rf.oob_score_:.3f}, training {metric.__name__}={metric(y_train, rf.predict(X_train))}")
    print(f"testing {metric.__name__}={metric(y_test, rf.predict(X_test))}")

    all_importances = get_multiple_imps(X_train, y_train, X_test, y_test,
                                        n_stratpd_trees=n_stratpd_trees,
                                        bootstrap=bootstrap,
                                        stratpd_min_samples_leaf=stratpd_min_samples_leaf,
                                        n_estimators=n_estimators,
                                        n_shap=n_shap,
                                        catcolnames=catcolnames,
                                        min_slopes_per_x=min_slopes_per_x,
                                        supervised=supervised,
                                        include=include)

    print("Spearman\n", all_importances['Spearman'])
    print("PCA\n", all_importances['PCA'])
    print("OLS\n", all_importances['OLS'])
    print("OLS SHAP\n", all_importances['OLS SHAP'])
    print("RF SHAP\n", all_importances['RF SHAP'])
    print("RF perm\n", all_importances['RF perm'])
    print("Our importances\n",all_importances['StratImpact'])

    if top_features_range is None:
        top_features_range = (1, X.shape[1])

    features_names = include #['OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact']

    print(f"n_train={len(X_train)}, n_top={top_features_range[1]}, n_estimators={n_estimators}, n_shap={n_shap}, min_samples_leaf={stratpd_min_samples_leaf}")
    topscores = []
    for top in range(top_features_range[0], top_features_range[1] + 1):
        # ols_top = ols_I.iloc[:top, 0].index.values
        # shap_ols_top = shap_ols_I.iloc[:top, 0].index.values
        # rf_top = rf_I.iloc[:top, 0].index.values
        # perm_top = perm_I.iloc[:top, 0].index.values
        # our_top = our_I.iloc[:top, 0].index.values
        # features_set = [ols_top, shap_ols_top, rf_top, perm_top, our_top]
        all = []
        for i in range(trials):
            # print(i, end=' ')
            results = []
            feature_sets = [I.iloc[:top, 0].index.values for I in all_importances.values() if I is not None]
            for name, features in zip(include, feature_sets):
                # print(f"Train with {features} from {name}")
                # Train RF model with top-k features
                rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=use_oob,
                                           min_samples_leaf=1, n_jobs=-1)
                rf.fit(X_train[features], y_train)
                if use_oob:
                    # make so it's a metric; lower is better
                    s = rf.oob_score_ if rf.oob_score_ >= 0 else 0
                    s = 1 - s
                else:
                    y_pred = rf.predict(X_test[features])
                    s = metric(y_test, y_pred)

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

    # unpack for users
    return (R, *all_importances.values())


def get_multiple_imps(X_train, y_train, X_test, y_test, n_shap=300, n_estimators=50,
                      stratpd_min_samples_leaf=10,
                      n_stratpd_trees=1,
                      bootstrap=False,
                      catcolnames=set(),
                      min_slopes_per_x=10,
                      supervised=True,
                      include=['Spearman', 'PCA', 'OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact']):
    spear_I = pca_I = ols_I = ols_shap_I = rf_I = perm_I = ours_I = None

    if 'Spearman' in include:
        spear_I = spearmans_importances(X_train, y_train)

    if 'PCA' in include:
        pca_I = pca_importances(X_train)

    if "OLS" in include:
        X_train_ = StandardScaler().fit_transform(X_train)
        X_train_ = pd.DataFrame(X_train_, columns=X_train.columns)
        lm = LinearRegression()
        lm.fit(X_train_, y_train)
        ols_I, score = linear_model_importance(lm, X_train_, y_train)

    if "OLS SHAP" in include:
        X_train_ = StandardScaler().fit_transform(X_train)
        X_train_ = pd.DataFrame(X_train_, columns=X_train.columns)
        lm = LinearRegression()
        lm.fit(X_train_, y_train)
        ols_shap_I = shap_importances(lm, X_train_, X_test, n_shap=len(X_test)) # fast enough so use all data

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    rf.fit(X_train, y_train)

    if "RF SHAP" in include:
        rf_I = shap_importances(rf, X_train, X_test, n_shap)

    if "RF perm" in include:
        perm_I = rfpimp.importances(rf, X_test, y_test) # permutation; drop in test accuracy

    if "StratImpact" in include:
        # RF SHAP and RF perm get to look at the test data to decide which features
        # are more predictive and useful for generality's sake so it's fair to
        # let this method see all data as well
        # X_full = pd.concat([X_train, X_test], axis=0)
        # y_full = pd.concat([y_train, y_test], axis=0)
        X_full = X_train
        y_full = y_train
        ours_I = importances(X_full, y_full, verbose=False,
                             min_samples_leaf=stratpd_min_samples_leaf,
                             n_trees=n_stratpd_trees,
                             bootstrap=bootstrap,
                             catcolnames=catcolnames,
                             min_slopes_per_x=min_slopes_per_x,
                             supervised=supervised)
    d = OrderedDict()
    d['Spearman'] = spear_I
    d['PCA'] = pca_I
    d['OLS'] = ols_I
    d['OLS SHAP'] = ols_shap_I
    d['RF SHAP'] = rf_I
    d["RF perm"] = perm_I
    d['StratImpact'] = ours_I
    return d


def plot_topk(R, ax, k=None):
    GREY = '#444443'
    if k is None:
        k = R.shape[0]
    feature_counts = range(1, k + 1)
    fmts = {'Spearman':'P-', 'PCA':'D-',
            'OLS':'o-', 'OLS SHAP':'v-', 'RF SHAP':'s-',
            "RF perm":'x-', 'StratImpact':'-'}
    for i,technique in enumerate(R.columns):
        fmt = fmts[technique]
        ms = 8
        if fmt == 'x-': ms = 11
        if fmt == 'P-': ms = 11
        if technique == 'StratImpact':
            color = '#A22396' #'#4574B4'  # '#415BA3'
            lw = 2
        else:
            color = GREY# '#415BA3'  # '#8E8E8F'
            lw = .5
        ax.plot(feature_counts, R[technique][:k], fmt, lw=lw, label=technique,
                c=color, alpha=.9, markersize=ms, fillstyle='none')

    plt.legend(loc='upper right')  # usually it's out of the way
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
    if os.path.exists("data/bulldozer-train-all.feather"):
        print("Loading cached version...")
        df = pd.read_feather("data/bulldozer-train-all.feather")
    else:
        dtypes = {col: str for col in
                  ['fiModelSeries', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow']}
        df = pd.read_csv('data/Train.csv', dtype=dtypes, parse_dates=['saledate'])  # 35s load
        df = df.sort_values('saledate')
        df = df.reset_index(drop=True)
        df.to_feather("data/bulldozer-train-all.feather")

    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    df.loc[df.eval("MachineHours==0"),
           'MachineHours'] = np.nan
    fix_missing_num(df, 'MachineHours')

    df.loc[df.YearMade < 1950, 'YearMade'] = np.nan
    fix_missing_num(df, 'YearMade')
    df_split_dates(df, 'saledate')
    df['age'] = df['saleyear'] - df['YearMade']
    df['YearMade'] = df['YearMade'].astype(int)
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
