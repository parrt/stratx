from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from timeit import default_timer as timer

from rfpimp import *
from impimp import *

import shap

import xgboost as xgb


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


def load_rent(n=3_000):
    df = pd.read_json('data/train.json')

    # Create ideal numeric data set w/o outliers etc...
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

    df = df.sort_values(by='created').sample(n, replace=False)  # get a small subsample
    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price',
                  'interest_level']+
                 hoodfeatures+
                 ['num_photos', 'num_desc_words', 'num_features']]
    # print(df_rent.head(3))

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    return X, y


def load_bulldozer():
    df = pd.read_feather("../notebooks/data/bulldozer-train.feather")

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


def linear_model_importance(model, X, y):
    model.fit(X, y)
    score = model.score(X, y)

    imp = np.abs(model.coef_)

    # use statsmodels to get stderr for betas
    if isinstance(model, LinearRegression):
        # stderr for betas makes no sense in Lasso
        beta_stderr = sm.OLS(y, X).fit().bse
        imp /= beta_stderr

    imp /= np.sum(imp) # normalize
    I = pd.DataFrame(data={'Feature': X.columns, 'Importance': imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I, score


def ginidrop_importances(rf, X):
    ginidrop_I = rf.feature_importances_
    ginidrop_I = pd.DataFrame(data={'Feature': X.columns, 'Importance': ginidrop_I})
    ginidrop_I = ginidrop_I.set_index('Feature')
    ginidrop_I = ginidrop_I.sort_values('Importance', ascending=False)
    return ginidrop_I


def shap_importances(model, X, n_shap, normalize=True):
    start = timer()
    #X = shap.kmeans(X, k=n_shap)
    X = X.sample(n=n_shap, replace=False)
    if isinstance(model, RandomForestRegressor) or isinstance(model, GradientBoostingRegressor):
        explainer = shap.TreeExplainer(model, data=X, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X, check_additivity=False)
    elif isinstance(model, Lasso) or isinstance(model, LinearRegression):
        shap_values = shap.LinearExplainer(model, X, feature_dependence='independent').shap_values(X)
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
    shapI = shapI.sort_values('Importance', ascending=False)
    # plot_importances(shapI)
    return shapI


def get_multiple_imps(X, y, n_shap=300, n_estimators=50, min_samples_leaf=10,
                      catcolnames=set(),
                      min_slopes_per_x=10):

    lm = LinearRegression()
    lm.fit(X, y)
    X_ = pd.DataFrame(normalize(X), columns=X.columns)
    ols_I, score = linear_model_importance(lm, X_, y.values)
    ols_shap_I = shap_importances(lm, X_, n_shap)

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    rf.fit(X, y)
    rf_I = shap_importances(rf, X, n_shap)

    ours_I = impact_importances(X, y, verbose=False, min_samples_leaf=min_samples_leaf,
                                catcolnames=catcolnames,
                                min_slopes_per_x=min_slopes_per_x)
    return ols_I, ols_shap_I, rf_I, ours_I


def rmse(y_true, y_pred):
    return np.sqrt( mean_squared_error(y_true, y_pred) )


def avg_model_for_top_features(X, y,
                               models={'RF'},
                               metric=mean_absolute_error,
                               use_oob=False):
    if use_oob and metric!=r2_score:
        print("Warning: use_oob can only give R^2; flipping metric to r2_score")
    scores = []

    # OLS
    if 'OLS' in models:
        lm = LinearRegression()
        lm.fit(X, y)
        y_pred = lm.predict(X)
        s = metric(y, y_pred)
        scores.append(s)

    # Lasso
    if 'LASSO' in models:
        lm = Lasso(alpha=.1)
        # model = GridSearchCV(Lasso(), cv=5,
        #                    param_grid={"alpha": [1, .5, .1, .01, .001]})
        # model.fit(X_train, y_train)
        # lm = model.best_estimator_
        # print("LASSO best:",model.best_params_)
        lm.fit(X, y)
        y_pred = lm.predict(X)
        s = metric(y, y_pred)
        scores.append(s)

    # SVM
    if 'SVM' in models:
        svr = svm.SVR(gamma=0.001, C=5000)
        # model = GridSearchCV(svm.SVR(), cv=5,
        #                    param_grid={"C": [1, 1000, 2000, 3000, 5000],
        #                                "gamma": [1e-5, 1e-4, 1e-3]})
        # model.fit(X, y)
        # svr = model.best_estimator_
        # print("SVM best:",model.best_params_)
        svr.fit(X, y)
        y_pred = svr.predict(X)
        s = metric(y, y_pred)
        scores.append(s)

    # GBM
    if 'GBM' in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = xgb.XGBRegressor(objective ='reg:squarederror')
            x.fit(X, y)
            y_pred = x.predict(X)
            s = metric(y, y_pred)
        # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        # num_round = 2
        # watchlist = [(X, 'eval'), (X, 'train')]
        # dtrain = xgb.DMatrix(X, label=y)
        # bst = xgb.train(param, dtrain, num_round, watchlist)
        scores.append(s)

    # RF
    if 'RF' in models:
        n_estimators = 40
        rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=use_oob, min_samples_leaf=1, n_jobs=-1)
        rf.fit(X, y)
        if use_oob:
            s = rf.oob_score_
        else:
            y_pred = rf.predict(X)
            s = metric(y, y_pred)
        scores.append(s)

    # print(scores)

    # If we use OLS or RF to get recommended top features, not fair to use those
    # models to measure fitness of recommendation; drop those scores from avg.
    # if name == 'OLS':
    #     del scores[0]
    # elif name == 'RF': # SHAP RF
    #     del scores[4]

    return np.mean(scores)
    # for now, return just RF
    # return s


def compare_top_features(X, y, top_features_range=None,
                         n_shap=300,
                         metric = mean_absolute_error,
                         use_oob = False,
                         n_estimators=40,
                         trials=1,
                         min_samples_leaf=10,
                         min_slopes_per_x=5):

    rf = RandomForestRegressor(n_estimators=40, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"Sanity check: R^2 OOB on {X.shape[0]} records: {rf.oob_score_:.3f}, training {metric.__name__}={metric(y, rf.predict(X))}")

    ols_I, shap_ols_I, rf_I, our_I = get_multiple_imps(X, y,
                                                       min_samples_leaf=min_samples_leaf,
                                                       n_estimators=n_estimators,
                                                       n_shap=n_shap,
                                                       catcolnames={'AC','ModelID'},
                                                       min_slopes_per_x=min_slopes_per_x)
    print("OLS\n", ols_I)
    print("OLS SHAP\n", shap_ols_I)
    print("RF SHAP\n",rf_I)
    print("OURS\n",our_I)

    if top_features_range is None:
        top_features_range = (1, X.shape[1])

    features_names = ['OLS', 'OLS SHAP', 'RF SHAP', 'OUR']

    print("OUR FEATURES", our_I.index.values)

    print("n, n_top, n_estimators, n_shap, min_samples_leaf",
          len(X), top_features_range[1], n_estimators, n_shap, min_samples_leaf)
    topscores = []
    for top in range(top_features_range[0], top_features_range[1] + 1):
        ols_top = ols_I.iloc[:top, 0].index.values
        shap_ols_top = shap_ols_I.iloc[:top, 0].index.values
        rf_top = rf_I.iloc[:top, 0].index.values
        our_top = our_I.iloc[:top, 0].index.values
        features_set = [ols_top, shap_ols_top, rf_top, our_top]
        all = []
        for i in range(trials):
            # print(i, end=' ')
            results = []
            for name, features in zip(features_names, features_set):
                # print(f"Train with {features} from {name}")
                s = avg_model_for_top_features(X[features], y, metric=metric, use_oob=use_oob)
                results.append(s)
                # print(f"{name} valid R^2 {s:.3f}")
            all.append(results)
        # print(pd.DataFrame(data=all, columns=['OLS','RF','Ours']))
        # print()
        topscores.append( [round(m,2) for m in np.mean(all, axis=0)] )

        # avg = [f"{round(m,2):9.3f}" for m in np.mean(all, axis=0)]
        # print(f"Avg top-{top} valid {metric.__name__} {', '.join(avg)}")

    R = pd.DataFrame(data=topscores, columns=features_names)
    R.index = [f"top-{top} {'OOB' if use_oob else 'training'} {metric.__name__}" for top in range(top_features_range[0], top_features_range[1] + 1)]
    return R
