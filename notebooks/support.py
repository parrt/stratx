from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import statsmodels.api as sm

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

    basefeatures = ['SalesID', 'MachineID', 'ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID', 'MachineHours']
    X = df[basefeatures+
           ['age',
            'YearMade_na',
            'AC',
            'saleyear', 'salemonth', 'saleday', 'saledayofweek', 'saledayofyear']+
           ['ProductSize']]

    X = X.fillna(0)  # flip missing numeric values to zeros
    y = df['SalePrice']
    return X, y


def linear_model_importance(model, X_test, X_train, y_test, y_train):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    imp = np.abs(model.coef_)

    # use statsmodels to get stderr for betas
    if isinstance(model, LinearRegression):
        beta_stderr = sm.OLS(y_train, X_train).fit().bse
        imp /= beta_stderr
    # stderr for betas makes no sense in Lasso

    imp /= np.sum(imp) # normalize
    I = pd.DataFrame(data={'Feature': X_train.columns, 'Importance': imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I, score


def ginidrop_importances(rf, X):
    ginidrop_I = rf.feature_importances_
    ginidrop_I = pd.DataFrame(data={'Feature': X.columns, 'Importance': ginidrop_I})
    ginidrop_I = ginidrop_I.set_index('Feature')
    ginidrop_I = ginidrop_I.sort_values('Importance', ascending=False)
    return ginidrop_I


def shap_importances(model, X, normalize=True):
    start = timer()
    if isinstance(model, RandomForestRegressor) or isinstance(model, GradientBoostingRegressor):
        explainer = shap.TreeExplainer(model, data=X, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X, check_additivity=True)
    elif isinstance(model, Lasso) or isinstance(model, LinearRegression):
        shap_values = shap.LinearExplainer(model, X, feature_dependence='independent').shap_values(X)
    shapimp = np.mean(np.abs(shap_values), axis=0)
    stop = timer()
    print(f"SHAP time for {len(X)} records = {(stop - start):.1f}s")

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


def get_multiple_imps(X, y, test_size=0.2, n_shap=100, n_estimators=50, min_samples_leaf=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    lm = LinearRegression()
    lm.fit(X_train, y_train)
    ols_I, score = linear_model_importance(lm, X_test, X_train, y_test, y_train)

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    rf.fit(X_train, y_train)
    rf_I = shap_importances(rf, X_train[:n_shap])

    ours_I = impact_importances(X, y, verbose=False, min_samples_leaf=min_samples_leaf)
    return ols_I, rf_I, ours_I


def avg_model_for_top_features(name:('OLS', 'RF', 'OUR'), X_test, X_train, y_test, y_train,
                               models={'RF'}):
    scores = []

    # OLS
    if 'OLS' in models:
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        s = lm.score(X_test, y_test)
        scores.append(s)

    # Lasso
    if 'LASSO' in models:
        lm = Lasso(alpha=.1)
        # model = GridSearchCV(Lasso(), cv=5,
        #                    param_grid={"alpha": [1, .5, .1, .01, .001]})
        # model.fit(X_train, y_train)
        # lm = model.best_estimator_
        # print("LASSO best:",model.best_params_)
        lm.fit(X_train, y_train)
        s = lm.score(X_test, y_test)
        scores.append(s)

    # SVM
    if 'SVM' in models:
        svr = svm.SVR(gamma=0.001, C=5000)
        # model = GridSearchCV(svm.SVR(), cv=5,
        #                    param_grid={"C": [1, 1000, 2000, 3000, 5000],
        #                                "gamma": [1e-5, 1e-4, 1e-3]})
        # model.fit(X_train, y_train)
        # svr = model.best_estimator_
        # print("SVM best:",model.best_params_)
        svr.fit(X_train, y_train)
        s = svr.score(X_test, y_test)
        scores.append(s)

    # GBM
    if 'GBM' in models:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = xgb.XGBRegressor(objective ='reg:squarederror')
            x.fit(X_train, y_train)
            y_pred = x.predict(X_test)
            s = r2_score(y_test, y_pred)
        # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        # num_round = 2
        # watchlist = [(X_test, 'eval'), (X_train, 'train')]
        # dtrain = xgb.DMatrix(X_train, label=y_train)
        # bst = xgb.train(param, dtrain, num_round, watchlist)
        scores.append(s)

    # RF
    if 'RF' in models:
        n_estimators = 40
        rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        s = rf.score(X_test, y_test)
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
