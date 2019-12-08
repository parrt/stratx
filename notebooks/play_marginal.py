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

from rfpimp import *

import xgboost as xgb

import matplotlib
import time
import timeit
from timeit import default_timer as timer
from stratx.partdep import *
#from stratx.cy_partdep import *
from rfpimp import plot_importances
import rfpimp
import shap
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata

palette = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928"
]

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


def toy_weather_data():
    df_yr1 = toy_weather_data_()
    df_yr1['year'] = 1980
    df_yr2 = toy_weather_data_()
    df_yr2['year'] = 1981
    df_yr3 = toy_weather_data_()
    df_yr3['year'] = 1982
    df_raw = pd.concat([df_yr1, df_yr2, df_yr3], axis=0)
    df = df_raw.copy()
    return df


def toy_weather_data_():
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


def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n // 2
    nwomen = n // 2
    df['sex'] = ['M'] * nmen + ['F'] * nwomen
    df.loc[df['sex'] == 'F', 'pregnant'] = np.random.randint(0, 2, size=(nwomen,))
    df.loc[df['sex'] == 'M', 'pregnant'] = 0
    df.loc[df['sex'] == 'M', 'height'] = 5 * 12 + 8 + np.random.uniform(-7, +8,
                                                                        size=(nmen,))
    df.loc[df['sex'] == 'F', 'height'] = 5 * 12 + 5 + np.random.uniform(-4.5, +5,
                                                                        size=(nwomen,))
    df.loc[df['sex'] == 'M', 'education'] = 10 + np.random.randint(0, 8, size=nmen)
    df.loc[df['sex'] == 'F', 'education'] = 12 + np.random.randint(0, 8, size=nwomen)
    df['weight'] = 120 \
                   + (df['height'] - df['height'].min()) * 10 \
                   + df['pregnant'] * 30 \
                   - df['education'] * 1.5
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    eqn = "y = 120 + 10(x_{height} - min(x_{height})) + 30x_{pregnant} - 1.5x_{education}"
    return df, eqn


def synthetic_poly_data(n, p, noise=0):
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p) # get p random coefficients
    # coeff = np.array([1,1,1,1,1,1,1,1,1])
    coeff = np.array([1,1,1])
    exponents = np.array([1,1,1])
    for j in range(p):
        df[f'x{j+1}'] = np.round(np.random.random_sample(size=n)*5,1)

    if noise>0:
        for j in range(noise):
            df[f'noise{p+j+1}'] = np.round(np.random.random_sample(size=n)*1,1)

    # multiply coefficients x each column (var) and sum along columns
    yintercept = 10
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}']**exponents[i] for i in range(p)], axis=0 ) + yintercept

    # add hidden var
    #df['y'] += np.random.random_sample(size=n)*2-1

    terms = [f"{coeff[i]:.1f}x_{i+1}^{exponents[i]}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms)
    if noise>0:
        eqn += ' + '
        eqn += '+'.join([f'noise{p+j+1}' for j in range(noise)])
    return df, coeff, eqn


def synthetic_poly2dup_data(n, p):
    """
    SHAP seems to make x3, x1 very different despite same coeffs. over several runs,
    it varies a lot. e.g., i see one where x1,x2,x3 are same as they should be.
    very unstable. same with permutation. dropcol shows x1,x3 as 0.

    Adding noise so x3=x1+noise seems to overcome need for RF in stratpd and we
    get roughly equal imps.   SHAP still varies.
    """
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p)*10 # get p random coefficients
    coeff = np.array([5, 3, 9])
    coeff = np.array([1,1,1])
    for i in range(p):
        df[f'x{i+1}'] = np.round(np.random.random_sample(size=n)*10,5)
    df['x3'] = df['x1'] + np.random.random_sample(size=n) # copy x1 into x3
    yintercept = 100
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    # df['y'] = 5*df['x1'] + 3*df['x2'] + 9*df['x3']
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms)
    return df, coeff, eqn+" where x_3 = x_1 + noise"


def shap_importances(model, X_train, X_test, n=20_000):
    X_train = X_train[-n:]
    X_test = X_test[-n:]
    start = time.time()
    if isinstance(model, RandomForestRegressor) or isinstance(model, GradientBoostingRegressor):
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
    elif isinstance(model, Lasso) or isinstance(model, LinearRegression):
        shap_values = shap.LinearExplainer(model, X_train, feature_dependence='independent').shap_values(X_test)
    else:
        shap_values = shap.KernelExplainer(model.predict, data=X_train).shap_values(X_test, l1_reg="aic")
    shapimp = np.mean(np.abs(shap_values), axis=0)
    stop = time.time()
    print(f"SHAP time for {len(X_train)} = {(stop - start):.1f}s")

    total_imp = np.sum(shapimp)

    normalized_shap = shapimp / total_imp
    # print("SHAP", normalized_shap)
    shapI = pd.DataFrame(data={'Feature': X_test.columns, 'Importance': normalized_shap})
    shapI = shapI.set_index('Feature')
    shapI = shapI.sort_values('Importance', ascending=False)
    # plot_importances(shapI)
    return shapI


def ginidrop_importances(rf, X):
    ginidrop_I = rf.feature_importances_
    ginidrop_I = pd.DataFrame(data={'Feature': X.columns, 'Importance': ginidrop_I})
    ginidrop_I = ginidrop_I.set_index('Feature')
    ginidrop_I = ginidrop_I.sort_values('Importance', ascending=False)
    return ginidrop_I


def compare_imp(rf, X, y, catcolnames=set(), eqn="n/a"):
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.2))

    I = impact_importances(X, y, catcolnames=catcolnames)
    plot_importances(I, imp_range=(0, 1), ax=axes[0])

    shap_I = shap_importances(rf, X)
    plot_importances(shap_I, ax=axes[1], imp_range=(0, 1))

    gini_I = ginidrop_importances(rf, X)
    plot_importances(gini_I, ax=axes[2])

    perm_I = rfpimp.importances(rf, X, y)
    plot_importances(perm_I, ax=axes[3])
    drop_I = rfpimp.dropcol_importances(rf, X, y)
    plot_importances(drop_I, ax=axes[4])

    axes[0].set_title("Impact Imp")
    axes[1].set_title("SHAP Imp")
    axes[2].set_title("ginidrop Imp")
    axes[3].set_title("Permute column")
    axes[4].set_title("Drop column")
    plt.suptitle(f"${eqn}$", y=1.02)


def impact_importances(X: pd.DataFrame,
                       y: pd.Series,
                       catcolnames=set(),
                       n_samples=None,  # use all by default
                       bootstrap_sampling=True,
                       n_trials:int=1,
                       n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                       verbose=False) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    if n_trees==1:
        bootstrap_sampling = False

    n,p = X.shape
    imps = np.zeros(shape=(p, n_trials)) # track p var importances for ntrials; cols are trials
    for i in range(n_trials):
        bootstrap_sample_idxs = resample(range(n), n_samples=n_samples, replace=bootstrap_sampling)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        imps[:,i] = impact_importances_(X_, y_, catcolnames=catcolnames,
                                        n_trees=n_trees,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=bootstrap,
                                        max_features=max_features,
                                        verbose=verbose)

    avg_imps = np.mean(imps, axis=1)
    stddev_imps = np.std(imps, axis=1)

    I = pd.DataFrame(data={'Feature': X.columns,
                           'Importance': avg_imps,
                           "Sigma":stddev_imps})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)

    return I


def impact_importances_(X: pd.DataFrame, y: pd.Series, catcolnames=set(),
                        n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                        verbose=False) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    p = X.shape[1]
    avg_pdp = np.zeros(shape=(p,)) # track avg pdp, not magnitude
    avg_abs_pdp = np.zeros(shape=(p,)) # like area under PDP curve but not including width
    total_avg_pdpy = 0.0
    for j, colname in enumerate(X.columns):
        # Ask stratx package for the partial dependence of y with respect to X[colname]
        if colname in catcolnames:
            start = time.time()
            leaf_histos, avg_per_cat, ignored = \
                cat_partial_dependence(X, y, colname=colname,
                                       ntrees=n_trees,
                                       min_samples_leaf=min_samples_leaf,
                                       bootstrap=bootstrap,
                                       max_features=max_features,
                                       verbose=verbose)
            #         print(f"Ignored for {colname} = {ignored}")
            stop = time.time()
            # print(f"PD time {(stop - start) * 1000:.0f}ms")
            min_avg_value = np.nanmin(avg_per_cat)
            avg_per_cat_from_0 = avg_per_cat - min_avg_value # all positive now, relative to 0 for lowest cat
            # some cats have NaN, such as 0th which is for "missing values"
            avg_abs_pdp[j] = np.nanmean(avg_per_cat_from_0)# * (ncats - 1)
            avg_pdp[j] = np.mean(avg_per_cat_from_0)
            total_avg_pdpy += avg_abs_pdp[j]
        else:
            start = time.time()
            leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy, ignored = \
                partial_dependence(X=X, y=y, colname=colname,
                                   ntrees=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=bootstrap,
                                   max_features=max_features,
                                   verbose=verbose)
        #         print(f"Ignored for {colname} = {ignored}")
            stop = time.time()
            # print(f"PD time {(stop-start)*1000:.0f}ms")
            avg_abs_pdp[j] = np.mean(np.abs(pdpy))# * (np.max(pdpx) - np.min(pdpx))
            avg_pdp[j] = np.mean(pdpy)
            total_avg_pdpy += avg_abs_pdp[j]

    # print("avg_pdp", avg_pdp, "sum", np.sum(avg_pdp), "avg y", np.mean(y), "avg y-min(y)", np.mean(y)-np.min(y))
    normalized_importances = avg_abs_pdp / total_avg_pdpy

    return normalized_importances


def plot_partials(X,y,eqn, yrange=(.5,1.5)):
    p = X.shape[1]
    fig, axes = plt.subplots(p, 1, figsize=(7, p*2+2))

    plt.suptitle('$'+eqn+'$', y=1.02)
    for j,colname in enumerate(X.columns):
        xj = X[colname]
        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy, ignored = \
            partial_dependence(X=X, y=y, colname=colname)
        # Plot dydx
        axes[j].scatter(pdpx, dydx, c='k', s=3)
        axes[j].plot([min(xj),max(xj)], [np.mean(dydx),np.mean(dydx)], c='orange')

        # Plot PD
        axes[j].plot(pdpx, pdpy, c='blue', lw=.5)

        axes[j].set_xlim(min(xj), max(xj))
        if yrange is not None:
            axes[j].set_ylim(*yrange)
        axes[j].set_xlabel(colname)
        axes[j].set_ylabel("y")
        axes[j].set_title(#(min(xj)+max(xj))/2, 1.4,
                     f"$\\sigma(dy)$={np.std(dydx):.3f}, $\\mu(pdpy)$={np.mean(pdpy):.3f}, $\\sigma(pdpy)$={np.std(pdpy):.3f}",
                     horizontalalignment='center')


def weight():
    df, eqn = toy_weight_data(2000)
    X = df.drop('weight', axis=1)
    y = df['weight']
    X['pregnant'] = X['pregnant'].astype(int)
    X['sex'] = X['sex'].map({'M': 0, 'F': 1}).astype(int)
    print(X.head())

    print(eqn)
    # X = df.drop('y', axis=1)
    # #X['noise'] = np.random.random_sample(size=len(X))
    # y = df['y']

    I = impact_importances(X, y, catcolnames={'sex', 'pregnant'})
    print(X.columns)
    print(I)
    plot_importances(I, imp_range=(0, 1.0))

    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X,y)

    compare_imp(rf, X, y, catcolnames={'sex','pregnant'}, eqn=eqn)


def weather():
    df = toy_weather_data()
    df_string_to_cat(df)
    df_cat_to_catcode(df)

    X = df.drop('temperature', axis=1)
    y = df['temperature']
    I = impact_importances(X, y, catcolnames={'state'})
    print(X.columns)
    print(I)
    plot_importances(I, imp_range=(0,1.0))

    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X,y)

    compare_imp(rf, X, y, catcolnames={'state'})


def poly():
    p=3
    df, coeff, eqn = synthetic_poly_data(1000,p,noise=1)
    #df, coeff, eqn = synthetic_poly2dup_data(2000,p)
    X = df.drop('y', axis=1)
    y = df['y']
    print(X.head())

    print(eqn)
    # X = df.drop('y', axis=1)
    # #X['noise'] = np.random.random_sample(size=len(X))
    # y = df['y']

    I = impact_importances(X, y, n_samples=None)
    print(I)
    # I = impact_importances(X, y, n_samples=100, bootstrap=False, n_trials=25)
    # print(I)
    plot_importances(I, imp_range=(0,1.0))
    plt.suptitle(f"${eqn}$", y=1.02)

    #compare_imp(rf, X, y, catcolnames={'sex','pregnant'}, eqn=eqn)

    #plot_partials(X, y, eqn, yrange=None)
    # plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)


def poly_dupcol():
    p=3
    df, coeff, eqn = synthetic_poly2dup_data(1000,p)
    X = df.drop('y', axis=1)
    y = df['y']
    print(X.head())

    print(eqn)
    # X = df.drop('y', axis=1)
    # #X['noise'] = np.random.random_sample(size=len(X))
    # y = df['y']

    I = impact_importances(X, y)
                           # ntrees=5, min_samples_leaf=10, bootstrap=False, max_features=1)
    print(I)
    # I = impact_importances(X, y, n_samples=100, bootstrap=False, n_trials=25)
    # print(I)
    plot_importances(I, imp_range=(0,1.0))


def unstable_SHAP():
    p=2 # p=2 and we're pretty stable and about even; p=2 implies don't add noisy x3 to y; hmm... shap seems stable but not even
        # damn, tried again and p=2 is the most unstable
        # bumping trees to 20 made shap more stable. both of us get x2 right but x1,x3
        # pair bounces quite a bit in shap, less so in ours but some. x3 is stealing importance
        # from x1 but x1 is more important as it's in the y equation.
    #p=3 # p=3 and x3 not added to y; ours has stairstep but stable, shap same order but can vary alot

    nplots=7
    fig, axes = plt.subplots(nplots, 2, figsize=(6, nplots+3))

    imps = np.zeros(shape=(p+1, nplots))
    shap_imps = np.zeros(shape=(p+1, nplots))

    for i in range(nplots):
        # make new data set each time
        df, coeff, eqn = synthetic_poly2dup_data(1000, p)
        X = df.drop('y', axis=1)
        y = df['y']

        print(eqn)
        # pass copies to ensure we don't alter them, giving fresh stuff to SHAP
        I = impact_importances(X.copy(), y.copy(), n_trials=1)
                               # ntrees=10, min_samples_leaf=10, bootstrap=False, max_features=1)
        imps[:,i] = I['Importance'].values
        plot_importances(I, imp_range=(0,1.0), ax=axes[i][0])
        axes[i][0].set_title("Impact", fontsize=9)

        rf = RandomForestRegressor(n_estimators=20, oob_score=True)
        rf.fit(X,y)
        shap_I = shap_importances(rf, X)
        shap_imps[:,i] = shap_I['Importance'].values
        plot_importances(shap_I, ax=axes[i][1], imp_range=(0, 1))
        axes[i][1].set_title(f"SHAP (OOB $R^2$ {rf.oob_score_:.2f})", fontsize=9)

    plt.suptitle(f"${eqn}$", y=1.0)

    stds = np.std(imps, axis=1)
    shap_stds = np.std(shap_imps, axis=1)
    print(stds)
    print(shap_stds)

    #compare_imp(rf, X, y, catcolnames={'sex','pregnant'}, eqn=eqn)

    #plot_partials(X, y, eqn, yrange=None)
    # plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)


def time_SHAP(n_estimators, n_records, model=None):
    df = pd.read_feather("../notebooks/data/bulldozer-train.feather")
    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    # basefeatures = ['ModelID', 'YearMade', 'MachineHours']
    basefeatures = ['SalesID', 'MachineID', 'ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID', 'MachineHoursCurrentMeter']

    # Get subsample; it's a (sorted) timeseries so get last records not random
    df = df.iloc[-n_records:]  # take only last records
    X, y = df[basefeatures], df['SalePrice']
    X = X.fillna(0)  # flip missing numeric values to zeros
    start = time.time()
    if model is None:
        model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    model.fit(X, y)
    stop = time.time()
    print(f"RF fit time for {n_records} = {(stop - start):.1f}s")
    start = timer()
    shap_I = shap_importances(model, X)
    stop = timer()
    if isinstance(model, RandomForestRegressor):
        print(f"SHAP OOB $R^2$ {model.oob_score_:.2f}, time for {n_records} = {(stop - start):.1f}s")
    else:
        print(f"SHAP time for {n_records} = {(stop - start):.1f}s")

    return shap_I


def our_bulldozer():
    df = pd.read_feather("../notebooks/data/bulldozer-train.feather")
    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    # basefeatures = ['ModelID', 'YearMade', 'MachineHours']
    basefeatures = ['SalesID', 'MachineID', 'ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID', 'MachineHoursCurrentMeter']

    # Get subsample; it's a (sorted) timeseries so get last records not random
    n_records = 2_000
    df = df.iloc[-n_records:]  # take only last records
    X, y = df[basefeatures], df['SalePrice']
    X = X.fillna(0)  # flip missing numeric values to zeros
    start = timer()
    I = impact_importances(X, y, verbose=True)#, n_samples=3000, bootstrap_sampling=False, n_trials=10)#, catcolnames={'ModelID','SalesID','auctioneerID','MachineID','datasource'})
    print(I)
    stop = timer()
    print(f"Time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1))


def boston():
    boston = load_boston()
    df = pd.DataFrame(normalize(boston.data), columns=boston.feature_names)
    df['MEDV'] = boston.target

    n_estimators=50
    n_records=len(df)

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    fig, axes = plt.subplots(6, 1, figsize=(5,12.5))

    lm = LinearRegression()
    I, score = linear_model_importance(lm, X_test, X_train, y_test, y_train)
    print(I)
    plot_importances(I, imp_range=(0, 1), ax=axes[0])
    axes[0].set_title(f"OLS $\\beta_i$, $R^2$ {score :.2f}")

    lm.fit(X_train, y_train)
    score = lm.score(X_test, y_test)
    print(f"SHAP OLS $R^2$ {score:.2f}")

    start = timer()
    I = shap_importances(lm, X_train, X_test)
    stop = timer()
    print(f"SHAP time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1), ax=axes[1])
    axes[1].set_title(f"SHAP OLS, $R^2$ {score :.2f}")

    alpha = .001 # found via CV=5
    lm = Lasso(alpha=alpha)
    # model = GridSearchCV(lm, cv=5,
    #                    param_grid={"alpha": [.2, .1, .05, .01, .005, .001]})
    # model.fit(X, y)
    # print(model.best_params_)
    I, score = linear_model_importance(lm, X_test, X_train, y_test, y_train)
    plot_importances(I, imp_range=(0, 1), ax=axes[2])
    axes[2].set_title(f"Lasso($\\alpha={alpha}$) $\\beta_i$, $R^2$ {score :.2f}")

    lm.fit(X_train, y_train)
    score = lm.score(X_test, y_test)
    print(f"SHAP Lasso($\\alpha={alpha}$) $R^2$ {score:.2f}")

    start = timer()
    I = shap_importances(lm, X_train, X_test)
    stop = timer()
    print(f"SHAP time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1), ax=axes[3])
    axes[3].set_title(f"SHAP Lasso($\\alpha={alpha}$), $R^2$ {score :.2f}")

    # model = svm.SVR(gamma='auto')
    # model = GridSearchCV(svm.SVR(), cv=5,
    #                    param_grid={"C": [1, 1000, 2000, 3000, 5000],
    #                                "gamma": [1e-5, 1e-4, 1e-3]})
    if False:
        model = svm.SVR(gamma='auto')
        model.fit(X_train, y_train)
        # print(model.best_params_)
        svm_score = model.score(X_test, y_test)
        print("svm_score", svm_score)

        start = timer()
        I = shap_importances(model, shap.kmeans(X_train, k=10), X_test)
        stop = timer()
        print(f"SHAP time for {n_records} = {(stop - start):.1f}s")
        plot_importances(I, imp_range=(0, 1), ax=axes[1])
        axes[1].set_title(f"SHAP SVM(gamma=auto), $R^2$ {svm_score:.2f},\n|backing data|={100}, |X_train|={len(X_train)}, |X_test|={len(X_test)}")


    #model = GradientBoostingRegressor()
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    model.fit(X_train, y_train)

    start = timer()
    I = shap_importances(model, X_train, X_test)
    stop = timer()
    print(f"SHAP OOB $R^2$ {model.oob_score_:.2f}, time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1), ax=axes[4])
    axes[4].set_title(f"SHAP RF(n_estimators={n_estimators}), OOB $R^2$ {model.oob_score_:.2f}")

    # Do ours
    I = impact_importances(X, y, verbose=True,
                           min_samples_leaf=5)
    plot_importances(I, imp_range=(0, 1), ax=axes[5])
    axes[5].set_title(f"Model-Free Impact Importance")

    plt.savefig("/Users/parrt/Desktop/shap_compare.pdf")


def get_multiple_imps(X, y, test_size=0.2, n_shap=3000, n_estimators=50, min_samples_leaf=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    lm = LinearRegression()
    lm.fit(X_train, y_train)
    ols_I, score = linear_model_importance(lm, X_test, X_train, y_test, y_train)

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    rf.fit(X_train, y_train)
    rf_I = shap_importances(rf, X_train, X_test, n=n_shap)

    ours_I = impact_importances(X, y, verbose=False, min_samples_leaf=min_samples_leaf)
    return ols_I, rf_I, ours_I


def boston_multicollinearity():
    boston = load_boston()
    df = pd.DataFrame(normalize(boston.data), columns=boston.feature_names)
    df['MEDV'] = boston.target
    n_estimators=50
    n_records=len(df)

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    rf.fit(X, y)

    DM = feature_dependence_matrix(X, rf, sort_by_dependence=True)
    viz = plot_dependence_heatmap(DM, figsize=(12, 12))
    viz.save("/Users/parrt/Desktop/codep.pdf")

def boston_pdp():
    boston = load_boston()
    df = pd.DataFrame(normalize(boston.data), columns=boston.feature_names)
    df['MEDV'] = boston.target
    n_estimators=200
    n_records=len(df)

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    plot_stratpd_gridsearch(X, y, 'B', 'MEDV')


def boston_drop_features():
    boston = load_boston()
    df = pd.DataFrame(normalize(boston.data), columns=boston.feature_names)
    df['MEDV'] = boston.target
    n_estimators=200
    n_records=len(df)

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    SHAP_RF_features = ['LSTAT','RM','CRIM','DIS','PTRATIO']
    our_features = ['LSTAT','RM','B','CRIM','DIS']
    ols_features = ['RM','LSTAT','B','PTRATIO','DIS']

    features_set = [SHAP_RF_features, our_features, ols_features]
    features_names = ['SHAP', 'our', 'ols']

    ntrials = 5
    all = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        results = []
        for name,features in zip(features_names,features_set):
            X_train_ = X_train[features]
            X_test_ = X_test[features]
            rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1, oob_score=True)
            rf.fit(X_train_, y_train)
            s = rf.score(X_test_, y_test)
            results.append(s)
            # print(f"{name:4s} top 5 OOB R^2 {rf.oob_score_:.3f}, valid R^2 {s:.3f}")
        all.append(results)

    print(f"Avg top-5 valid R^2 {np.mean(all,axis=0)}, stddev {np.std(all,axis=0)}")
    print()

    SHAP_drop = ['ZN','CHAS','NOX','RAD','AGE']
    our_drop = ['RAD','TAX','NOX','INDUS','CHAS']
    ols_drop = ['INDUS','AGE','ZN','TAX','CRIM']
    features_drop_set = [SHAP_drop, our_drop, ols_drop]

    all = []
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        results = []
        for name,features in zip(features_names,features_drop_set):
            X_train_ = X_train.drop(features,axis=1)
            X_test_ = X_test.drop(features,axis=1)
            rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1, oob_score=True)
            rf.fit(X_train_, y_train)
            s = rf.score(X_test_, y_test)
            results.append(s)
            print(f"{name:4s} drop 5 OOB R^2 {rf.oob_score_:.3f}, valid R^2 {s:.3f}")
        all.append(results)

    print(f"Avg drop-5 valid R^2 {np.mean(all,axis=0)}, stddev {np.std(all,axis=0)}")

    # X_our = X[our_features]
    # rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1, oob_score=True)
    # rf.fit(X_train, y_train)
    # s = rf.score(X_test, y_test)
    # print(f"Our top 5 OOB R^2 {rf.oob_score_:.3f}, valid R^2 {s:.3f}")
    #
    # X_ols = X[ols_features]
    # rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1, oob_score=True)
    # rf.fit(X_train, y_train)
    # s = rf.score(X_test, y_test)
    # print(f"OLS top 5 OOB R^2 {rf.oob_score_:.3f}, valid R^2 {s:.3f}")


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


def bulldozer():
    n_estimators=50
    n_records=3000

    df = pd.read_feather("../notebooks/data/bulldozer-train.feather")
    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    # basefeatures = ['ModelID', 'YearMade', 'MachineHours']
    basefeatures = ['SalesID', 'MachineID', 'ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID', 'MachineHoursCurrentMeter']

    # Get subsample; it's a (sorted) timeseries so get last records not random
    df = df.iloc[-n_records:]  # take only last records
    X, y = df[basefeatures], df['SalePrice']
    X = X.fillna(0)  # flip missing numeric values to zeros

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    fig, axes = plt.subplots(3, 1)

    model = Lasso(alpha=1e4)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"SHAP Lasso $R^2$ {score:.2f}")

    start = timer()
    I = shap_importances(model, X_train, X_test)
    stop = timer()
    print(f"SHAP time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1), ax=axes[0])
    axes[0].set_title(f"SHAP Lasso(), $R^2$ {score :.2f}")

    # model = svm.SVR(gamma='auto')
    model = GridSearchCV(svm.SVR(), cv=5,
                       param_grid={"C": [1, 1000, 2000, 3000, 5000],
                                   "gamma": [1e-5, 1e-4, 1e-3]})
    model.fit(X_train, y_train)
    print(model.best_params_)
    svm_score = model.score(X_test, y_test)
    print("svm_score", svm_score)

    start = timer()
    I = shap_importances(model, shap.kmeans(X_train, k=10), X_test)
    stop = timer()
    print(f"SHAP time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1), ax=axes[1])
    axes[1].set_title(f"SHAP SVM(gamma=auto), $R^2$ {svm_score:.2f},\n|backing data|={100}, |X_train|={len(X_train)}, |X_test|={len(X_test)}")


    #model = GradientBoostingRegressor()
    model = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    model.fit(X_train, y_train)

    start = timer()
    I = shap_importances(model, X_train, X_test)
    stop = timer()
    print(f"SHAP OOB $R^2$ {model.oob_score_:.2f}, time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1), ax=axes[2])
    axes[2].set_title(f"SHAP RF(n_estimators={n_estimators}), OOB $R^2$ {model.oob_score_:.2f}")



def speed_SHAP():
    n_estimators = 30
    print("n_estimators",n_estimators)
    for n_records in [1000, 3000, 5000, 8000]:
        time_SHAP(n_estimators, n_records)


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
    hoodfeatures = list(hoods.keys())

    df = df.sort_values(by='created')[-n:]  # get a small subsample
    df_rent = df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price',
                  'interest_level']+
                 hoodfeatures+
                 ['num_photos', 'num_desc_words', 'num_features']]
    # print(df_rent.head(3))

    X = df_rent.drop('price', axis=1)
    y = df_rent['price']
    return X, y


def rent_drop():
    n_drop = 1
    n_estimators = 50
    trials = 10
    X, y = load_rent(n=2_000)
    ols_I, rf_I, our_I = get_multiple_imps(X, y, min_samples_leaf=10)
    print("OLS\n", ols_I)
    print("RF\n",rf_I)
    print("OURS\n",our_I)

    ols_drop = ols_I.iloc[-n_drop:,0].index.values
    rf_drop = rf_I.iloc[-n_drop:,0].index.values
    our_drop = our_I.iloc[-n_drop:,0].index.values
    features_names = ['OLS', 'RF', 'OUR']
    features_set = [ols_drop, rf_drop, our_drop]

    all = []
    for i in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        results = []
        for name, features in zip(features_names, features_set):
            X_train_ = X_train.drop(features, axis=1)
            X_test_ = X_test.drop(features, axis=1)
            rf = RandomForestRegressor(n_estimators=n_estimators,
                                       min_samples_leaf=1)
            rf.fit(X_train_, y_train)
            s = rf.score(X_test_, y_test)
            results.append(s)
            # print(f"{name} valid R^2 {s:.3f}")
        all.append(results)
    print(pd.DataFrame(data=all, columns=['OLS','RF','Ours']))
    print(f"Avg drop-{n_drop} valid R^2 {np.mean(all,axis=0)}, stddev {np.std(all,axis=0)}")


def avg_model_for_top_features(name:('OLS', 'RF', 'OUR'), X_test, X_train, y_test, y_train):
    scores = []

    # OLS
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    s = lm.score(X_test, y_test)
    scores.append(s)

    # Lasso
    lm = Lasso(alpha=.1)
    lm.fit(X_train, y_train)
    s = lm.score(X_test, y_test)
    scores.append(s)

    # SVM
    model = svm.SVR(gamma=0.001, C=5000)
    # model = GridSearchCV(svm.SVR(), cv=5,
    #                    param_grid={"C": [1, 1000, 2000, 3000, 5000],
    #                                "gamma": [1e-5, 1e-4, 1e-3]})
    model.fit(X_train, y_train)
    #print("SVM best:",model.best_params_)
    s = model.score(X_test, y_test)
    scores.append(s)

    # GBM
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
    n_estimators = 40
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1)
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


def rent_top(top_range=(1, 7),
             n_estimators=40,
             trials=7,
             n=3_000,
             min_samples_leaf=10):
    n_shap = n
    X, y = load_rent(n=n)
    ols_I, rf_I, our_I = get_multiple_imps(X, y, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_shap=n_shap)
    # print("OLS\n", ols_I)
    # print("RF\n",rf_I)
    # print("OURS\n",our_I)

    top = top_range[1]

    print("OUR FEATURES", our_I.index.values)

    for top in range(top_range[0], top+1):
        ols_top = ols_I.iloc[:top, 0].index.values
        rf_top = rf_I.iloc[:top, 0].index.values
        our_top = our_I.iloc[:top, 0].index.values
        features_names = ['OLS', 'RF', 'OUR']
        features_set = [ols_top, rf_top, our_top]
        all = []
        for i in range(trials):
            # print(i, end=' ')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            results = []
            for name, features in zip(features_names, features_set):
                # print(f"Train with {features} from {name}")
                X_train_ = X_train[features]
                X_test_ = X_test[features]
                s = avg_model_for_top_features(name, X_test_, X_train_, y_test, y_train)
                results.append(s)
                # print(f"{name} valid R^2 {s:.3f}")
            all.append(results)
        print()
        # print(pd.DataFrame(data=all, columns=['OLS','RF','Ours']))
        # print()
        print("n_top, n_estimators, n, n_shap, min_samples_leaf", top, n_estimators, n, n_shap, min_samples_leaf)
        print(f"Avg top-{top} valid R^2 {np.mean(all, axis=0)}")#, stddev {np.std(all, axis=0)}")
        print


rent_top(min_samples_leaf=10)

#weather()
#poly()
# unstable_SHAP()
# poly_dupcol()
#speed_SHAP()
# bulldozer()
#boston_pdp()
#boston_drop_features()
#boston()
#boston_multicollinearity()

#our_bulldozer()


#plt.tight_layout()
#plt.subplots_adjust(top=0.85) # must be after tight_layout
plt.show()
