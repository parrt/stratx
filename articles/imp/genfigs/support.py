import numpy as np
import pandas as pd

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
from timeit import default_timer as timer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import xgboost as xgb
from sklearn import svm
from sklearn.model_selection import KFold

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


def cv_features(kf, X, y, features, metric, catcolnames=None, time_sensitive=False, kfolds=5, model='RF'):
    # if time_sensitive:
    #     n_test = int(0.20 * len(X))
    #     n_train = len(X) - n_test
    #     X_train, X_test = X[:n_train], X[n_train:]
    #     y_train, y_test = y[:n_train], y[n_train:]
    # else:
    scores = []
    n_estimators = 40  # for both SHAP and testing purposes

    if kf is not None:
        indexes = kf.split(X)
    else:
        end_training = int(.8 * len(X))
        indexes = (range(0, end_training), range(end_training,len(X)))
    for train_index, test_index in indexes:
        # for k in range(kfolds):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if model=='RF':
            m = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1, n_jobs=-1)
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        elif model == 'SVM':
            m = svm.SVR(gamma='auto')
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        elif model == 'GBM':
            m = xgb.XGBRegressor(objective='reg:squarederror',
                                 learning_rate=0.7, # default is 1
                                 max_depth=4, # default is 3
                                 n_estimators=100 # default is 100
                                 )
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        elif model == 'OLS':
            if catcolnames is not None:
                # TODO: this is broken: doesn't apply training cats to X_test
                X_train = todummies(X_train[features], features, catcolnames)
                X_test = todummies(X_test[features], features, catcolnames)
            else:
                X_train = X_train[features]
                X_test = X_test[features]

            m = LinearRegression()
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            s = metric(y_test, y_pred)
        elif model == 'Lasso':
            m = Lasso(normalize=True)
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        else:
            raise ValueError(model+" is not valid model")
        scores.append(s)

    return np.array(scores)


def todummies(X, features, catcolnames):
    df = pd.DataFrame(X, columns=features)
    converted = set()
    for cat in catcolnames:
        if cat in features:
            df[cat] = df[cat].astype('category').cat.as_ordered()
            converted.add(cat)
    if len(converted)>0:
        dummies = pd.get_dummies(df)
        X = pd.concat([df, dummies], axis=1)
        X = X.drop(converted, axis=1)
        X = X.values

    return X


def compare_top_features(X, y, top_features_range=None,
                         n_shap=300,
                         metric = mean_absolute_error,
                         model='RF',
                         imp_n_trials=1,
                         imp_pvalues_n_trials=0,
                         use_oob = False,
                         #time_sensitive=False,
                         kfolds=5,
                         n_stratpd_trees=1,
                         bootstrap=False,
                         stratpd_min_samples_leaf=10,
                         stratpd_cat_min_samples_leaf=10,
                         min_slopes_per_x=5,
                         catcolnames=set(),
                         normalize=True,
                         supervised=True,
                         include=['Spearman', 'PCA', 'OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact'],
                         drop=()):
    if use_oob and metric!=r2_score:
        #     print("Warning: use_oob can only give R^2; flipping metric to r2_score")
        metric=r2_score

    include = include.copy()
    for feature in drop:
        include.remove(feature)

    # if time_sensitive:
    #     n_test = int(0.20 * len(X))
    #     n_train = len(X) - n_test
    #     X_train, X_test = X[:n_train], X[n_train:]
    #     y_train, y_test = y[:n_train], y[n_train:]
    # else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_estimators = 40 # for both SHAP and testing purposes

    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Sanity check: R^2 OOB on {X_train.shape[0]} training records: {rf.oob_score_:.3f}, training {metric.__name__}={metric(y_train, rf.predict(X_train))}")
    print(f"testing {metric.__name__}={metric(y_test, rf.predict(X_test))}")

    all_importances = get_multiple_imps(X_train, y_train, X_test, y_test,
                                        n_stratpd_trees=n_stratpd_trees,
                                        imp_n_trials=imp_n_trials,
                                        imp_pvalues_n_trials=imp_pvalues_n_trials,
                                        bootstrap=bootstrap,
                                        stratpd_min_samples_leaf=stratpd_min_samples_leaf,
                                        stratpd_cat_min_samples_leaf=stratpd_cat_min_samples_leaf,
                                        n_estimators=n_estimators,
                                        n_shap=n_shap,
                                        catcolnames=catcolnames,
                                        min_slopes_per_x=min_slopes_per_x,
                                        supervised=supervised,
                                        include=include,
                                        normalize=normalize)\

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
    topstddevs = []
    for top in range(top_features_range[0], top_features_range[1] + 1):
        # ols_top = ols_I.iloc[:top, 0].index.values
        # shap_ols_top = shap_ols_I.iloc[:top, 0].index.values
        # rf_top = rf_I.iloc[:top, 0].index.values
        # perm_top = perm_I.iloc[:top, 0].index.values
        # our_top = our_I.iloc[:top, 0].index.values
        # features_set = [ols_top, shap_ols_top, rf_top, perm_top, our_top]
        # print(i, end=' ')
        results = []
        stddevs = []
        feature_sets = [I.iloc[:top, 0].index.values for I in all_importances.values() if I is not None]
        kf = KFold(n_splits=kfolds) if kfolds > 1 else None  # use same set of folds for all techniques
        for technique_name, features in zip(include, feature_sets):
            # print(f"Train with {features} from {technique_name}")
            # Train RF model with top-k features
            # Do 5-fold cross validation using original X, y passed in to this method
            scores = cv_features(kf, X, y, features, metric=metric, kfolds=kfolds, model=model,
                                 catcolnames=catcolnames)
            results.append(np.mean(scores))
            stddevs.append(np.std(scores))
            # print(f"{technique_name} valid R^2 {s:.3f}")
        topscores.append( results )
        topstddevs.append( stddevs )

        # avg = [f"{round(m,2):9.3f}" for m in np.mean(all, axis=0)]
        # print(f"Avg top-{top} valid {metric.__name__} {', '.join(avg)}")

    R = pd.DataFrame(data=topscores, columns=features_names)
    R.index = [f"top-{top} {'OOB' if use_oob else 'training'} {metric.__name__}" for top in range(top_features_range[0], top_features_range[1] + 1)]
    Rstddev = pd.DataFrame(data=topstddevs, columns=features_names)
    Rstddev.index = [f"top-{top} stddev" for top in range(top_features_range[0], top_features_range[1] + 1)]
    print(Rstddev)

    # unpack for users
    return (R, Rstddev, *all_importances.values())


def best_single_feature(X, y, kfolds=5, model='RF'):
    means = []
    kf = KFold(n_splits=kfolds) if kfolds>1 else None
    for colname in X.columns:
        scores = cv_features(kf, X, y, [colname],
                             # metric=mean_squared_error,
                             metric=mean_absolute_error,
                             kfolds=kfolds, model=model)
        print(colname, scores, np.mean(scores))
        means.append(np.mean(scores))
    df = pd.DataFrame()
    df['Feature'] = X.columns
    df['MAE'] = means
    df = df.sort_values(by='MAE', ascending=True)
    return df


def get_multiple_imps(X_train, y_train, X_test, y_test, n_shap=300, n_estimators=50,
                      stratpd_min_samples_leaf=10,
                      stratpd_cat_min_samples_leaf=10,
                      imp_n_trials=1,
                      imp_pvalues_n_trials=0,
                      n_stratpd_trees=1,
                      bootstrap=False,
                      catcolnames=set(),
                      min_slopes_per_x=10,
                      supervised=True,
                      include=['Spearman', 'PCA', 'OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact'],
                      normalize=True):
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

    if "RF SHAP" in include:
        # Limit to training RFs with 20,000 records as it sometimes crashes above
        X_train_ = X_train[:min(20_000,len(X_train))] # already randomly selected, just grab first part
        y_train_ = y_train[:min(20_000,len(X_train))]
        rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
        rf.fit(X_train_, y_train_)
        rf_I = shap_importances(rf, X_train_, X_test, n_shap, normalize=normalize)

    if "RF perm" in include:
        rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
        rf.fit(X_train, y_train)
        perm_I = rfpimp.importances(rf, X_test, y_test) # permutation; drop in test accuracy

    if "StratImpact" in include:
        # RF SHAP and RF perm get to look at the test data to decide which features
        # are more predictive and useful for generality's sake so it's fair to
        # let this method see all data as well
        # X_full = pd.concat([X_train, X_test], axis=0)
        # y_full = pd.concat([y_train, y_test], axis=0)
        # Actually use just X_train
        X_full = X_train
        y_full = y_train
        ours_I = importances(X_full, y_full, verbose=False,
                             min_samples_leaf=stratpd_min_samples_leaf,
                             cat_min_samples_leaf=stratpd_cat_min_samples_leaf,
                             n_trials=imp_n_trials,
                             pvalues=imp_pvalues_n_trials>0,
                             pvalues_n_trials=imp_pvalues_n_trials,
                             n_trees=n_stratpd_trees,
                             bootstrap=bootstrap,
                             catcolnames=catcolnames,
                             min_slopes_per_x=min_slopes_per_x,
                             supervised=supervised,
                             normalize=normalize)
    d = OrderedDict()
    d['Spearman'] = spear_I
    d['PCA'] = pca_I
    d['OLS'] = ols_I
    d['OLS SHAP'] = ols_shap_I
    d['RF SHAP'] = rf_I
    d["RF perm"] = perm_I
    d['StratImpact'] = ours_I
    return d


def plot_topk(R, ax=None, k=None,
              title=None,
              fontname='Arial',
              title_fontsize=11,
              label_fontsize=11,
              ticklabel_fontsize=11,
              ylabel=None,
              emphasis_color = '#A22396',
              figsize=None):
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)

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
            color = emphasis_color
            lw = 2
        else:
            color = GREY
            lw = .5
        ax.plot(feature_counts, R[technique][:k], fmt, lw=lw, label=technique,
                c=color, alpha=.9, markersize=ms, fillstyle='none')

    plt.legend(loc='upper right')  # usually it's out of the way
    ax.set_xlabel("Top $k$ most important features", fontsize=label_fontsize,
                       fontname=fontname)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize = label_fontsize, fontname=fontname)

    ax.xaxis.set_ticks(feature_counts)
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)


def stability(X, y, sample_size, n_trials, technique='StratImpact',
              catcolnames=None,
              imp_n_trials=1,
              min_slopes_per_x=5,
              n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0
              ):
    n = len(X)
    all_I = pd.DataFrame(data=X.columns, columns=['Feature'])
    all_I = all_I.set_index('Feature')
    for i in range(n_trials):
        bootstrap_sample_idxs = resample(range(n), n_samples=sample_size, replace=False)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        if technique=='StratImpact':
            I = importances(X_, y_,
                            catcolnames=catcolnames,
                            n_trials=imp_n_trials,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf,
                            min_slopes_per_x=min_slopes_per_x,
                            n_trees=n_trees,
                            bootstrap=bootstrap)
        elif technique=='RFSHAP':
            print("RFSHAP",i)
            rf = RandomForestRegressor(n_estimators=40)
            rf.fit(X_, y_)
            I = shap_importances(rf, X_, X_, n_shap=300)
        else:
            raise ValueError("bad mode: "+model)
        print(I.iloc[:8])
        all_I[i] = I['Importance']
        # print(all_I)
    I = pd.DataFrame(data={'Feature': X.columns,
                           'Importance': np.mean(all_I, axis=1),
                           'Sigma': np.std(all_I, axis=1)})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    I.reset_index().to_feather("/tmp/t.feather")
    return I


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

    # times are in 830 to mean 08:30, convert to two columns, hour and min
    def cvt_time(df, colname):
        df[f'{colname}_HOUR'] = df[colname] / 100
        df[f'{colname}_HOUR'] = df[f'{colname}_HOUR'].astype(int)
        df[f'{colname}_MIN']  = df[colname] - df[f'{colname}_HOUR'] * 100
        df[f'{colname}_MIN']  = df[f'{colname}_MIN'].astype(int)

    # cvt_time(df_flights, 'SCHEDULED_DEPARTURE')
    # cvt_time(df_flights, 'SCHEDULED_ARRIVAL')
    # cvt_time(df_flights, 'DEPARTURE_TIME')

    features = [#'YEAR',  # drop year as it's a constant
                'MONTH', 'DAY', 'DAY_OF_WEEK', 'dayofyear',
                'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                'SCHEDULED_DEPARTURE',
                # 'SCHEDULED_DEPARTURE_HOUR', 'SCHEDULED_DEPARTURE_MIN',
                'SCHEDULED_ARRIVAL',
                # 'SCHEDULED_ARRIVAL_HOUR',   'SCHEDULED_ARRIVAL_MIN',
                'DEPARTURE_TIME',
                # 'DEPARTURE_TIME_HOUR',      'DEPARTURE_TIME_MIN',
                'FLIGHT_NUMBER', 'TAIL_NUMBER',
                'AIR_TIME', 'DISTANCE',
                'TAXI_IN', 'TAXI_OUT',
                'SCHEDULED_TIME',
                'ARRIVAL_DELAY']  # target

    df_flights = df_flights[features]
    df_flights = df_flights.dropna()  # ignore missing stuff for ease and reduce size
    df_flights = df_flights.sample(n)
    df_string_to_cat(df_flights)
    df_cat_to_catcode(df_flights)

    X, y = df_flights.drop('ARRIVAL_DELAY', axis=1), df_flights['ARRIVAL_DELAY']

    return X, y, df_flights


def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n // 2 # 50/50 men/women
    nwomen = n // 2
    # nmen = int(.7 * n)
    # nwomen = int(.3 * n)
    df['sex'] = ['M'] * nmen + ['F'] * nwomen
    # df.loc[df['sex'] == 'F', 'pregnant'] = np.random.randint(0, 2, size=(nwomen,))
    df.loc[df['sex'] == 'F', 'pregnant'] = 1 # assume all women are pregnant
    df.loc[df['sex'] == 'M', 'pregnant'] = 0
    df.loc[df['sex'] == 'M', 'height'] = 5 * 12 + 8 + np.random.uniform(-7, +8,
                                                                        size=(nmen,))
    df.loc[df['sex'] == 'F', 'height'] = 5 * 12 + 5 + np.random.uniform(-4.5, +5,
                                                                        size=(nwomen,))
    df.loc[df['sex'] == 'M', 'education'] = 10 + np.random.randint(0, 8, size=nmen)
    df.loc[df['sex'] == 'F', 'education'] = 12 + np.random.randint(0, 8, size=nwomen)
    df['weight'] = 120 \
                   + (df['height'] - df['height'].min()) * 10 \
                   + df['pregnant'] * 70 \
                   - df['education'] * 1.5
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    eqn = "y = 120 + 10(x_{height} - min(x_{height})) + 30x_{pregnant} - 1.5x_{education}"

    df['pregnant'] = df['pregnant'].astype(int)
    df['sex'] = df['sex'].map({'M': 0, 'F': 1}).astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    return X, y, df, eqn


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

    # df.loc[df.YearMade < 1950, 'YearMade'] = np.nan
    # fix_missing_num(df, 'YearMade')
    df = df.loc[df.YearMade > 1950].copy()
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
           [
            'age',
            'AC',
            'ProductSize',
            'MachineHours_na',
            'saleyear', 'salemonth', 'saleday', 'saledayofweek', 'saledayofyear']
           ]

    X = X.fillna(0)  # flip missing numeric values to zeros
    y = df['SalePrice']
    return X, y


def load_rent(n:int=None, clean_prices=True):
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

    # The numeric stratpd can't extract data too well when so many data points sit
    # on same values; flip it to integers from flops like 1.5 baths; can consider
    # categorical nominal or as ordinal but it stratpd ignores lots of data as ordinal
    # so best to use catstratpd
    uniq_b = np.unique(df['bathrooms'])
    df['bathrooms'] = df['bathrooms'].map({v: i + 1 for i, v in enumerate(uniq_b)})

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

    if n is not None:
        howmany = min(n, len(df))
        df = df.sort_values(by='created').sample(howmany, replace=False)
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
