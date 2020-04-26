"""
MIT License

Copyright (c) 2019 Terence Parr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy.stats import spearmanr

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import xgboost as xgb

from timeit import default_timer as timer
from collections import OrderedDict
import os

import rfpimp

import shap

import stratx.featimp as featimp #import plot_importances, importances, friedman_partial_dependences

# THIS FILE IS INTENDED FOR USE BY PARRT TO TEST / GENERATE SAMPLE IMAGES

datadir = "/Users/parrt/data"

def set_data_dir(dir):
    global datadir
    datadir = dir

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)

# Didn't allow super huge forests in RF and GBM as generating top-k images takes many hours
# Very small impact on scores.

models = {
    ("boston", "RF"):{'max_features': 0.5, 'min_samples_leaf': 1, 'n_estimators': 125},
    ("boston", "GBM"):{'learning_rate': 0.08, 'max_depth': 5, 'n_estimators': 125},
    ("boston", "SVM"):{'C': 5000, 'gamma': 0.01, 'kernel': 'rbf'},
    ("flights", "RF"):{'max_features': 0.9, 'min_samples_leaf': 1, 'n_estimators': 150},
    ("flights", "GBM"):{'learning_rate': 0.15, 'max_depth': 5, 'n_estimators': 300},
    ("bulldozer", "RF"):{'max_features': 0.9, 'min_samples_leaf': 1, 'n_estimators': 150},
    ("bulldozer", "GBM"):{'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 300},
    ("rent", "RF"):{'max_features': 0.3, 'min_samples_leaf': 1, 'n_estimators': 150},
    ("rent", "GBM"):{'learning_rate': 0.15, 'max_depth': 8, 'n_estimators': 300},
}

valscores = {
    ("boston", "RF"):0.856519266005153,
    ("boston", "GBM"):0.868056221039585,
    ("boston", "SVM"):0.8343629038412331,
    ("flights", "RF"):0.7028457737934841,
    ("flights", "GBM"):0.8126057415992978,
    ("bulldozer", "RF"):0.8420184784544134,
    ("bulldozer", "GBM"):0.8736748179201128,
    ("rent", "RF"):0.8382309546554223,
    ("rent", "GBM"):0.8442431273765862,
}

trnscores = {
    ("boston", "RF"):0.9825246077935933,
    ("boston", "GBM"):0.994336518612878,
    ("boston", "SVM"):0.9505130826031901,
    ("flights", "RF"):0.955548646849625,
    ("flights", "GBM"):0.9881243175358493,
    ("bulldozer", "RF"):0.9795046871298032,
    ("bulldozer", "GBM"):0.9575068348928771,
    ("rent", "RF"):0.9785130266916385,
    ("rent", "GBM"):0.9724591756843246,
}

pairs = [
    ("boston", "RF"),
    ("boston", "SVM"),
    ("boston", "GBM"),
    ("flights", "RF"),
    ("flights", "GBM"),
    ("bulldozer", "RF"),
    ("bulldozer", "GBM"),
    ("rent", "RF"),
    ("rent", "GBM")
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
    score = model.score(X, y)

    imp = np.abs(model.coef_)

    # Without dividing by stderr, OLS mirrors OLS SHAP for most part

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
    # only use n_shap from X_test
    X_test = X_test.sample(n=min(n_shap, len(X_test)), replace=False)
    if isinstance(model, RandomForestRegressor) or \
        isinstance(model, GradientBoostingRegressor) or \
        isinstance(model, xgb.XGBRegressor):
        """
        We get this warning for big X_train so choose smaller
        'Passing 20000 background samples may lead to slow runtimes. Consider using shap.sample(data, 100) to create a smaller background data set.'
        """
        explainer = shap.TreeExplainer(model,
                                       data=shap.sample(X_train, 100),
                                       feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    elif isinstance(model, Lasso) or isinstance(model, LinearRegression):
        explainer = shap.LinearExplainer(model,
                                         shap.sample(X_train, 100),
                                         feature_perturbation='interventional')
        shap_values = explainer.shap_values(X_test)
    else:
        # gotta use really small sample; verrry slow
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
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


def get_shap(rf,to_explain,backing=None,assume_independence=True):
    if assume_independence:
        explainer = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
    else:
        explainer = shap.TreeExplainer(rf, data=backing, feature_perturbation='interventional')
    shap_values = explainer.shap_values(to_explain, check_additivity=False)
    shapimp = np.mean(np.abs(shap_values), axis=0)
    total_imp = np.sum(shapimp)
    normalized_shap = shapimp / total_imp
    I = pd.DataFrame(data={'Feature': to_explain.columns, 'Importance': normalized_shap})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


def cv_features(dataset, kfold_indexes, X, y, features, metric, model):
    scores = []

    if kfold_indexes is None:
        end_training = int(.8 * len(X))
        kfold_indexes = [(range(0, end_training), range(end_training,len(X)))]
    for train_index, test_index in kfold_indexes:
        # for k in range(kfolds):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if model != 'OLS':
            tuned_params = models[(dataset, model)]
        if model=='RF':
            m = RandomForestRegressor(**tuned_params, n_jobs=-1)
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        elif model == 'SVM':
            m = svm.SVR(**tuned_params)
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        elif model == 'GBM':
            m = xgb.XGBRegressor(**tuned_params, n_jobs=-1)
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        elif model == 'OLS':
            # no need to normalize for prediction purposes
            m = LinearRegression()
            m.fit(X_train[features], y_train)
            y_pred = m.predict(X_test[features])
            s = metric(y_test, y_pred)
        # elif model == 'Lasso':
        #     m = Lasso(normalize=True)
        #     m.fit(X_train[features], y_train)
        #     y_pred = m.predict(X_test[features])
        #     s = metric(y_test, y_pred)
        else:
            raise ValueError(model+" is not valid model")
        scores.append(s)

    return np.array(scores)


def validate_features(dataset,
                      X_train, y_train, X_test, y_test,
                      features, metric, model):
    scores = []
    if model != 'OLS':
        tuned_params = models[(dataset, model)]
    if model=='RF':
        m = RandomForestRegressor(**tuned_params, n_jobs=-1)
        m.fit(X_train[features], y_train)
        y_pred = m.predict(X_test[features])
        s = metric(y_test, y_pred)
    elif model == 'SVM':
        m = svm.SVR(**tuned_params)
        m.fit(X_train[features], y_train)
        y_pred = m.predict(X_test[features])
        s = metric(y_test, y_pred)
    elif model == 'GBM':
        m = xgb.XGBRegressor(**tuned_params, n_jobs=-1)
        m.fit(X_train[features], y_train)
        y_pred = m.predict(X_test[features])
        s = metric(y_test, y_pred)
    elif model == 'OLS':
        # no need to normalize for prediction purposes
        m = LinearRegression()
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


def test_top_features(dataset,
                      X, y,
                      X_train, y_train, X_test, y_test,
                      all_importances,
                      top_features_range=None,
                      metric=mean_absolute_error,
                      model='RF'):
    # Compute k-k curves for all techniques, including both import and impact
    if top_features_range is None:
        top_features_range = (1, X.shape[1])

    technique_names = ['Spearman', 'PCA', 'OLS', 'OLS SHAP', 'RF SHAP', "RF perm",
                       'StratImport', 'StratImpact']

    if dataset=='bulldozer':
        technique_names.remove('OLS')
        technique_names.remove('OLS SHAP')

    print(f"test_top_features {dataset} using {technique_names} and {model}")

    feature_sets = []
    for name in technique_names:
        I = all_importances[name]
        if I is not None:
            top_features = list(I.index.values)
            print(f"Top {name} features:", top_features[:8])
            feature_sets.append(top_features)

    topscores = []
    topstddevs = []
    for k in range(top_features_range[0], top_features_range[1] + 1):
        results = []
        stddevs = []
        for technique_name, features in zip(technique_names, feature_sets):
            # print(f"Train with {features} from {technique_name}")
            # Train model with top-k features
            scores = validate_features(dataset,
                                       X_train, y_train, X_test, y_test,
                                       features[:k], metric=metric, model=model)
            results.append(np.mean(scores))
            stddevs.append(np.std(scores))
            # print(f"{technique_name} valid R^2 {s:.3f}")
        topscores.append( results )
        topstddevs.append( stddevs )

        # avg = [f"{round(m,2):9.3f}" for m in np.mean(all, axis=0)]
        # print(f"Avg k-{k} valid {metric.__name__} {', '.join(avg)}")

    R = pd.DataFrame(data=topscores, columns=technique_names)
    R.index = [f"k-{top}" for top in range(top_features_range[0], top_features_range[1] + 1)]
    return R


def gen_topk_figs(n_trials,dataset,targetname,title,yunits,catcolnames=set(),
                  drop_high_variance_features=True,
                  yrange=None,figsize=(3.5, 3.0),
                  min_slopes_per_x=5,
                  cat_min_samples_leaf=5,
                  min_samples_leaf=15):
    test_size = .2  # Some techniques use validation set to pick best features
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # use same set of folds for all techniques
    # kf = KFold(n_splits=kfolds, shuffle=True)
    # kfold_indexes = list(kf.split(X))

    # get all importances

    # df_train = pd.read_csv(f'{datadir}/{dataset}-train.csv')
    # df_test = pd.read_csv(f'{datadir}/{dataset}-test.csv')
    #
    # X_train = df_train.drop(targetname, axis=1)
    # X_test = df_test.drop(targetname, axis=1)
    # y_train = df_train[targetname]
    # y_test = df_test[targetname]
    #
    # X = pd.concat([X_train, X_test], axis=0)
    # y = pd.concat([y_train, y_test], axis=0)
    #
    X, y, X_train, X_test, y_train, y_test = load_dataset(dataset, targetname)

    imps = get_multiple_imps(dataset,
                             X, y,
                             X_train, y_train, X_test, y_test,
                             drop_high_variance_features=drop_high_variance_features,
                             normalize=True,
                             catcolnames=catcolnames,
                             n_shap=300,
                             min_slopes_per_x=min_slopes_per_x,
                             stratpd_cat_min_samples_leaf=cat_min_samples_leaf,
                             stratpd_min_samples_leaf=min_samples_leaf,
                             imp_n_trials=n_trials,
                             )

    w = 4.5 if dataset == 'flights' else 3
    featimp.plot_importances(imps['Strat'].iloc[:8], imp_range=(0, 0.4), width=w,
                     title=f"{dataset} StratImpact importances")
    plt.tight_layout()
    plt.savefig(f"../images/{dataset}-features.pdf")
    # plt.show()
    plt.close()

    featimp.plot_importances(imps['RF SHAP'].iloc[:8], imp_range=(0, 0.4), width=w,
                     title=f"{dataset} SHAP RF importances")
    plt.tight_layout()
    plt.savefig(f"../images/{dataset}-features-shap-rf.pdf")
    # plt.show()
    plt.close()

    model = "RF"
    topk_for_one_model(dataset, model, X, y,
                       X_train, y_train, X_test, y_test,
                       imps,
                       figsize,
                       title,
                       yrange,
                       yunits)

    model = "GBM"
    topk_for_one_model(dataset, model, X, y,
                       X_train, y_train, X_test, y_test,
                       imps,
                       figsize,
                       title,
                       yrange,
                       yunits)

    # Do OLS special cases for rent and boston

    main_techniques = ['OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImport']
    if dataset!='bulldozer': # purely numerical features or only important features are numerical
        R = test_top_features(dataset,
                              X, y,
                              X_train, y_train, X_test, y_test,
                              imps,
                              top_features_range=(1,8),
                              metric=mean_absolute_error,
                              model="OLS")
        if dataset=='rent':
            yrange=(500,1200)
        elif dataset=='boston':
            yrange=(2,6.5)
        else:
            yrange=(15,30)

        R_ = R[main_techniques]
        plot_topk(R_, k=8, title=f"OLS {title}",
                  ylabel=f"5-fold CV MAE ({yunits})",
                  xlabel=f"Top $k$ feature $Importance$",
                  title_fontsize=14,
                  label_fontsize=14,
                  ticklabel_fontsize=10,
                  # legend_location='lower left',
                  legend_location='upper right',
                  yrange=yrange,
                  figsize=figsize)
        plt.tight_layout()
        plt.savefig(f"../images/{dataset}-topk-OLS-Importance.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()


def topk_for_one_model(dataset, model,
                       X, y,
                       X_train, y_train, X_test, y_test,
                       imps, figsize, title,
                       yrange, yunits):
    # GET ALL TOP-K CURVES
    R = test_top_features(dataset,
                          X, y,
                          X_train, y_train, X_test, y_test,
                          imps,
                          top_features_range=(1, 8),
                          metric=mean_absolute_error,
                          model=model)
    print(f"TOP-k {model} CURVES\n",R)

    # OK, so now we have curves for {model} with importance, impact from StratImpact; save these

    def plotimp(sortby, techniques):
        plot_topk(R[techniques], k=8, title=f"{model} {title}",
                  ylabel=f"5-fold CV MAE ({yunits})",
                  xlabel=f"Top $k$ feature ${sortby}$",
                  title_fontsize=14,
                  label_fontsize=14,
                  ticklabel_fontsize=10,
                  yrange=yrange,
                  figsize=figsize)
        plt.tight_layout()
        plt.savefig(f"../images/{dataset}-topk-{model}-{sortby}.pdf", bbox_inches="tight",
                    pad_inches=0)
        plt.show()

    if dataset=='bulldozer':
        plotimp(sortby='Importance', techniques=['RF SHAP', "RF perm", 'StratImport'])
        plotimp(sortby='Impact',     techniques=['RF SHAP', "RF perm", 'StratImpact'])
    else:
        plotimp(sortby='Importance', techniques=['OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImport'])
        plotimp(sortby='Impact',     techniques=['OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact'])
    plotimp(sortby='baseline',   techniques=['Spearman', 'PCA', 'StratImport'])


def best_single_feature(X, y, dataset, kfolds=5, model='RF'):
    means = []
    kf = KFold(n_splits=kfolds) if kfolds>1 else None
    kfold_indexes = kf.split(X)
    for colname in X.columns:
        scores = cv_features(dataset,
                             kfold_indexes, X, y, [colname],
                             # metric=mean_squared_error,
                             metric=mean_absolute_error,
                             model=model)
        print(colname, scores, np.mean(scores))
        means.append(np.mean(scores))
    df = pd.DataFrame()
    df['Feature'] = X.columns
    df['MAE'] = means
    df = df.sort_values(by='MAE', ascending=True)
    return df


def get_multiple_imps(dataset,
                      X, y,
                      X_train, y_train, X_test, y_test,
                      n_shap=300,
                      drop_high_variance_features=True,
                      sortby='Importance',
                      stratpd_min_samples_leaf=15,
                      stratpd_cat_min_samples_leaf=5,
                      imp_n_trials=1,
                      imp_pvalues_n_trials=0,
                      n_stratpd_trees=1,
                      rf_bootstrap=False,
                      bootstrap=True,
                      catcolnames=set(),
                      min_slopes_per_x=5,
                      supervised=True,
                      # include=['Spearman', 'PCA', 'OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact'],
                      normalize=True):
    spear_I = pca_I = ols_I = ols_shap_I = rf_I = perm_I = ours_I = None

    # Do everything now
    include = ['Spearman', 'PCA', 'OLS', 'OLS SHAP', 'RF SHAP', "RF perm", 'StratImpact']
    # include = ['StratImpact']

    if dataset=='bulldozer':
        include.remove('OLS')
        include.remove('OLS SHAP')

    if 'Spearman' in include:
        spear_I = spearmans_importances(X, y)

    if 'PCA' in include:
        pca_I = pca_importances(X)

    if "OLS" in include:
        # since we use coefficients, look at all data
        X_ = StandardScaler().fit_transform(X)
        X_ = pd.DataFrame(X_, columns=X.columns)
        lm = LinearRegression()
        lm.fit(X_, y)
        ols_I, score = linear_model_importance(lm, X_, y)
        print("OLS\n",ols_I)

    if "OLS SHAP" in include:
        # since we use coefficients, look at all data, explain n_shap
        X_ = StandardScaler().fit_transform(X)
        X_ = pd.DataFrame(X_, columns=X.columns)
        lm = LinearRegression()
        lm.fit(X_, y)
        ols_shap_I = shap_importances(lm, X_, X_, n_shap=n_shap)

    if "RF SHAP" in include:
        tuned_params = models[(dataset, "RF")]
        rf = RandomForestRegressor(**tuned_params, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_I = shap_importances(rf, X_train, X_test, n_shap, normalize=normalize)
        print("RF SHAP\n",rf_I)

    if "RF perm" in include:
        tuned_params = models[(dataset, "RF")]
        rf = RandomForestRegressor(**tuned_params, n_jobs=-1)
        rf.fit(X_train, y_train)
        perm_I = rfpimp.importances(rf, X_test, y_test) # permutation; drop in test accuracy
        print("RF perm\n",perm_I)

    if "StratImpact" in include:
        # RF SHAP and RF perm get to look at the test data to decide which features
        # are more predictive and useful for generality's sake
        # So, we get to look at all data as well, not just training data
        ours_I = featimp.importances(X, y, #X_train, y_train,
                                     verbose=False,
                                     sortby=sortby,
                                     min_samples_leaf=stratpd_min_samples_leaf,
                                     cat_min_samples_leaf=stratpd_cat_min_samples_leaf,
                                     n_trials=imp_n_trials,
                                     pvalues=imp_pvalues_n_trials > 0,
                                     pvalues_n_trials=imp_pvalues_n_trials,
                                     n_trees=n_stratpd_trees,
                                     bootstrap=bootstrap,
                                     rf_bootstrap=rf_bootstrap,
                                     catcolnames=catcolnames,
                                     min_slopes_per_x=min_slopes_per_x,
                                     supervised=supervised,
                                     normalize=normalize,
                                     drop_high_stddev=2.0 if drop_high_variance_features else 9999)
        print("OURS\n",ours_I)

    if "PDP" in include:
        tuned_params = models[(dataset, "RF")]
        rf = RandomForestRegressor(**tuned_params, n_jobs=-1)
        rf.fit(X, y)
        pdpy = featimp.friedman_partial_dependences(rf, X, mean_centered=True)
        pdp_I = pd.DataFrame(data={'Feature': X.columns})
        pdp_I = pdp_I.set_index('Feature')
        pdp_I['Importance'] = np.mean(np.mean(np.abs(pdpy)), axis=1)

    d = OrderedDict()
    d['Spearman'] = spear_I
    d['PCA'] = pca_I
    d['OLS'] = ols_I
    d['OLS SHAP'] = ols_shap_I
    d['RF SHAP'] = rf_I
    d["RF perm"] = perm_I
    d['Strat'] = ours_I

    # Put both orders for Strat approach into same imps dictionary
    I = featimp.Isortby(ours_I, 'Importance')
    d['StratImport'] = pd.DataFrame(I['Importance'])

    I = featimp.Isortby(ours_I, 'Impact')
    d['StratImpact'] = pd.DataFrame(I['Impact'])

    print(d['StratImport'])
    print(d['StratImpact'])
    return d


def plot_topk(R, ax=None, k=None,
              title=None,
              fontname='Arial',
              title_fontsize=11,
              label_fontsize=11,
              ticklabel_fontsize=11,
              ylabel=None,
              xlabel=None,
              yrange=None,
              legend_location='upper right',
              emphasis_color='#A22396',
              figsize=None):
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)

    GREY = '#444443'
    if k is None:
        k = R.shape[0]
    feature_counts = range(1, k + 1)
    fmts = {'Spearman':'P-', 'PCA':'D-',
            'OLS':'o-', 'OLS SHAP':'v-', 'RF SHAP':'s-',
            "RF perm":'x-', 'StratImport':'-', 'StratImpact':'-'}
    for i,technique in enumerate(R.columns):
        fmt = fmts[technique]
        ms = 8
        if fmt == 'x-': ms = 11
        if fmt == 'P-': ms = 11
        if technique in ('StratImport','StratImpact'):
            color = emphasis_color
            lw = 2
        else:
            color = GREY
            lw = .5
        ax.plot(feature_counts, R[technique][:k], fmt, lw=lw, label=technique,
                c=color, alpha=.9, markersize=ms, fillstyle='none')

    plt.legend(loc=legend_location)  # usually it's out of the way

    if xlabel is None:
        ax.set_xlabel("Top $k$ most important features", fontsize=label_fontsize,
                           fontname=fontname)
    else:
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontname=fontname)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize = label_fontsize, fontname=fontname)

    ax.xaxis.set_ticks(feature_counts)
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)

    if yrange is not None:
        ax.set_ylim(*yrange)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)


def load_flights(n):
    global datadir
    if not os.path.exists(datadir):
        datadir = "data"

    msg = """Download from https://www.kaggle.com/usdot/flight-delays/download and save
    flight-delays.zip; unzip to convenient data dir.
    """
    dir = f"{datadir}/flight-delays"
    if os.path.exists(dir+"/flights.feather"):
        df_flights = pd.read_feather(dir + "/flights.feather")
    elif not os.path.exists(f"{dir}/flights.csv"):
        raise ValueError(msg)
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

    print(f"Flight has {len(df_flights)} records")

    df_flights = df_flights[features]
    df_flights = df_flights.dropna()  # ignore missing stuff for ease and reduce size
    df_flights = df_flights.sample(n)
    df_string_to_cat(df_flights)
    df_cat_to_catcode(df_flights)

    X, y = df_flights.drop('ARRIVAL_DELAY', axis=1), df_flights['ARRIVAL_DELAY']

    return X, y, df_flights


def synthetic_interaction_data(n, yintercept = 10):
    df = pd.DataFrame()
    df[f'x1'] = np.random.random(size=n)*10
    df[f'x2'] = np.random.random(size=n)*10
    df[f'x3'] = np.random.random(size=n)*10
    df['y'] = df['x1']**2 + df['x1']*df['x2'] + 5*df['x1']*np.sin(3*df['x2'])  + yintercept
    return df


def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n // 2 # 50/50 men/women
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
                   + df['pregnant'] * 40 \
                   - df['education'] * 1.5
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    eqn = "y = 120 + 10(x_{height} - min(x_{height})) + 30x_{pregnant} - 1.5x_{education}"

    df['pregnant'] = df['pregnant'].astype(int)
    df['sex'] = df['sex'].map({'M': 0, 'F': 1}).astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    return X, y, df, eqn


def load_dataset(dataset, targetname):
    df_train = pd.read_csv(f'{datadir}/{dataset}-train.csv')
    df_test = pd.read_csv(f'{datadir}/{dataset}-test.csv')

    X_train = df_train.drop(targetname, axis=1)
    X_test = df_test.drop(targetname, axis=1)
    y_train = df_train[targetname]
    y_test = df_test[targetname]

    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    return X, y, X_train, X_test, y_train, y_test


def load_bulldozer(n):
    global datadir
    if not os.path.exists(datadir):
        datadir = "data"

    msg = "Download Train.csv data from https://www.kaggle.com/c/bluebook-for-bulldozers/data and save in data subdir"
    if os.path.exists(f"{datadir}/bulldozer-train-all.feather"):
        print("Loading cached version...")
        df = pd.read_feather(f"{datadir}/bulldozer-train-all.feather")
    elif not os.path.exists(f"{datadir}/Train.csv"):
        raise ValueError(msg)
    else:
        dtypes = {col: str for col in
                  ['fiModelSeries', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow']}
        df = pd.read_csv(f'{datadir}/Train.csv', dtype=dtypes, parse_dates=['saledate'])  # 35s load
        df = df.sort_values('saledate')
        df = df.reset_index(drop=True)
        df.to_feather(f"{datadir}/bulldozer-train-all.feather")

    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    df.loc[df.eval("MachineHours==0"), 'MachineHours'] = np.nan
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

    features = ['ModelID',
                'datasource', 'YearMade',
                # some missing values but use anyway:
                'auctioneerID',
                'MachineHours',
                'age',
                'AC',
                'ProductSize',
                'saleyear', 'salemonth', 'saleday', 'saledayofweek', 'saledayofyear'
                ]

    X = df[features]
    X = X.fillna(0)  # flip missing numeric values to zeros
    y = df['SalePrice']

    # Most recent timeseries data is more relevant so get big recent chunk
    # then we can sample from that to get n
    X = X.iloc[-50_000:]
    y = y.iloc[-50_000:]

    print(f"Bulldozer has {len(df)} records")

    idxs = resample(range(50_000), n_samples=n, replace=False, )
    X, y = X.iloc[idxs], y.iloc[idxs]

    return X, y


def load_rent(n:int=None, clean_prices=True):
    global datadir
    if not os.path.exists(datadir):
        datadir = "data"
    msg = """Download train.json from https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data
    and save into data subdir."""
    if not os.path.exists(f"{datadir}/train.json"):
        raise ValueError(msg)

    df = pd.read_json(f'{datadir}/train.json')
    print(f"Rent has {len(df)} records")

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


def tune_RF(X, y, verbose=0):
    tuning_parameters = {'n_estimators': [30, 40, 50, 80, 125, 150],
                        'min_samples_leaf': [1, 3, 5, 7],
                        'max_features': [.3, .5, .7, .9]}
    grid = GridSearchCV(
        RandomForestRegressor(),
        tuning_parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=verbose
    )
    grid.fit(X, y)  # does CV on entire data set
    rf = grid.best_estimator_
    print("RF best:", grid.best_params_)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # rf.fit(X_train, y_train)
    # print("validation R^2", rf.score(X_test, y_test))
    return rf, grid.best_params_, grid.best_score_


def tune_XGBoost(X, y, verbose=0):
    # for these data sets we don't get much boost using many more trees
    tuning_parameters = {'n_estimators': [50, 100, 125, 150, 200, 300], #[300, 400, 450, 500, 600, 1000],
                        'learning_rate': [0.05, 0.08, 0.1, 0.15, 0.2],
                        'max_depth': [3, 4, 5, 6, 7, 8]}
    grid = GridSearchCV(
        xgb.XGBRegressor(),
        tuning_parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=verbose
    )
    grid.fit(X, y)  # does CV on entire data set to tune
    print("XGB best:", grid.best_params_)
    b = grid.best_estimator_

    return b, grid.best_params_, grid.best_score_


def tune_SVM(X, y, verbose=0):
    X_ = StandardScaler().fit_transform(X)
    tuning_parameters = {"kernel": ['rbf'],  # 'linear','poly' are too slow. ugh
          "gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
          "C": [1, 10, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 6000]}
    grid = GridSearchCV(
        svm.SVR(),
        param_grid=tuning_parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        refit=True,
        #return_train_score=True,
        verbose=verbose
    )
    grid.fit(X_, y)
    print(grid.best_params_)
    s = grid.best_estimator_
    print("SVM best:", grid.best_params_)
    # print("Scores:")
    # print(pd.DataFrame.from_dict(grid.cv_results_))
    return s, grid.best_params_, grid.best_score_


def tune_all(pairs_to_tune=list(models.keys()), verbose=1):
    "Find hyper parameters for all models / datasets using training data only"
    data = {}

    X, y, X_train, X_test, y_train, y_test = load_dataset("boston", "MEDV")
    data['boston'] = (X_train,y_train)

    X, y, X_train, X_test, y_train, y_test = load_dataset("flights", "ARRIVAL_DELAY")
    data['flights'] = (X_train,y_train)

    X, y, X_train, X_test, y_train, y_test = load_dataset("bulldozer", "SalePrice")
    data['bulldozer'] = (X_train,y_train)

    X, y, X_train, X_test, y_train, y_test = load_dataset("rent", "price")
    data['rent'] = (X_train,y_train)

    for dataset, modelname in pairs_to_tune:
        print(dataset, modelname)
        X, y = data[dataset]
        if modelname=='RF':
            m, bestparams, bestscore = tune_RF(X, y, verbose=verbose)
        elif modelname == 'GBM':
            m, bestparams, bestscore = tune_XGBoost(X, y, verbose=verbose)
        elif modelname=='SVM':
            X = StandardScaler().fit_transform(X)
            m, bestparams, bestscore = tune_SVM(X, y, verbose=verbose)
        else:
            raise ValueError(f"Invalid modelname {modelname}")
        models[dataset, modelname] = bestparams
        valscores[dataset, modelname] = bestscore
        m.fit(X,y) # already fit
        trnscores[dataset, modelname] = m.score(X, y)

    # Generate Python code

    print("models = {")
    for dataset, modelname in models.keys():
        print(f'    ("{dataset}", "{modelname}"):{models[(dataset, modelname)]},')
    print("}\n")
    print("valscores = {")
    for dataset, modelname in models.keys():
        print(f'    ("{dataset}", "{modelname}"):{valscores[(dataset, modelname)]},')
    print("}\n")
    print("trnscores = {")
    for dataset, modelname in models.keys():
        print(f'    ("{dataset}", "{modelname}"):{trnscores[(dataset, modelname)]},')
    print("}")


if __name__ == '__main__':
    # foo = [
    #     ("boston", "GBM"),
    #     ("flights", "GBM"),
    #     ("bulldozer", "GBM"),
    #     ("rent", "GBM")
    # ]
    #
    tune_all(pairs_to_tune=pairs, verbose=1)

