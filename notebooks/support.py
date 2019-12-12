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

import shap

import xgboost as xgb


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
        explainer = shap.TreeExplainer(model, data=X,feature_perturbation='interventional')
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
