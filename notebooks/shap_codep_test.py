from sklearn.linear_model import LinearRegression, Lasso
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

import shap

import time
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances

def synthetic_poly_dup_data(n):
    """
    SHAP seems to make x3, x1 very different despite same coeffs. over several runs,
    it varies a lot. e.g., i see one where x1,x2,x3 are same as they should be.
    """
    p = 3 # x1, x2, x3
    df = pd.DataFrame()
    coeff = np.array([1,1,1])
    # Add independent x variables in [0.0, 1.0)
    for i in range(p):
        df[f'x{i+1}'] = np.round(np.random.random_sample(size=n)*10,5)
    df['x3'] = df['x1'] + np.random.random_sample(size=n) # copy x1 into x3
    yintercept = 100
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    # df['y'] = 5*df['x1'] + 3*df['x2'] + 9*df['x3']
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms)
    return df, coeff, eqn+" \\,\\,where\\,\\, x_3 = x_1 + noise"


def shap_importances(model, X_train, X_test, n=20_000):
    X_train = X_train[-n:]
    X_test = X_test[-n:]
    start = time.time()
    shap_values = shap.TreeExplainer(model).shap_values(X_test)
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


def SHAP_trials():
    nplots=8
    fig, axes = plt.subplots(nplots, 2, figsize=(6, nplots+3))

    for i in range(nplots):
        # make new data set each time
        df, coeff, eqn = synthetic_poly_dup_data(1000)
        X = df.drop('y', axis=1)
        y = df['y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rf = RandomForestRegressor(n_estimators=35, oob_score=True)
        rf.fit(X_train, y_train)
        shap_I = shap_importances(rf, X_train, X_test)
        plot_importances(shap_I, ax=axes[i][0], imp_range=(0, 1))
        axes[i][0].set_title(f"RF SHAP (OOB $R^2$ {rf.oob_score_:.2f})", fontsize=9)

        drop_I = dropcol_importances(rf, X_train, y_train, X_test, y_test)
        plot_importances(drop_I, ax=axes[i][1], imp_range=(0, 1))
        axes[i][1].set_title(f"Drop-column importance", fontsize=9)

    plt.suptitle('$'+eqn+'$', y=1.0)
    plt.savefig("/Users/parrt/Desktop/shap_results.png", bbox_inches=0, dpi=200)
    plt.show()


SHAP_trials()