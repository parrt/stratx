# prepared by Terence Parr for consumption by SHAP authors

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer

import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

def synthetic_poly_dup_data(n):
    p = 3 # x1, x2, x3
    df = pd.DataFrame()
    coeff = np.array([1,1,1])
    for i in range(p):
        df[f'x{i + 1}'] = np.random.random_sample(size=n) * 10
    df['x3'] = df['x1'] + np.random.random_sample(size=n) # copy x1 into x3 with noise
    yintercept = 100
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " \\,\\,where\\,\\, x_3 = x_1 + noise"
    return df, coeff, eqn


def shap_importances(model, X, normalize=True):
    start = timer()
    explainer = shap.TreeExplainer(model, data=X, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X, check_additivity=True)
    stop = timer()
    print(f"SHAP time for {len(X)} records = {(stop - start):.1f}s")

    shapimp = np.mean(np.abs(shap_values), axis=0)
    total_imp = np.sum(shapimp)
    normalized_shap = shapimp
    if normalize:
        normalized_shap = shapimp / total_imp

    shapI = pd.DataFrame(data={'Feature': X.columns, 'Importance': normalized_shap})
    shapI = shapI.set_index('Feature')
    shapI = shapI.sort_values('Importance', ascending=False)
    return shapI


def SHAP_trials():
    print("SHAP version", shap.__version__)
    ntrials=8
    fig, axes = plt.subplots(ntrials, 3, figsize=(8, ntrials+3))

    for i in range(ntrials):
        # print(i, end=' ')
        # make new data set each trial
        df, coeff, eqn = synthetic_poly_dup_data(1000)
        X = df.drop('y', axis=1)
        y = df['y']

        rf = RandomForestRegressor(n_estimators=30, oob_score=True)
        rf.fit(X, y)
        shap_I = shap_importances(rf, X)
        plot_importances(shap_I, ax=axes[i][0], imp_range=(0, 1))
        axes[i][0].set_title(f"RF SHAP (OOB $R^2$ {rf.oob_score_:.2f})", fontsize=8)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rf = RandomForestRegressor(n_estimators=30, oob_score=True)
        rf.fit(X_train, y_train)
        perm_I = importances(rf, X_test, y_test)
        plot_importances(perm_I, ax=axes[i][1], imp_range=(0, 1))
        axes[i][1].set_title(f"Permutation importance", fontsize=8)

        drop_I = dropcol_importances(rf, X_train, y_train, X_test, y_test)
        plot_importances(drop_I, ax=axes[i][2], imp_range=(0, 1))
        axes[i][2].set_title(f"Drop-column importance", fontsize=8)

    plt.suptitle('$'+eqn+'$', y=1.0)
    plt.savefig("shap_results.png", bbox_inches=0, dpi=200)
    plt.show()


def verify_shap_importance_computation():
    "Verify my feature importance plot derived from SHAP values looks like SHAP's plot"
    df, coeff, eqn = synthetic_poly_dup_data(500)
    X = df.drop('y', axis=1)
    y = df['y']

    rf = RandomForestRegressor(n_estimators=30, oob_score=True)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf, data=X, feature_perturbation='interventional')
    shap_values_XGB_train = explainer.shap_values(X)
    shap.summary_plot(shap_values_XGB_train, X, plot_type="bar")

    shap_I = shap_importances(rf, X, normalize=False)
    plot_importances(shap_I, imp_range=(0, 1))
    plt.show()


#verify_shap_importance_computation()
SHAP_trials()