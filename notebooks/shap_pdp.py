from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
import statsmodels.api as sm

import shap

from impimp import *
from stratx.partdep import *
from stratx.ice import *

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
    df['x3'] = df['x1'] + np.random.random_sample(size=n)-0.5 # copy x1 into x3 with noise
    yintercept = 100
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " \\,\\,where\\,\\, x_3 = x_1 + noise"
    return df, coeff, eqn


# plot_stratpd(X, y, colname='x3', targetname='y')

def shap_slope_expectation(newdata=True):
    ntrials = 100
    min_samples_leaf = 3
    backgroundsize = 300

    n = 2000
    if not newdata:
        df, coeff, eqn = synthetic_poly_dup_data(n)
        X = df.drop('y', axis=1)
        y = df['y']

    results = []
    for i in range(ntrials):
        if newdata:
            df, coeff, eqn = synthetic_poly_dup_data(n)
            X = df.drop('y', axis=1)
            y = df['y']
        rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=min_samples_leaf, oob_score=True)
        rf.fit(X, y)
        explainer = shap.TreeExplainer(rf, data=X.iloc[:backgroundsize], feature_perturbation='interventional')
        #explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X, check_additivity=True)
        #abs_shap_values = np.abs(shap_values)

        x1 = X[['x1']]
        x2 = X[['x2']]
        x3 = X[['x3']]
        shap_x1 = shap_values[:,0]
        shap_x2 = shap_values[:,1]
        shap_x3 = shap_values[:,2]

        slopes = []
        lm = LinearRegression().fit(x1, shap_x1)
        slopes.append(lm.coef_[0])
        lm = LinearRegression().fit(x2, shap_x2)
        slopes.append(lm.coef_[0])
        lm = LinearRegression().fit(x3, shap_x3)
        slopes.append(lm.coef_[0])
        # print(np.mean(np.abs(shap_values),axis=0))
        #print("Coeffs ", slopes)
        results.append(slopes)

    print(f"RF min_samples_leaf={min_samples_leaf}, backgroundsize={backgroundsize}, {ntrials} trials of {n} records: {np.mean(results, axis=0)} with stddev {np.std(results, axis=0)}")


def our_slope_expectation(newdata=True):
    ntrials = 15
    n = 5000
    for min_samples_leaf in [3,5,7,8,10,12,14]:
        if not newdata:
            df, coeff, eqn = synthetic_poly_dup_data(n)
            X = df.drop('y', axis=1)
            y = df['y']

        results = []
        for i in range(ntrials):
            if newdata:
                df, coeff, eqn = synthetic_poly_dup_data(n)
                X = df.drop('y', axis=1)
                y = df['y']

            leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx1, pdpy1, ignored = \
                partial_dependence(X=X, y=y, colname='x1', min_samples_leaf=min_samples_leaf)
            leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx2, pdpy2, ignored = \
                partial_dependence(X=X, y=y, colname='x2', min_samples_leaf=min_samples_leaf)
            leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx3, pdpy3, ignored = \
                partial_dependence(X=X, y=y, colname='x3', min_samples_leaf=min_samples_leaf)

            slopes = []
            lm = LinearRegression().fit(pdpx1.reshape(-1,1), pdpy1)
            slopes.append(lm.coef_[0])
            lm = LinearRegression().fit(pdpx2.reshape(-1,1), pdpy2)
            slopes.append(lm.coef_[0])
            lm = LinearRegression().fit(pdpx3.reshape(-1,1), pdpy3)
            slopes.append(lm.coef_[0])
            # print("Coeffs ", slopes)
            results.append(slopes)
        print(f"StratImpact w/min_samples_leaf={min_samples_leaf:2d}, {ntrials} trials of {n} records: {np.mean(results, axis=0)} with stddev {np.std(results, axis=0)}")


def shap_pdp_plots(n=1000, feature_perturbation='interventional'):
    min_samples_leaf = 3
    backgroundsize = 1000
    df, coeff, eqn = synthetic_poly_dup_data(n=n)
    X = df.drop('y', axis=1)
    y = df['y']
    rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=min_samples_leaf,
                               oob_score=True)
    rf.fit(X, y)
    if feature_perturbation=='interventional':
        explainer = shap.TreeExplainer(rf, data=X.iloc[:backgroundsize],
                                       feature_perturbation=feature_perturbation)
    else:
        explainer = shap.TreeExplainer(rf, feature_perturbation=feature_perturbation)
    shap_values = explainer.shap_values(X, check_additivity=True)
    shap.dependence_plot("x1", shap_values, X, interaction_index=None)
    shap.dependence_plot("x2", shap_values, X, interaction_index=None)  # , ax=axes[1])
    shap.dependence_plot("x3", shap_values, X, interaction_index=None)  # , ax=axes[2])


def OLS():
    df, coeff, eqn = synthetic_poly_dup_data(n=1000)
    X = df.drop('y', axis=1)
    y = df['y']
    lm = LinearRegression().fit(X, y)
    print("OLS coeff", lm.coef_)
    y_pred = lm.predict(X)
    print(f"Training MSE {np.mean((y-y_pred)**2):.5f}")

    beta_stderr = sm.OLS(y, X).fit().bse
    print(f"beta stderr {beta_stderr.values}")
    imp = lm.coef_
    imp /= beta_stderr
    print(f"Adjusted betas {imp.values}")


#OLS()

#our_slope_expectation()
#shap_slope_expectation()

#fig, axes = plt.subplots(1, 3, figsize=(8, 3))

# OH! partial_dependence_plot really is just plain PDP so don't use
#shap.partial_dependence_plot(0, rf.predict, X, feature_names=X.columns)


shap_pdp_plots()
#shap_pdp_plots(feature_perturbation='tree_path_dependent')

#plot_stratpd(X, y, colname='x1', targetname='y', min_samples_leaf=5, ax=axes[0])
# plot_stratpd(X, y, colname='x2', targetname='y', min_samples_leaf=5, ax=axes[1])
# plot_stratpd(X, y, colname='x3', targetname='y', min_samples_leaf=5, ax=axes[2])

# plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/polydup_strat_vs_ice.pdf", bbox_inches=0)
# plt.show()