from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

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
    ntrials = 10

    n = 1000
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
        rf = RandomForestRegressor(n_estimators=30, oob_score=True)
        rf.fit(X, y)
        explainer = shap.TreeExplainer(rf, data=X.iloc[:100], feature_perturbation='interventional')
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
        print("Coeffs ", slopes)
        results.append(slopes)

    print(f"{ntrials} trials of {n} records: {np.mean(results, axis=0)} with stddev {np.std(results, axis=0)}")


def our_slope_expectation(newdata=True):
    ntrials = 10

    n = 1000
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
        rf = RandomForestRegressor(n_estimators=30, oob_score=True)
        rf.fit(X, y)

        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy1, ignored = \
            partial_dependence(X=X, y=y, colname='x1', min_samples_leaf=5)
        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy2, ignored = \
            partial_dependence(X=X, y=y, colname='x1', min_samples_leaf=5)
        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy3, ignored = \
            partial_dependence(X=X, y=y, colname='x1', min_samples_leaf=5)

        x1 = np.array(sorted(np.unique(X['x1'].values[:-1])))
        x2 = np.array(sorted(np.unique(X['x2'].values[:-1])))
        x3 = np.array(sorted(np.unique(X['x3'].values[:-1])))

        slopes = []
        lm = LinearRegression().fit(x1.reshape(-1,1), pdpy1)
        slopes.append(lm.coef_[0])
        lm = LinearRegression().fit(x2.reshape(-1,1), pdpy2)
        slopes.append(lm.coef_[0])
        lm = LinearRegression().fit(x3.reshape(-1,1), pdpy3)
        slopes.append(lm.coef_[0])
        # print(np.mean(np.abs(shap_values),axis=0))
        print("Coeffs ", slopes)
        results.append(slopes)

    print(f"StratImpact {ntrials} trials of {n} records: {np.mean(results, axis=0)} with stddev {np.std(results, axis=0)}")


our_slope_expectation()
#shap_slope_expectation()

#fig, axes = plt.subplots(1, 3, figsize=(8, 3))

# OH! partial_dependence_plot really is just plain PDP so don't use
#shap.partial_dependence_plot(0, rf.predict, X, feature_names=X.columns)

if False:
    shap.dependence_plot("x1", shap_values, X, interaction_index=None)
    shap.dependence_plot("x2", shap_values, X, interaction_index=None)#, ax=axes[1])
    shap.dependence_plot("x3", shap_values, X, interaction_index=None)#, ax=axes[2])

#plot_stratpd(X, y, colname='x1', targetname='y', min_samples_leaf=5, ax=axes[0])
# plot_stratpd(X, y, colname='x2', targetname='y', min_samples_leaf=5, ax=axes[1])
# plot_stratpd(X, y, colname='x3', targetname='y', min_samples_leaf=5, ax=axes[2])

# plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/polydup_strat_vs_ice.pdf", bbox_inches=0)
# plt.show()