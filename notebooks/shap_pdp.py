from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from impimp import *
from stratx import *

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


df, coeff, eqn = synthetic_poly_dup_data(500)
X = df.drop('y', axis=1)
y = df['y']
# plot_stratpd(X, y, colname='x3', targetname='y')

rf = RandomForestRegressor(n_estimators=40, oob_score=True)
rf.fit(X, y)
explainer = shap.TreeExplainer(rf, data=X, feature_perturbation='interventional')
shap_values = explainer.shap_values(X, check_additivity=True)

#fig, axes = plt.subplots(3, 1, figsize=(3, 5))

"""
def partial_dependence_plot(ind, model, features, xmin="percentile(0)", xmax="percentile(100)",
                            npoints=None, nsamples=100, feature_names=None, hist=True,
                            ylabel=None, ice=False, opacity=None, linewidth=None, show=True):
                            """

shap.partial_dependence_plot(2, rf.predict, X, feature_names=X.columns)
# shap.partial_dependence_plot("x2", shap_values, X, ax=axes[1])
# shap.partial_dependence_plot("x3", shap_values, X, ax=axes[2])

#plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/polydup_strat_vs_ice.pdf", bbox_inches=0)
#plt.show()