from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from impimp import *
from stratx import *
from support import *

import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

def synthetic_nonlinear_data(n, p=2):
    df = pd.DataFrame()
    for i in range(p):
        df[f'x{i + 1}'] = np.random.random_sample(size=n) * 3
    yintercept = 100
    df['y'] = df['x1']**2 + df['x2'] + yintercept
    eqn = f"y = x1^2 + x2 + {yintercept}, xi ~ U(0,3)"
    return df, eqn


n = 1000
shap_test_size = n
df, eqn = synthetic_nonlinear_data(n, p=2)
X = df.drop('y', axis=1)
y = df['y']

def OLS():
    lm = LinearRegression()
    lm.fit(X,y)
    print("OLS coeff", lm.coef_)
    y_pred = lm.predict(X)
    print(f"OLS Training MSE {np.mean((y - y_pred) ** 2):.5f}")
    explainer = shap.LinearExplainer(lm, X, feature_dependence='independent')
    shap_values = explainer.shap_values(X)
    return shap_values

def RF():
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    rf.fit(X,y)
    # print("OOB", rf.oob_score_)

    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100), feature_perturbation='interventional')
    shap_values = explainer.shap_values(X[:shap_test_size], check_additivity=False)
    return shap_values

# shap_values = OLS()
shap_values = RF()

print(f"n={n}, {eqn}, avg={np.mean(y)}")
shapimp = np.mean(np.abs(shap_values), axis=0)
s = np.sum(shapimp)
print("\nRF SHAP importances", list(shapimp), list(shapimp / s))
# print(shap_values[:10])

#shap.summary_plot(shap_values, X[:shap_test_size])
shap.dependence_plot("x1", shap_values, X[:shap_test_size], interaction_index=None)
shap.dependence_plot("x2", shap_values, X[:shap_test_size], interaction_index=None)

# plot_stratpd_gridsearch(X, y, 'x1', 'price')

I = impact_importances(X, y, normalize=False)
print(I)

# plot_stratpd(X, y, colname='x1', targetname='y', min_samples_leaf=10,
#              min_slopes_per_x=15)
# plot_stratpd(X, y, colname='x2', targetname='y', min_samples_leaf=10,
#              min_slopes_per_x=15)

# R = compare_top_features(X, y, n_shap=500, min_samples_leaf=10,
#                          min_slopes_per_x=15,
#                          n_estimators=40,
#                          metric=mean_squared_error,
#                          use_oob=False)
#
# print(R)
#dupcol()
plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/foo.png", bbox_inches=0, dpi=150)
plt.show()