from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from stratx.featimp import *
from support import *

import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1) # reproducible for paper

width = 3

def synthetic_interaction_data(n):
    df = pd.DataFrame()
    for i in range(2):
        df[f'x{i + 1}'] = np.random.random_sample(size=n) * width
    yintercept = 100
    df['y'] = df['x1']**2 + df['x2'] + yintercept
    eqn = "y = x1 * x2"
    return df, eqn


def RF():
    rf = RandomForestRegressor(n_estimators=30, oob_score=True, n_jobs=-1)
    rf.fit(X,y)
    print("OOB", rf.oob_score_)

    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100), feature_perturbation='interventional')
    shap_values = explainer.shap_values(X[:shap_test_size], check_additivity=False)
    shapimp = np.mean(np.abs(shap_values), axis=0)
    s = np.sum(shapimp)
    print("\nRF SHAP importances", list(shapimp), shapimp*width, list(shapimp/s))
    return shap_values


n = 2000
shap_test_size = 2000
df, eqn = synthetic_interaction_data(n)
X = df.drop('y', axis=1)
y = df['y']

print("y bar", np.mean(y) - 100)

shap_values = RF()

shap.dependence_plot("x1", shap_values, X[:shap_test_size], interaction_index=None)
shap.dependence_plot("x2", shap_values, X[:shap_test_size], interaction_index=None)

plot_stratpd(X, y, colname='x1', targetname='y')
plot_stratpd(X, y, colname='x2', targetname='y')

I = importances(X, y, normalize=False,
                min_slopes_per_x=5, # data very clean and uniform, don't toss much out
                n_trials=5)
print(I)

plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/foo.png", bbox_inches=0, dpi=150)
plt.show()