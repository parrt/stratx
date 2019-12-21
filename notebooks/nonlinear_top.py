from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from impimp import *
from stratx import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

def synthetic_nonlinear_data(n, p=2):
    df = pd.DataFrame()
    for i in range(p):
        df[f'x{i + 1}'] = np.random.random_sample(size=n) * 3
    yintercept = 100
    df['y'] = 4*df['x1']**2 + df['x2'] + yintercept
    eqn = "y = 4*x1^2 + x2"
    return df, eqn


n = 1000
df, eqn = synthetic_nonlinear_data(n, p=2)
X = df.drop('y', axis=1)
y = df['y']

plot_stratpd_gridsearch(X, y, 'x1', 'price')

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
# plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/polydup_strat_vs_ice.pdf", bbox_inches=0)
plt.show()