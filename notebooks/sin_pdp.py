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

def synthetic_poly_dup_data(n, p=2):
    df = pd.DataFrame()
    for i in range(p):
        df[f'x{i + 1}'] = np.random.random_sample(size=n) * 2*np.pi
    yintercept = 100
    df['y'] = df['x1']**2 + df['x2'] + np.sin(df['x3'])*5 + yintercept
    eqn = "y = x1^2 + x2 + 5sin(x3) vars in range [0,2pi)"
    return df, eqn


n = 3000
df, eqn = synthetic_poly_dup_data(n, p=3)
X = df.drop('y', axis=1)
y = df['y']
# plot_stratpd(X, y, colname='x1', targetname='y', min_samples_leaf=10)
# plot_stratpd(X, y, colname='x2', targetname='y', min_samples_leaf=10)
#plot_stratpd(X, y, colname='x3', targetname='y', min_samples_leaf=25)

# plot_stratpd_gridsearch(X, y, 'x3', 'y')

#
R = compare_top_features(X, y, n_shap=500, min_samples_leaf=25,
                         min_slopes_per_x=10,
                         n_estimators=40,
                         metric=mean_squared_error,
                         use_oob=False)

print(R)
#dupcol()
# plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/polydup_strat_vs_ice.pdf", bbox_inches=0)
plt.show()