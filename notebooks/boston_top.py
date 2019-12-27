from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)
n = X.shape[0]
use_oob=False
metric = mean_absolute_error

# need small or 0 min_slopes_per_x given tiny toy dataset
R = compare_top_features(X, y, n_shap=n, min_samples_leaf=10,
                         metric=metric,
                         use_oob=use_oob,
                         min_slopes_per_x=0)
print(R)

fig, ax = plt.subplots(1,1,figsize=(4,3.5))

plot_topN(R, ax)
# ax.set_ylabel("OOB $R^2$")
# ax.set_ylabel("Training RMSE (dollars)")
ax.set_ylabel("Training MAE (dollars)")
plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/imp-RMSE.png", dpi=150)
plt.savefig("/Users/parrt/Desktop/boston-MAE.png", dpi=150)
# plt.savefig("/Users/parrt/Desktop/imp-OOB.png", dpi=150)
plt.show()