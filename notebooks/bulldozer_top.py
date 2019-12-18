from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rfpimp import plot_importances, dropcol_importances, importances

use_oob=False
metric = mean_squared_error
compute=False
if compute:
    n = 20_000

    X, y = load_bulldozer()

    X = X.iloc[-n:]
    y = y.iloc[-n:]

    R = compare_top_features(X, y, n_shap=500, min_samples_leaf=10, min_slopes_per_x=n*3/1000,
                             catcolnames={'AC','ModelID'},
                             metric=mean_squared_error,
                             use_oob=use_oob)

    R = R.reset_index(drop=True)
    if use_oob:
        R.to_feather("/tmp/sample-importances-oob.feather")
    elif metric == mean_squared_error:
        R.to_feather("/tmp/sample-importances-MSE.feather")
    else:
        R.to_feather("/tmp/sample-importances-MAE.feather")
else:
    R=pd.read_feather("/tmp/sample-importances-oob.feather")
    # R=pd.read_feather("/tmp/sample-importances-MAE.feather")
    # R=pd.read_feather("/tmp/sample-importances-MSE.feather")

print(R)

fig, ax = plt.subplots(1,1,figsize=(4,3.5))

plot_topN(R, ax)
ax.set_ylabel("OOB $R^2$")
# ax.set_ylabel("Training RMSE (dollars)")
# ax.set_ylabel("Training MAE (dollars)")
plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/imp-RMSE.png", dpi=150)
# plt.savefig("/Users/parrt/Desktop/imp-MAE.png", dpi=150)
plt.savefig("/Users/parrt/Desktop/imp-OOB.png", dpi=150)
plt.show()