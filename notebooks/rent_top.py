from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

n = 20_000 # more and shap gets bus error it seems
use_oob=False
metric = mean_absolute_error
compute=True
if compute:
    X, y = load_rent(n=n)

    R = compare_top_features(X, y, n_shap=300, min_samples_leaf=10,
                             metric=metric,
                             use_oob=use_oob,
                             min_slopes_per_x=15)

    R = R.reset_index(drop=True)
    if use_oob:
        R.to_feather("/tmp/rent-importances-oob.feather")
    elif metric == mean_squared_error:
        R.to_feather("/tmp/rent-importances-MSE.feather")
    else:
        R.to_feather("/tmp/rent-importances-MAE.feather")
else:
    R=pd.read_feather("/tmp/rent-importances-oob.feather")
    # R=pd.read_feather("/tmp/rent-importances-MAE.feather")
    # R=pd.read_feather("/tmp/rent-importances-MSE.feather")

fig, ax = plt.subplots(1,1,figsize=(5,3.5))

print(R)

plot_topN(R, ax)
ax.set_ylabel("OOB $R^2$")
# ax.set_ylabel("Training RMSE (dollars)")
# ax.set_ylabel("Training MAE (dollars)")
plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/rent-imp-RMSE.png", dpi=150)
# plt.savefig("/Users/parrt/Desktop/rent-imp-MAE.png", dpi=150)
plt.savefig("/Users/parrt/Desktop/rent-imp-OOB.png", dpi=150)
plt.show()