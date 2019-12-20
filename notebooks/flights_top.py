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
metric = mean_absolute_error
compute=True
if compute:
    n = 10_000

    X, y, _ = load_flights(n=n)

    R = compare_top_features(X, y, n_shap=300, min_samples_leaf=15,
                             min_slopes_per_x=15,
                             n_estimators=40,
                             catcolnames={'AIRLINE',
                                          'ORIGIN_AIRPORT','DESTINATION_AIRPORT',
                                          'FLIGHT_NUMBER',
                                          'DAY_OF_WEEK', 'dayofyear'},
                             metric=mean_squared_error,
                             use_oob=use_oob)

    R = R.reset_index(drop=True)
    if use_oob:
        R.to_feather("/tmp/flights-importances-oob.feather")
    elif metric == mean_squared_error:
        R.to_feather("/tmp/flights-importances-MSE.feather")
    else:
        R.to_feather("/tmp/flights-importances-MAE.feather")
else:
    # R=pd.read_feather("/tmp/flights-importances-oob.feather")
    R=pd.read_feather("/tmp/sample-importances-MAE.feather")
    # R=pd.read_feather("/tmp/sample-importances-MSE.feather")

print(R)

fig, ax = plt.subplots(1,1,figsize=(6,3.5))

plot_topN(R, ax)
ax.set_ylabel("OOB $R^2$")
# ax.set_ylabel("Training RMSE (dollars)")
# ax.set_ylabel("Training MAE (dollars)")
plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/flights-imp-RMSE.png", dpi=150)
# plt.savefig("/Users/parrt/Desktop/flights-imp-MAE.png", dpi=150)
plt.savefig("/Users/parrt/Desktop/flights-imp-OOB.png", dpi=150)
plt.show()