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

"""
Interesting. With clean_prices=False, the wacky 1M apts skew the
other models more than ours (20k records): 

StratImpact FEATURES ['financial' 'num_desc_words' 'UpperEast' 'astoria' 'hells' 'Wvillage'
 'LowerEast' 'Evillage' 'brooklynheights' 'gowanus' 'ParkSlope'
 'num_photos' 'latitude' 'longitude' 'bathrooms' 'bedrooms'
 'Prospect Park' 'Crown Heights' 'num_features' 'interest_level']
n, n_top, n_estimators, n_shap, min_samples_leaf 20000 20 40 300 10
         OLS  OLS SHAP   RF SHAP   RF perm  StratImpact
0   2018.526  1829.719  1726.609  1808.009     1447.951
1   1642.705   938.284   963.438  1006.186      875.948
2   1409.509   909.505   965.900   982.452      798.704
3    557.035   931.050   945.997   958.893      807.695
4    549.699   920.696   939.367   949.090      725.126
5    547.751   904.000   584.176   927.039      670.549
6    496.922   916.106   613.416   957.444      629.804
7    427.931   943.405   630.837   968.972      615.370
...

unsupervised after fixing to use df.sample(frac=1.0):

StratImpact FEATURES ['bathrooms' 'hells' 'bedrooms' 'Wvillage' 'longitude' 'Crown Heights'
 'brooklynheights' 'gowanus' 'latitude' 'num_photos' 'financial'
 'ParkSlope' 'Prospect Park' 'num_desc_words' 'num_features' 'astoria'
 'LowerEast' 'Evillage' 'interest_level' 'UpperEast']
n, n_top, n_estimators, n_shap, min_samples_leaf 20000 9 40 300 10
        OLS  OLS SHAP   RF SHAP   RF perm  StratImpact
0  1005.547  1182.408  1177.640  1182.362     1005.040
1   918.747   647.418   645.310   653.669      716.868
2   880.523   642.959   663.211   644.058      505.287
3   430.760   649.275   645.816   647.527      385.280
4   265.660   661.731   388.743   661.014      274.530
5   260.155   655.660   387.622   645.377      276.420
6   258.974   655.712   391.037   650.455      269.870
7   249.645   651.291   267.413   387.181      273.802
8   246.788   662.372   266.152   273.445      267.661

"""
n = 20_000 # more and shap gets bus error it seems
use_oob=False
metric = mean_absolute_error
compute=True
if compute:
    X, y = load_rent(n=n, clean_prices=False)

    R = compare_top_features(X, y, n_shap=300, min_samples_leaf=10,
                             metric=metric,
                             use_oob=use_oob,
                             min_slopes_per_x=15,
                             top_features_range=(1,9))

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