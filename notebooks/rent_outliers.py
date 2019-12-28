from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from stratx.featimp import *
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

Again:

StratImpact FEATURES ['bathrooms' 'bedrooms' 'UpperEast' 'hells' 'gowanus' 'latitude'
 'Wvillage' 'longitude' 'Evillage' 'Prospect Park' 'LowerEast' 'financial'
 'astoria' 'ParkSlope' 'Crown Heights' 'interest_level' 'brooklynheights'
 'num_photos' 'num_features' 'num_desc_words']
n, n_top, n_estimators, n_shap, min_samples_leaf 20000 9 40 300 10
       OLS  OLS SHAP  RF SHAP  RF perm  StratImpact
0  950.206  1111.517  998.863  997.666      949.394
1  858.357   645.964  648.281  650.861      858.488
2  821.884   644.835  369.715  371.632      367.207
3  396.133   644.631  235.546  236.858      274.419
4  242.285   647.259  219.490  231.367      252.637
5  202.653   369.401  190.470  231.526      232.102
6  180.456   368.736  185.306  229.807      230.669
7  180.338   366.814  186.535  214.497      227.837
8  177.912   367.602  182.968  214.473      229.847

UNSUPERVISED after fixing to use df.sample(frac=1.0):

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

Another unsupervised run:

StratImpact FEATURES ['bathrooms' 'bedrooms' 'longitude' 'Crown Heights' 'latitude' 'Wvillage'
 'brooklynheights' 'financial' 'hells' 'Prospect Park' 'gowanus'
 'Evillage' 'astoria' 'num_features' 'ParkSlope' 'UpperEast' 'LowerEast'
 'interest_level' 'num_photos' 'num_desc_words']
n, n_top, n_estimators, n_shap, min_samples_leaf 20000 9 40 300 10
        OLS  OLS SHAP   RF SHAP   RF perm  StratImpact
0  1002.063  1191.577  1178.158  1186.209      997.487
1   907.967   668.923   651.435   661.535      909.037
2   870.277   640.625   647.116   653.398      518.632
3   465.504   655.135   646.996   649.423      258.016
4   261.138   647.961   647.225   648.699      254.161
5   258.761   651.098   647.590   653.560      247.549
6   244.032   651.868   391.230   651.086      272.222
7   238.717   399.319   249.407   397.931      260.573
8   248.090   394.441   247.072   250.264      273.344


"""
n = 20_000 # more and shap gets bus error it seems
use_oob=False
metric = mean_absolute_error
compute=True
if compute:
    X, y = load_rent(n=n, clean_prices=False) # <----------- many outliers

    R = compare_top_features(X, y, n_shap=300, min_samples_leaf=10,
                             metric=metric,
                             use_oob=use_oob,
                             min_slopes_per_x=15,
                             top_features_range=(1,9),
                             n_stratpd_trees=1,
                             bootstrap=False,
                             supervised=True)

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