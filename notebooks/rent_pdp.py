from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from stratx.featimp import *
from stratx.partdep import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

"""
supervised on all ~48k records n_jobs=1:  seems stable

Impact importance time 34s
                 Importance
Feature                    
bedrooms           0.270667
bathrooms          0.168106
brooklynheights    0.099670
astoria            0.065124
hells              0.054125
num_features       0.051710
Crown Heights      0.050172
num_photos         0.039396
Wvillage           0.033934
financial          0.029263
interest_level     0.028220
LowerEast          0.019348
UpperEast          0.016349
longitude          0.015403
num_desc_words     0.015155
latitude           0.009595
gowanus            0.009375
Prospect Park      0.008850
Evillage           0.008173
ParkSlope          0.007366

UNSUPERVISED with n_jobs=4:
Impact importance time 59s
                 Importance
Feature                    
bedrooms           0.099625
bathrooms          0.097397
Wvillage           0.077631
latitude           0.074224
brooklynheights    0.065640
Crown Heights      0.065229
longitude          0.059962
gowanus            0.053844
hells              0.049902
Evillage           0.048451
LowerEast          0.046792
financial          0.041366
UpperEast          0.039705
Prospect Park      0.039120
num_features       0.036667
astoria            0.030857
ParkSlope          0.029442
num_photos         0.023127
interest_level     0.013837
num_desc_words     0.007180

With 1 job: Impact importance time 131s so about 2x longer. seems pretty stable.

"""
# def rent_pdp():
#     X, y = load_rent(n=2_000)
#     # plot_stratpd_gridsearch(X, y, 'bedrooms', 'price')
#     # plot_stratpd_gridsearch(X, y, 'bathrooms', 'price')
#     plot_stratpd_gridsearch(X, y, 'Wvillage', 'price',
#                             min_samples_leaf_values=(2,3,5,8,10,15))
#     # plot_stratpd_gridsearch(X, y, 'latitude', 'price')
#     # plot_stratpd_gridsearch(X, y, 'longitude', 'price')


#np.random.seed(999)

X, y = load_rent(n=50_000)

# I = impact_importances(X, y, 'price', n_jobs=1, supervised=False)
# print(I)

plot_stratpd(X, y, 'Wvillage', 'price',
             show_slope_counts=True,
             n_trees=1,
             bootstrap=False,
             # supervised=False,
             supervised=True,
             min_slopes_per_x=140,
             min_samples_leaf=10,
             show_slope_lines=False,
             )

# plot_stratpd_gridsearch(X, y, 'bedrooms', 'price',
#                         min_slopes_per_x_values=(0,5))
plt.tight_layout()
# rent_pdp()
plt.savefig("/Users/parrt/Desktop/james.png", pad_inches=0, dpi=150)
plt.show()