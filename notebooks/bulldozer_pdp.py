from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

n = 10_000

X, y = load_bulldozer()

X = X.iloc[-n:]
y = y.iloc[-n:]

min_slopes_per_x = 10
min_samples_leaf = 10
# plot_stratpd(X, y, colname='age', targetname='SalePrice',
#              min_slopes_per_x=min_slopes_per_x,
#              show_slope_lines=False,
#              min_samples_leaf=min_samples_leaf)

plot_catstratpd_gridsearch(X, y, 'ModelID', 'SalePrice',
                           sort=None,
                           show_xticks=False, min_y_shifted_to_zero=False)

# plot_catstratpd(X, y, 'ModelID', 'SalePrice',
#                 min_samples_leaf=min_samples_leaf, sort=None,
#                 show_xticks=False,
#                 min_y_shifted_to_zero=False)

# plt.title(f"min_slopes_per_x={min_slopes_per_x}, min_samples_leaf={min_samples_leaf}")
plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/james.png", pad_inches=0, dpi=150)
plt.show()