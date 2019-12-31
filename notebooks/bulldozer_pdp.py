from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from stratx.featimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

n = 20_000

X, y = load_bulldozer()

X = X.iloc[-n:]
y = y.iloc[-n:]

# plot_stratpd(X, y, colname='age', targetname='SalePrice',
#              min_slopes_per_x=min_slopes_per_x,
#              show_slope_lines=False,
#              min_samples_leaf=min_samples_leaf)

plot_stratpd_gridsearch(X, y, colname='YearMade', targetname='SalePrice',
                        min_samples_leaf_values=(10,20,30,40,50,80),
                        min_slopes_per_x_values=(5,10,20,30,40),
                        show_slope_lines=False,
                        yrange=None)

#
# plot_stratpd_gridsearch(X, y, colname='age', targetname='SalePrice',
#                         min_samples_leaf_values=(10,20,30,40,50,80),
#                         min_slopes_per_x_values=(5,10,20,30,40),
#                         show_slope_lines=False,
#                         yrange=(-30000,25000))
#
# plot_catstratpd_gridsearch(X, y, 'ModelID', 'SalePrice',
#                            min_samples_leaf_values=(5,10,15,20,25,30),
#                            sort=None,
#                            show_xticks=False,
#                            show_mean_line=True,
#                            show_all_cat_deltas=False,
#                            style='scatter',
#                            min_y_shifted_to_zero=False)

# plot_catstratpd(X, y, 'ModelID', 'SalePrice',
#                 min_samples_leaf=min_samples_leaf, sort=None,
#                 show_xticks=False,
#                 min_y_shifted_to_zero=False)

# plt.title(f"min_slopes_per_x={min_slopes_per_x}, min_samples_leaf={min_samples_leaf}")
plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/james-yearmade.png", pad_inches=0, dpi=150)
plt.show()