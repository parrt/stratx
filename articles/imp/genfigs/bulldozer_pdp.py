from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap
from sympy.simplify.radsimp import fraction_expand

from stratx.featimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=300, threshold=2000)

n = 20_000

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

idxs = resample(range(50_000), n_samples=n, replace=False,)
X_, y_ = X.iloc[idxs], y.iloc[idxs]

# order = X_['ModelID'].values.argsort()
# ranks = order.argsort()
# X_['ModelID'] = ranks  # aid debugging
X_ = X_.copy()
# X_['ModelID'] -= np.min(X_['ModelID'])  # aid debugging
# X_['ModelID'] += 1


# I = importances(X_, y_,
#                 n_trials=5,
#                 normalize=False,
#                 min_samples_leaf=10,
#                 cat_min_samples_leaf=15,
#                 pvalues=True,
#                 pvalues_n_trials=10,
#                 catcolnames={'AC','ModelID'})
# print(I)

#y_ = y_.sample(frac=1.0, replace=False)
# uniq_catcodes, combined_avg_per_cat, ignored, merge_ignored = \
#     plot_catstratpd(X_, y_, colname='ModelID', targetname='SalePrice',
#                     min_samples_leaf=5,
#                     # sort=None,
#                     alpha=.08,
#                     show_all_deltas=False,
#                     n_trials=1,
#                     show_xticks=False,
#                     show_impact=True,
#                     min_y_shifted_to_zero=False,
#                     figsize=(20,5),
#                     yrange=(-150_000, 150_000),
#                     verbose=False)
# plt.title(f"n={n}, ignored ={ignored}")
# print("ignored",ignored)
# plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/james-ModelID-25k.pdf", pad_inches=0)
# # plt.savefig(f"/Users/parrt/Desktop/james-ModelID-25k-shuffled.pdf", pad_inches=0)
# plt.show()

plot_stratpd(X_, y_, colname='YearMade', targetname='SalePrice',
             n_trials=3,
             show_slope_lines=False,
             show_impact=False,
             figsize=(3.8,3.2)
             )
plt.tight_layout()
plt.savefig(f"/Users/parrt/Desktop/james-YearMade.pdf", pad_inches=0)
plt.show()


# I = importances(X_, y_,
#                 min_samples_leaf=5,
#                 # min_slopes_per_x=5,
#                 n_trials=10,
#                 sort='Rank',
#                 catcolnames={'AC', 'ModelID'})
# print(I)
# I.to_csv("/tmp/t2.csv")

#plot_catstratpd(X_, y_, colname='ProductSize', targetname='SalePrice')


# plot_stratpd(X_, y_, colname='YearMade', targetname='SalePrice',
#              show_slope_lines=False,
#              show_impact=True,
#              min_samples_leaf=5,
#              min_slopes_per_x=7,
#              pdp_marker_cmap='coolwarm',#'YlGnBu',#'tab20b',
#              figsize=(4,3)
#              )
# plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/james-YearMade.pdf", pad_inches=0)
# plt.show()




# col = 'YearMade'
# plot_stratpd_gridsearch(X_, y_, colname=col, targetname='SalePrice',
#                         min_samples_leaf_values=(2,3,4,5,6,7),
#                         min_slopes_per_x_values=(2,3,5,6,7,8),
#                         show_slope_lines=False,
#                         show_impact=True
#                         #,yrange=(-20000,2000)
#                         )
# plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/james-{col}-grid.pdf", pad_inches=0)
# plt.show()

#
# plot_stratpd(X_, y_, colname='saledayofyear', targetname='SalePrice',
#              show_impact=True,
#              show_slope_lines=False)
# plt.tight_layout()
# plt.show()
# #
# plot_stratpd(X_, y_, colname='ProductSize', targetname='SalePrice',
#              show_impact=True,
#              show_slope_lines=False)
# plt.tight_layout()
# plt.show()




# I = importances(X_, y_, catcolnames={'AC', 'ModelID', 'ProductSize'},
#                 min_samples_leaf=10,
#                 min_slopes_per_x=5)
# print(I)


col = 'age'
# col = 'ProductSize'

# plot_catstratpd_gridsearch(X_, y_, 'ProductSize', 'SalePrice',
#                            min_samples_leaf_values=(5,10,15,20),
#                            sort=None,
#                            show_xticks=False,
#                            show_mean_line=True,
#                            show_all_cat_deltas=False,
#                            style='scatter',
#                            min_y_shifted_to_zero=False)

# plot_catstratpd(X_, y_, 'YearMade', 'SalePrice',
#                 min_samples_leaf=10,
#                 show_mean_line=True,
#                 sort=None,
#                 show_xticks=False,
#                 min_y_shifted_to_zero=False)

