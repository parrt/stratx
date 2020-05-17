from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from support import *
from stratx.featimp import *
from stratx.partdep import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=300)#, threshold=1e10)

np.random.seed(3)

X, y, X_train, X_test, y_train, y_test = load_dataset("flights", "ARRIVAL_DELAY")

print(f"Avg arrival delay {y.mean()}")

# plt.scatter(X['DEPARTURE_TIME'], y, s=1, alpha=.5)
# plt.plot([0,np.max(X['DEPARTURE_TIME'])], [0,0], c='k', lw=.5)
# plt.xlabel("DEPARTURE_TIME")
# plt.ylabel("ARRIVAL_DELAY")
# plt.show()
#
# plt.scatter(X['SCHEDULED_DEPARTURE'], y, s=1, alpha=.5)
# plt.plot([0,np.max(X['SCHEDULED_DEPARTURE'])], [0,0], c='k', lw=.5)
# plt.xlabel("SCHEDULED_DEPARTURE")
# plt.ylabel("ARRIVAL_DELAY")
# plt.show()
#
I = importances(#X, y,
                X_train, y_train,
                catcolnames={'AIRLINE',
                             'ORIGIN_AIRPORT',
                             'DESTINATION_AIRPORT',
                             'FLIGHT_NUMBER',
                             'TAIL_NUMBER',
                             'DAY_OF_WEEK'},
                normalize=False,
                n_trials=1,
                min_samples_leaf=20,
                cat_min_samples_leaf=20
                )
print(I)
exit()
#
# col = 'ORIGIN_AIRPORT'
# col = 'SCHEDULED_DEPARTURE'
# col = 'TAXI_OUT'
# col = 'FLIGHT_NUMBER'


# plot_stratpd_gridsearch(X, y, colname=col, targetname='delay',
#              show_slope_lines=False,
#              min_samples_leaf_values=(15,20,30),
#              # min_slopes_per_x_values=(5,10,15,20),
#              # min_samples_leaf=10,
#              n_trials=10,
#              show_impact=True,
#              show_x_counts=True,
#              # min_slopes_per_x=1
#              )
# plt.show()

# df_test = pd.read_csv(f'data/flights-test.csv')
# X = df_test.drop('ARRIVAL_DELAY', axis=1)
# y = df_test['ARRIVAL_DELAY']
# print(f"Avg arrival delay {y.mean()}, sigma={np.std(y)}")

# col = 'AIR_TIME'
# plot_stratpd(X, y, colname=col, targetname='delay',
#                 min_samples_leaf=20,
#                 n_trials=1,
#                 show_impact=True,
#                 show_x_counts=True,
#                 yrange=(-20, 140)
#                 # min_slopes_per_x=1
#                 )
# plt.show()
# exit()

# for i in range(10):
#     np.random.seed(i)
#
#     col = 'ORIGIN_AIRPORT'
#     uniq_catcodes, avg_per_cat, ignored, merge_ignored = \
#         plot_catstratpd(X, y, colname=col, targetname='delay',
#                         leftmost_shifted_to_zero=False,
#                         min_y_shifted_to_zero=False,
#                         min_samples_leaf=5,
#                         n_trials=1,
#                         show_xticks=False,
#                         show_all_pdp=False,
#                         show_impact=True,
#                         yrange=(-50,450),
#                         figsize=(20,10))
#
#     abs_avg = np.abs(avg_per_cat)
#     a, b = np.nanmin(avg_per_cat), np.nanmax(avg_per_cat)
#     m = np.nanmean(abs_avg)
#     straddle_mean = np.nanmean(np.abs(avg_per_cat - np.nanmean(avg_per_cat)))
#     print(f"mean {np.nanmean(avg_per_cat):6.1f}, abs mean {m:5.1f}, {straddle_mean :5.1f}, range {a:5.1f}..{b:5.1f} = {(b - a):5.1f}")

# plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/flight-{col}.pdf", pad_inches=0)
# plt.show()

# plot_catstratpd_gridsearch(X, y, 'ORIGIN_AIRPORT', 'ARRIVAL_DELAY',
#                            min_samples_leaf_values=(2, 5, 10, 15, 20, 30, 35, 40),
#                            show_all_cat_deltas=False, show_impact=True,
#                            show_xticks=False)
# plt.show()
# exit()

colname = 'TAIL_NUMBER'
uniq_catcodes, combined_avg_per_cat, ignored = \
    plot_catstratpd(X, y, colname, 'ARRIVAL_DELAY',
                    min_samples_leaf=2,
                    yrange=(-125,125),
                    figsize=(14,4),
                    n_trials=1,
                    min_y_shifted_to_zero=False,
                    show_unique_cat_xticks=False,
                    show_impact=True,
                    verbose=False)

print("IGNORED", ignored)
plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/flight-fnum-cat-most_common.pdf", pad_inches=0)
plt.show()

# plot_stratpd_gridsearch(X, y, 'TAXI_IN', 'ARRIVAL_DELAY',
#                         min_samples_leaf_values=(3,5,10,15),
#                         min_slopes_per_x_values=(15,20,25,30,40,50),
#                         show_slope_lines=False,
#                         yrange=(-10,90)
#                         )



# I = importances(X, y,
#                 catcolnames={'AIRLINE',
#                              'ORIGIN_AIRPORT',
#                              'DESTINATION_AIRPORT',
#                              'FLIGHT_NUMBER',
#                              'DAY_OF_WEEK'},
#                 min_samples_leaf=5,
#                 cat_min_samples_leaf=2,
#                 n_trials=1,
#                 normalize=False)
# print(I)

# plot_stratpd(X, y, colname=col, targetname='delay',
#              show_slope_lines=False,
#              min_samples_leaf=5,
#              n_trials=3,
#              show_impact=False,
#              show_x_counts=True,
#              )
#              # yrange=(-10,100))
# plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/{col}.pdf", pad_inches=0)
# plt.show()



# plot_stratpd_gridsearch(X, y, 'DEPARTURE_TIME', 'ARRIVAL_DELAY',
#                         show_slope_lines=False,
#                         show_impact=True)
# plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/flight-dep-time-4.pdf", pad_inches=0)
# plt.show()


# I = spearmans_importances(X, y)
# print(I)

# plot_stratpd_gridsearch(X, y, col, 'ARRIVAL_DELAY',
#                         min_samples_leaf_values=(10,15,20,30),
#                         min_slopes_per_x_values=(5,10,15,20,25),
#                         show_slope_lines=False,
#                         yrange=(-10,90)
#                         )

# #
# plot_catstratpd(X, y, 'SCHEDULED_DEPARTURE_HOUR', 'ARRIVAL_DELAY',
#                 min_samples_leaf=10,
#                 # sort=None,
#                 yrange=(-110,250),
#                 show_xticks=False,
#                 style='scatter')
# plt.title(f"X range {r[0]}..{r[1]} with {n} records")

# I = importances(X, y,
#                 min_samples_leaf=10, # default
#                 min_slopes_per_x=20,
#                 catcolnames={'AIRLINE',
#                              'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
#                              'FLIGHT_NUMBER',
#                              'DAY_OF_WEEK', 'dayofyear'},
#                 )
# print(I)



# plt.tight_layout()
# # rent_pdp()
# plt.show()