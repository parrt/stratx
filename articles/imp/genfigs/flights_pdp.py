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


# def rent_pdp():
#     X, y = load_rent(n=2_000)
#     # plot_stratpd_gridsearch(X, y, 'bedrooms', 'price')
#     # plot_stratpd_gridsearch(X, y, 'bathrooms', 'price')
#     plot_stratpd_gridsearch(X, y, 'Wvillage', 'price',
#                             min_samples_leaf_values=(2,3,5,8,10,15))
#     # plot_stratpd_gridsearch(X, y, 'latitude', 'price')
#     # plot_stratpd_gridsearch(X, y, 'longitude', 'price')


#np.random.seed(999)

n=20_000
#r = (500,600)
# r = (0,500)
_, _, df_flights = load_flights(n=n)
# df_flights = df_flights[df_flights['FLIGHT_NUMBER']>r[0]] # look at subset of flight numbers
# df_flights = df_flights[df_flights['FLIGHT_NUMBER']<r[1]] # look at subset of flight numbers
X, y = df_flights.drop('ARRIVAL_DELAY', axis=1), df_flights['ARRIVAL_DELAY']

print(f"Avg arrival delay {df_flights['ARRIVAL_DELAY'].mean()}")

# plot_stratpd_gridsearch(X, y, 'TAXI_IN', 'ARRIVAL_DELAY',
#                         min_samples_leaf_values=(3,5,10,15),
#                         min_slopes_per_x_values=(15,20,25,30,40,50),
#                         show_slope_lines=False,
#                         yrange=(-10,90)
#                         )

col = 'DEPARTURE_TIME'
col = 'SCHEDULED_DEPARTURE'
col = 'FLIGHT_NUMBER'
col = 'TAXI_OUT'

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

uniq_catcodes, combined_avg_per_cat, ignored = \
    plot_catstratpd(X, y, 'FLIGHT_NUMBER', 'ARRIVAL_DELAY',
                    min_samples_leaf=10,
                    sort=None,
                    # yrange=(-110,250),
                    figsize=(20,4),
                    n_trials=3,
                    show_all_deltas=False,
                    show_xticks=False,
                    show_impact=True,
                    verbose=False)

print("IGNORED", ignored)
plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/flight-fnum-cat-most_common.pdf", pad_inches=0)
plt.savefig(f"/Users/parrt/Desktop/flight-fnum-cat.pdf", pad_inches=0)
plt.show()


# plot_stratpd(X, y, colname='DEPARTURE_TIME_HOUR', targetname='delay',
#              show_slope_lines=False,
#              show_impact=True)
#              # yrange=(-10,100))
# plt.tight_layout()
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

# plot_catstratpd_gridsearch(X, y, 'FLIGHT_NUMBER', 'ARRIVAL_DELAY',
#                            min_samples_leaf_values=(2, 3, 5),
#                            show_all_cat_deltas=False, show_impact=True,
#                            show_xticks=False,
#                            min_y_shifted_to_zero=False,
#                            sort=False)



# plt.tight_layout()
# # rent_pdp()
# plt.show()