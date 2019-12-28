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


# def rent_pdp():
#     X, y = load_rent(n=2_000)
#     # plot_stratpd_gridsearch(X, y, 'bedrooms', 'price')
#     # plot_stratpd_gridsearch(X, y, 'bathrooms', 'price')
#     plot_stratpd_gridsearch(X, y, 'Wvillage', 'price',
#                             min_samples_leaf_values=(2,3,5,8,10,15))
#     # plot_stratpd_gridsearch(X, y, 'latitude', 'price')
#     # plot_stratpd_gridsearch(X, y, 'longitude', 'price')


np.random.seed(999)

n=10_000
r = (500,600)
# r = (0,500)
_, _, df_flights = load_flights(n=n)
df_flights = df_flights[df_flights['FLIGHT_NUMBER']>r[0]] # look at subset of flight numbers
df_flights = df_flights[df_flights['FLIGHT_NUMBER']<r[1]] # look at subset of flight numbers
X, y = df_flights.drop('ARRIVAL_DELAY', axis=1), df_flights['ARRIVAL_DELAY']

print(f"Avg arrival delay {df_flights['ARRIVAL_DELAY'].mean()}")

# plot_stratpd(X, y, 'FLIGHT_NUMBER', 'ARRIVAL_DELAY',
#              show_slope_counts=True,
#              min_slopes_per_x=n*3.5/1000,
#              min_samples_leaf=5,
#              show_slope_lines=True)
#
plot_catstratpd(X, y, 'FLIGHT_NUMBER', 'ARRIVAL_DELAY',
                min_samples_leaf=10,
                # sort=None,
                yrange=(-110,250),
                show_xticks=False,
                style='scatter')
plt.title(f"X range {r[0]}..{r[1]} with {n} records")

# plot_catstratpd_gridsearch(X, y, 'ORIGIN_AIRPORT', 'ARRIVAL_DELAY',
#                            yrange=(-100,500))
plt.tight_layout()
# rent_pdp()
plt.savefig("/Users/parrt/Desktop/james.svg", pad_inches=0, dpi=150)
plt.show()