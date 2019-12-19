from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from impimp import *
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


n=15_000
X, y = load_flights(n=n)

# plot_stratpd(X, y, 'FLIGHT_NUMBER', 'ARRIVAL_DELAY',
#              show_slope_counts=True,
#              min_slopes_per_x=n*3.5/1000,
#              min_samples_leaf=5,
#              show_slope_lines=True)
#
plot_catstratpd(X, y, 'FLIGHT_NUMBER', 'ARRIVAL_DELAY',
                min_samples_leaf=10,
                sort=None)

# plot_catstratpd_gridsearch(X, y, 'FLIGHT_NUMBER', 'ARRIVAL_DELAY')
plt.tight_layout()
# rent_pdp()
#plt.savefig("/Users/parrt/Desktop/james.png", pad_inches=0, dpi=150)
plt.show()