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


np.random.seed(999)

X, y = load_rent(n=2_000)
leaf_xranges, leaf_slopes, slope_counts_at_x, pdpx, pdpy, ignored = \
    plot_stratpd(X, y, 'num_desc_words', 'price',
                 min_slopes_per_x=5,
                 min_samples_leaf=10,
                 show_slope_lines=True)

# rent_pdp()
plt.savefig("/Users/parrt/Desktop/james.png", dpi=200)
plt.show()