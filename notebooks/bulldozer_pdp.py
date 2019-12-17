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

plot_catstratpd(X, y, 'ModelID', 'SalePrice', min_samples_leaf=5, sort=None)

# plot_stratpd_gridsearch(X, y, 'Wvillage', 'price')
plt.tight_layout()
# rent_pdp()
plt.savefig("/Users/parrt/Desktop/james.png", pad_inches=0, dpi=150)
plt.show()