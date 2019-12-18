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

n = 20_000

X, y = load_bulldozer()

X = X.iloc[-n:]
y = y.iloc[-n:]

R = compare_top_features(X, y, n_shap=500, min_samples_leaf=10, min_slopes_per_x=n*3/1000,
                         catcolnames={'AC','ModelID'},
                         use_oob=True)
print(R)
