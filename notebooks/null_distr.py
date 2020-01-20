from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from stratx.featimp import *
from stratx import *
from support import *

import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# X, y = load_rent(n=30_000)

n = 10_000

if True:
    X, y = load_bulldozer()

    # Most recent timeseries data is more relevant so get big recent chunk
    # then we can sample from that to get n
    X = X.iloc[-50_000:]
    y = y.iloc[-50_000:]

    idxs = resample(range(50_000), n_samples=n, replace=False)
    X, y = X.iloc[idxs], y.iloc[idxs]

# X, y, _ = load_flights(n=n)

# importances_pvalues(X, y, n_trials=50)
I = importances(X, y, n_trials=5, min_samples_leaf=10, pvalues=True, n_pvalue_trials=50, n_jobs=1,
                catcolnames={'AC','ModelID'}, normalize=False)
# I = impact_importances(X, y, stddev=True, n_stddev_trials=5, pvalues=True, n_pvalue_trials=5)
print(I)