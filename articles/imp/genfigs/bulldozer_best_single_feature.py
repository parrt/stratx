from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

from support import *
from stratx.featimp import *
from stratx.partdep import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=300)#, threshold=1e10)


n=25_000
X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

idxs = resample(range(50_000), n_samples=n, replace=False)
X, y = X.iloc[idxs], y.iloc[idxs]

col = 'DEPARTURE_TIME'
col = 'SCHEDULED_DEPARTURE'
col = 'FLIGHT_NUMBER'
col = 'TAXI_OUT'


df = best_single_feature(X, y, kfolds=5, model='RF')
print(df)

print("MAE predicting mean(y) always", mean_absolute_error(y, [np.mean(y)]*len(y)))
print("MSE predicting mean(y) always", mean_squared_error(y, [np.mean(y)]*len(y)))
