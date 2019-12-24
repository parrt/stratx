from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from impimp import *
from stratx import *
from support import *

import shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

X, y = load_rent(n=30_000)

# importances_pvalues(X, y, n_trials=50)
I = impact_importances(X, y, pvalues=True, n_pvalue_trials=80, n_jobs=4)
# I = impact_importances(X, y, stddev=True, n_stddev_trials=5, pvalues=True, n_pvalue_trials=5)
print(I)