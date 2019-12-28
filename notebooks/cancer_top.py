from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
    load_breast_cancer, load_diabetes, fetch_mldata

from stratx.featimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

X, y = load_cancer_regr()
n = X.shape[0]
use_oob=False
metric = mean_absolute_error

R = compare_top_features(X, y, n_shap=n, min_samples_leaf=5,
                         metric=metric,
                         use_oob=use_oob,
                         min_slopes_per_x=0)
print(R)