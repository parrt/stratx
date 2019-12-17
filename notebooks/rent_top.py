from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

n = 10_000
X, y = load_rent(n=n)

R = compare_top_features(X, y, n_shap=300, min_samples_leaf=10, min_slopes_per_x=n*3.5/1000,
                         top_features_range=(1,9))

print(R)
