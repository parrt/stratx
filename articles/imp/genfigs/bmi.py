from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from stratx.featimp import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=300, threshold=2000)

# data from http://wiki.stat.ucla.edu/socr/index.php/SOCR_Data_BMI_Regression
df = pd.read_csv("data/bmi.csv")
response = 'Density'
response = "BodyFat"
X, y = df.drop(['Density','BodyFat'], axis=1), df[response]
rf = RandomForestRegressor(n_estimators=30, oob_score=True)
rf.fit(X,y)
print(f"Response {response} = OOB R^2 {rf.oob_score_:.2f}")
I = importances(X, y, n_trials=20)
print(I)