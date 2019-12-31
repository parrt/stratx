from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import shap

from stratx.featimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rfpimp import plot_importances, dropcol_importances, importances

n = 20_000  # shap crashes above this; 20k works
n_shap=1000

X, y = load_bulldozer()

X = X.iloc[-n:]
y = y.iloc[-n:]

lm = LinearRegression()
lm.fit(X, y)
X_ = data = StandardScaler().fit_transform(X)
X_ = pd.DataFrame(X_, columns=X.columns)
ols_I, score = linear_model_importance(lm, X_, y)
ols_shap_I = shap_importances(lm, X_, n_shap=n_shap)  # fast enough so use all data
print(ols_shap_I)

rf = RandomForestRegressor(n_estimators=30, oob_score=True)
rf.fit(X, y)
rf_I = shap_importances(rf, X, n_shap)
print(rf_I)

m = xgb.XGBRegressor(max_depth=3, eta=.02)
m.fit(X, y)
m_I = shap_importances(m, X, n_shap)
print(m_I)

fig, axes = plt.subplots(1,3,figsize=(10,2.0))

plot_importances(ols_shap_I.iloc[:6], ax=axes[0], imp_range=(0,.5))
plot_importances(rf_I.iloc[:6], ax=axes[1], imp_range=(0,.5))
plot_importances(m_I.iloc[:6], ax=axes[2], imp_range=(0,.5))

plt.suptitle(f"SHAP importances for bulldozer dataset: {n:,d} records, {n_shap} SHAP test records")
plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/diff-models.png", bbox_inches=0, dpi=150)
plt.show()

