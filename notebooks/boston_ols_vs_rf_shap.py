from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import shap

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rfpimp import plot_importances, dropcol_importances, importances

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)
n = X.shape[0]
n_shap=n

lm = LinearRegression()
#lm = Lasso(alpha=.001)
lm.fit(X, y)
#X_ = pd.DataFrame(normalize(X), columns=X.columns)
ols_I, score = linear_model_importance(lm, X, y)
ols_shap_I = shap_importances(lm, X, n_shap=n_shap)  # fast enough so use all data
# print(ols_shap_I)
lm_score = lm.score(X,y)
print("OLS", lm_score, mean_absolute_error(y, lm.predict(X)))

rf = RandomForestRegressor(n_estimators=30, oob_score=True)
rf.fit(X, y)
rf_I = shap_importances(rf, X, n_shap)
# print(rf_I)
rf_score = rf.score(X, y)
print("RF", rf_score, rf.oob_score_, mean_absolute_error(y, rf.predict(X)))

b = xgb.XGBRegressor(max_depth=3, eta=.01, n_estimators=50)
b.fit(X, y)
m_I = shap_importances(b, X, n_shap)
# print(m_I)
b_score = b.score(X, y)
print("XGBRegressor", b_score, mean_absolute_error(y, b.predict(X)))

fig, axes = plt.subplots(1,3,figsize=(10,2.0))

plot_importances(ols_shap_I.iloc[:6], ax=axes[0], imp_range=(0,.5))
axes[0].set_title(f"OLS training $R^2$={lm_score:.2f}")
plot_importances(rf_I.iloc[:6], ax=axes[1], imp_range=(0,.5))
axes[1].set_title(f"RF training $R^2$={rf_score:.2f}")
plot_importances(m_I.iloc[:6], ax=axes[2], imp_range=(0,.5))
axes[2].set_title(f"XGBoost training $R^2$={b_score:.2f}")

plt.suptitle(f"SHAP importances for Boston dataset: {n:,d} records, {n_shap} SHAP test records")
plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/diff-models.png", bbox_inches=0, dpi=150)
plt.show()

