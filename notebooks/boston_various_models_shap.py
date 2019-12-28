from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn import svm

import shap

from stratx.featimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(44) # choose a seed that demonstrates diff RF/GBM importances

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

# X = X[:100]
# y = y[:100]

n = X.shape[0]
n_shap=n

fig, axes = plt.subplots(1,4,figsize=(12,2.5))

lm = LinearRegression()
#lm = Lasso(alpha=.001)
lm.fit(X, y)
ols_I, score = linear_model_importance(lm, X, y)
ols_shap_I = shap_importances(lm, X, n_shap=n_shap)  # fast enough so use all data
# print(ols_shap_I)
lm_score = lm.score(X,y)
print("OLS", lm_score, mean_absolute_error(y, lm.predict(X)))

rf = RandomForestRegressor(n_estimators=30, oob_score=True)
rf.fit(X, y)
rf_I = shap_importances(rf, X, n_shap)
rf_score = rf.score(X, y)
print("RF", rf_score, rf.oob_score_, mean_absolute_error(y, rf.predict(X)))

b = xgb.XGBRegressor(max_depth=5, eta=.01, n_estimators=50)
b.fit(X, y)
m_I = shap_importances(b, X, n_shap)
b_score = b.score(X, y)
print("XGBRegressor", b_score, mean_absolute_error(y, b.predict(X)))

# model = GridSearchCV(svm.SVR(), cv=5,
#                      param_grid={"C": [1, 1000, 2000, 3000, 5000],
#                                "gamma": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]})
# model.fit(X, y)
# svr = model.best_estimator_
# print("SVM best:",model.best_params_)
s = svm.SVR(gamma=0.001, C=100.)
s.fit(X, y)
# print(model.best_params_)
svm_score = s.score(X, y)
print("svm_score", svm_score)
svm_shap_I = shap_importances(s, X, n_shap=n_shap)  # fast enough so use all data

plot_importances(ols_shap_I.iloc[:8], ax=axes[0], imp_range=(0,.4))
axes[0].set_title(f"OLS training $R^2$={lm_score:.2f}")
plot_importances(rf_I.iloc[:8], ax=axes[1], imp_range=(0,.4))
axes[1].set_title(f"RF training $R^2$={rf_score:.2f}")
plot_importances(m_I.iloc[:8], ax=axes[2], imp_range=(0,.4))
axes[2].set_title(f"XGBoost training $R^2$={b_score:.2f}")
plot_importances(svm_shap_I.iloc[:8], ax=axes[3], imp_range=(0,.4))
axes[3].set_title(f"SVM training $R^2$={svm_score:.2f}")

plt.suptitle(f"SHAP importances for Boston dataset: {n:,d} records, {n_shap} SHAP test records")
plt.tight_layout()
plt.savefig("/Users/parrt/github/stratx/articles/imp/images/diff-models.pdf", bbox_inches=0)
plt.show()

