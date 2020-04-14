
from stratx.featimp import importances, plot_importances
from support import tune_SVM, tune_RF, tune_XGBoost, shap_importances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import tempfile
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from sklearn import svm

np.random.seed(44) # choose a seed that demonstrates diff RF/GBM importances

TUNE_XGB = False
TUNE_SVM = False

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

n = X.shape[0]
n_shap=len(X)

fig, axes = plt.subplots(1,4,figsize=(10,2.5))

lm = LinearRegression()
X_ = StandardScaler().fit_transform(X)
X_ = pd.DataFrame(X_, columns=X.columns)
lm.fit(X_, y)
lm_score = lm.score(X_,y)
print("OLS", lm_score, mean_absolute_error(y, lm.predict(X_)))
ols_shap_I = shap_importances(lm, X_, X_, n_shap=n_shap)  # fast enough so use all data
print(ols_shap_I)

"""
Uncomment to compute variance of RF SHAP
n_rf_trials = 15
all_rf_I = np.empty(shape=(n_rf_trials,X.shape[1]))
for i in range(n_rf_trials):
    rf = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, oob_score=True)
    rf.fit(X, y)
    rf_I = shap_importances(rf, X, X, n_shap, sort=False)
    all_rf_I[i,:] = rf_I['Importance'].values
    rf_score = rf.score(X, y)
    print("RF", rf_score, rf.oob_score_, mean_absolute_error(y, rf.predict(X)))

rf_I = pd.DataFrame(data={'Feature': X.columns,
                           'Importance': np.mean(all_rf_I, axis=0),
                           'Sigma': np.std(all_rf_I, axis=0)})
rf_I = rf_I.set_index('Feature')
rf_I = rf_I.sort_values('Importance', ascending=False)
rf_I.reset_index().to_feather("/tmp/t.feather")
# print(rf_I)
"""

rf = RandomForestRegressor(n_estimators=30, oob_score=True)
rf.fit(X, y)
rf_I = shap_importances(rf, X, X, n_shap)
rf_score = rf.score(X, y)
print("RF", rf_score, rf.oob_score_, mean_absolute_error(y, rf.predict(X)))

if TUNE_XGB:
    b, bestparams = tune_XGBoost(X, y)
    # XGB best: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300}
    # XGB training R^2 0.9925102435056687
    # SHAP time for 10 test records using XGBRegressor = 0.2s
    # XGBRegressor 0.9925102435056687 0.6102726817602225
else:
    b = xgb.XGBRegressor(n_estimators=300,
                         max_depth=3,
                         learning_rate=.1,
                         verbose=2,
                         n_jobs=8)

b.fit(X, y)
xgb_score = b.score(X, y)
print("XGB training R^2", xgb_score)
m_I = shap_importances(b, X, X, n_shap)
b_score = b.score(X, y)
print("XGBRegressor", b_score, mean_absolute_error(y, b.predict(X)))

X_ = StandardScaler().fit_transform(X)
X_ = pd.DataFrame(X_, columns=X.columns)

if TUNE_SVM:
    s, bestparams = tune_SVM(X_, y)
    # SVM best: {'C': 5000, 'gamma': 0.001, 'kernel': 'rbf'}
    # svm_score 0.8591641343478678
else:
    s = svm.SVR(kernel='rbf', gamma=0.001, C=5000.)

s.fit(X_, y)
svm_score = s.score(X_, y)
print("svm_score", svm_score)
svm_shap_I = shap_importances(s, X_, X_, n_shap=n_shap)
"""
Takes 13 minutes for all records
100%|██████████| 506/506 [13:30<00:00,  1.60s/it]
SHAP time for 506 test records using SVR = 810.1s
"""

plot_importances(ols_shap_I.iloc[:8], ax=axes[0], imp_range=(0,.4), width=2.5, xlabel='(a)')
axes[0].set_title(f"OLS train $R^2$={lm_score:.2f}")
plot_importances(rf_I.iloc[:8], ax=axes[1], imp_range=(0,.4), width=2.5, xlabel='(b)')
axes[1].set_title(f"RF train $R^2$={rf_score:.2f}")
plot_importances(m_I.iloc[:8], ax=axes[2], imp_range=(0,.4), width=2.5, xlabel='(c)')
axes[2].set_title(f"XGBoost train $R^2$={b_score:.2f}")
plot_importances(svm_shap_I.iloc[:8], ax=axes[3], imp_range=(0,.4), width=2.5, xlabel='(d)')
axes[3].set_title(f"SVM train $R^2$={svm_score:.2f}")

plt.suptitle(f"SHAP importances for Boston dataset: {n:,d} records, {n_shap} SHAP test records")
plt.tight_layout()
plt.savefig("../images/diff-models.pdf",
            bbox_inches="tight", pad_inches=0)
plt.show()

