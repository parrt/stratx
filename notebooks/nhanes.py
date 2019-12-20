import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 80)

import shap

from stratx.partdep import *
from impimp import *
from support import *

def fix_missing_num(df, colname):
    df[colname+'_na'] = pd.isnull(df[colname]).astype(int)
    df[colname].fillna(df[colname].median(), inplace=True)

# From https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
def c_statistic_harrell(pred, labels):
    total = 0
    matches = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[j] > 0 and abs(labels[i]) > labels[j]:
                total += 1
                if pred[j] > pred[i]:
                    matches += 1
    return matches/total

X,y = shap.datasets.nhanesi()
X = X.drop('Unnamed: 0', axis=1)
X['Race'] = X['Race'].astype(int)
X['Sex'] = X['Sex'].astype(int)
n = X.shape[0]

shap_paper_features = ['Age', 'Sex', 'Poverty index', 'Systolic BP', 'Serum Cholesterol',
                       'Pulse pressure', 'BMI', 'White blood cells',
                       'Sedimentation rate', 'TS', 'Serum Magnesium',
                       'Serum Protein', 'Serum Iron',
                       'Red blood cells', 'Race']

X = X[shap_paper_features].copy()

for feature in ['Sedimentation rate', 'Systolic BP', 'White blood cells', 'Pulse pressure']:
    fix_missing_num(X, feature)

rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
rf.fit(X,y)
print("OOB", rf.oob_score_)

R = compare_top_features(X, pd.Series(y), n_shap=300, min_samples_leaf=15,
                         min_slopes_percentile_x=0.001,
                         catcolnames={'Sex', 'Race'})
print(R)

# From https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
# see how well we can order people by survival
#c_statistic_harrell(rf.predict(X), y)

shap_test_size = 200
# shap_values = shap.TreeExplainer(rf).shap_values(X[:shap_test_size])
#
# shap.summary_plot(shap_values, X[:shap_test_size])

# I = impact_importances(X, pd.Series(y), min_samples_leaf=10,
#                             catcolnames={'Sex','Race'},
#                             min_slopes_percentile_x=0.01)
# plot_importances(I)
#
# min_samples_leaf = 5
# min_slopes_percentile_x = 0.001
# plot_stratpd(X, pd.Series(y), 'TS', 'y',
#              show_slope_counts=True,
#              min_slopes_percentile_x=min_slopes_percentile_x,
#              min_samples_leaf=min_samples_leaf,
#              show_slope_lines=False)
#
# plt.title(f"min_slopes_percentile_x={min_slopes_percentile_x}, min_samples_leaf={min_samples_leaf}")

# plt.savefig("/Users/parrt/Desktop/nhanes-MAE.svg", dpi=150)

plt.show()