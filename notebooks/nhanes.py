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


def compare():
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    rf.fit(X,y)
    print("OOB", rf.oob_score_)

    I = impact_importances(X, pd.Series(y), min_samples_leaf=40,
                           n_jobs=-1,
                           catcolnames={'Sex', 'Race'})
    plot_importances(I)
    #
    # shap_test_size = 200
    # R = compare_top_features(X, pd.Series(y), n_shap=shap_test_size,
    #                          min_samples_leaf=30,
    #                          min_slopes_per_x=15,
    #                          catcolnames={'Sex', 'Race'},
    #                          top_features_range=(1, 3))
    # print(R)


def harrell(shap_test_size=200):
    # From https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
    # see how well we can order people by survival
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    rf.fit(X,y)
    print("OOB", rf.oob_score_)
    c_statistic_harrell(rf.predict(X), y)

    shap_values = shap.TreeExplainer(rf).shap_values(X[:shap_test_size])

    shap.summary_plot(shap_values, X[:shap_test_size])


def examine_sex_feature(shap_test_size=200):
    # rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    # rf.fit(X,y)
    # print("OOB", rf.oob_score_)
    #
    # explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100), feature_perturbation='interventional')
    # shap_values = explainer.shap_values(X[:shap_test_size], check_additivity=False)
    # shapimp = np.mean(np.abs(shap_values), axis=0)
    # print("\nRF SHAP importances", list(shapimp))
    #shap.summary_plot(shap_values, X[:shap_test_size])
    # shap.dependence_plot("Sex", shap_values, X[:shap_test_size], interaction_index=None)
    # plot_catstratpd(X, pd.Series(y), 'Sex', 'y', yrange=(-3,5), min_y_shifted_to_zero=False,
    #                 sort=None)
    I = impact_importances(X, pd.Series(y), catcolnames={'Sex','Race'},
                           min_samples_leaf=40,
                           normalize=False)
    print(I)


def examine_Poverty_index_feature(shap_test_size=200):
    # rf = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)
    # rf.fit(X,y)
    # print("OOB", rf.oob_score_)
    #
    # explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100), feature_perturbation='interventional')
    # shap_values = explainer.shap_values(X[:shap_test_size], check_additivity=False)
    # shap.dependence_plot("Poverty index", shap_values, X[:shap_test_size], interaction_index=None)
    plot_stratpd_gridsearch(X, pd.Series(y), 'Poverty index', 'y',
                            min_samples_leaf_values=(20,30,40))
    # plot_stratpd(X, pd.Series(y), 'Poverty index', 'y', min_samples_leaf=40,
    #              min_slopes_per_x=15,
    #              show_slope_lines=False,
    #              show_mean_line=False)

# min_samples_leaf = 10
# min_slopes_per_x = 0
#
# plot_stratpd_gridsearch(X, pd.Series(y), 'Age', 'y',
#                         min_samples_leaf_values=(20,40,60,70,80,90,100),
#                         min_slopes_per_x_values=(0,))

# plot_stratpd(X, pd.Series(y), 'Age', 'y',
#              show_slope_counts=True,
#              min_slopes_per_x=min_slopes_per_x,
#              min_samples_leaf=min_samples_leaf,
#              show_slope_lines=False)

#plt.title(f"min_slopes_per_x={min_slopes_per_x}, min_samples_leaf={min_samples_leaf}")

#examine_Poverty_index_feature()

#examine_sex_feature()

compare()

plt.tight_layout()

plt.savefig("/Users/parrt/Desktop/nhanes-MAE.png", dpi=150)

plt.show()