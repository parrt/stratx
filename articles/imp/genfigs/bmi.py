from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from support import *
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

# I = importances(X, y, n_trials=20)
# print(I)

R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=300,
                         kfolds=3,
                         imp_n_trials=10,
                         model="RF",
                         stratpd_min_samples_leaf=10,
                         # min_slopes_per_x=8,
                         top_features_range=(1, 8),
                         #include=['StratImpact']
                         drop=['Spearman','PCA']
                         )

print(R)

plot_topk(R, k=8, title=f"RF bmi",
          ylabel="20% 5-fold CV MAE ($)",
          title_fontsize=14,
          label_fontsize=14,
          ticklabel_fontsize=10)
plt.tight_layout()
plt.savefig(f"../images/bmi-topk.pdf", bbox_inches="tight", pad_inches=0)
plt.show()