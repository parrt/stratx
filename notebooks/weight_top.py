from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rfpimp import plot_importances, dropcol_importances, importances

def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n // 2
    nwomen = n // 2
    df['sex'] = ['M'] * nmen + ['F'] * nwomen
    df.loc[df['sex'] == 'F', 'pregnant'] = np.random.randint(0, 2, size=(nwomen,))
    df.loc[df['sex'] == 'M', 'pregnant'] = 0
    df.loc[df['sex'] == 'M', 'height'] = 5 * 12 + 8 + np.random.uniform(-7, +8,
                                                                        size=(nmen,))
    df.loc[df['sex'] == 'F', 'height'] = 5 * 12 + 5 + np.random.uniform(-4.5, +5,
                                                                        size=(nwomen,))
    df.loc[df['sex'] == 'M', 'education'] = 10 + np.random.randint(0, 8, size=nmen)
    df.loc[df['sex'] == 'F', 'education'] = 12 + np.random.randint(0, 8, size=nwomen)
    df['weight'] = 120 \
                   + (df['height'] - df['height'].min()) * 10 \
                   + df['pregnant'] * 30 \
                   - df['education'] * 1.5
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    eqn = "y = 120 + 10(x_{height} - min(x_{height})) + 30x_{pregnant} - 1.5x_{education}"

    df['pregnant'] = df['pregnant'].astype(int)
    df['sex'] = df['sex'].map({'M': 0, 'F': 1}).astype(int)
    X = df.drop('weight', axis=1)
    y = df['weight']

    return X, y, df, eqn


n = 1500
X, y, df, eqn = toy_weight_data(n=n)

R = compare_top_features(X, y, n_shap=n//3, min_samples_leaf=10, min_slopes_per_x=n*3.5/1000,
                         n_estimators=40,
                         catcolnames={'sex', 'pregnant'},
                         metric=mean_squared_error)

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)
print(R)

fig, ax = plt.subplots(1,1,figsize=(6,3.5))
plot_topN(R, ax)
# ax.set_ylabel("OOB $R^2$")
# ax.set_ylabel("Training RMSE (dollars)")
ax.set_ylabel("Training MAE (pounds)")
plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/weight-RMSE.png", dpi=150)
plt.savefig("/Users/parrt/Desktop/weight-MAE.png", dpi=150)
# plt.savefig("/Users/parrt/Desktop/weight-OOB.png", dpi=150)
plt.title("Weight")
plt.show()