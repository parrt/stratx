import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

from stratx import featimp
from stratx.partdep import *
from rfpimp import plot_importances
import rfpimp

import shap

palette = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928"
]
def plot_all_PD(X, y, eqn=None, min_samples_leaf=5, min_clip=False, max_features=1, ntrees=5):
    fig, ax = plt.subplots(1,1)
    for i,colname in enumerate(X.columns):
        leaf_xranges, leaf_slopes, pdpx, pdpy, ignored = \
            plot_stratpd(X, y, colname, 'y', ax=ax,
                         show_slope_lines=False,
                         # min_samples_leaf=min_samples_leaf,
                         bootstrap=False,
                         max_features=max_features,
                         ntrees=ntrees,
                         pdp_marker_size=2,
                     pdp_marker_color=palette[i])
        plt.text(np.max(pdpx), np.max(pdpy), colname)
    ax.set_xlabel(','.join(X.columns))

    if eqn is not None:
        plt.title(f"${eqn}$")
    plt.tight_layout()
    # ax.legend()
    plt.show()


def synthetic_poly2dup_data(n, p):
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    # coeff = np.random.random_sample(size=p)*10 # get p random coefficients
    coeff = np.array(np.arange(p)*3+1)
    print("coeff", coeff)
    for i in range(p):
        df[f'x{i+1}'] = np.round(np.random.random_sample(size=n)*10,1)
    df['x3'] = df['x1']# + np.random.random_sample(size=n)  # copy x1 into x3
    # multiply coefficients x each column (var) and sum along columns
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 )
    #df['y'] = 5*df['x1'] + 3*df['x2'] + df['x3']**2
    #TODO add noise
    eqn = ' '.join(f"{coeff[i]} x_{i+1}" for i in range(p))
    return df, coeff, eqn + " where x_3 = x_1"

"""
It looks like I need to subtract out the y-intercept and not the average. 
We are really trying to normalize the Y so that it starts at 0, just like the
PDP y's.  Summing the PDP abs(y-yintercept) gives almost exactly mean(y)-yintercept.
"""

df, coeff, eqn = synthetic_poly2dup_data(1000,6)
print(df.head(3))

X = df.drop('y', axis=1)
y = df['y']
#X = featimp.standardize(X)

rf = RandomForestRegressor(n_estimators=10)
rf.fit(X,y)

max_features = 1.0
ntrees = 5

I, y_mass, pdpy_mass = featimp.importances(X, y,
                                           bootstrap=False,
                                           max_features=max_features,
                                           ntrees=ntrees,
                                           )
plot_importances(I, imp_range=(0, 1))
plt.show()

df_ = df.sort_values(by=['x2'], ascending=True)
#print(df_.head(5))
X = df_.drop('y', axis=1)
y = df_['y']

plot_all_PD(X, y, eqn, min_clip=True, max_features=max_features, ntrees=ntrees)
