from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from impimp import *
from stratx import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

def synthetic_poly_dup_data(n):
    p = 3 # x1, x2, x3
    df = pd.DataFrame()
    coeff = np.array([1,1,1])
    for i in range(p):
        df[f'x{i + 1}'] = np.random.random_sample(size=n) * 10
    df['x3'] = df['x1'] + np.random.random_sample(size=n)#-0.5 # copy x1 into x3 with noise
    yintercept = 100
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " \\,\\,where\\,\\, x_3 = x_1 + noise"
    return df, coeff, eqn

def dupcol():
    df, coeff, eqn = synthetic_poly_dup_data(1000)
    X = df.drop('y', axis=1)
    y = df['y']
    print(X.head())

    print(eqn)

    ntrials=3
    fig, axes = plt.subplots(ntrials, 2, figsize=(4.5, ntrials))

    for i in range(ntrials):
        I = impact_importances(X, y, min_samples_leaf=5, pdp='stratpd')
        plot_importances(I, imp_range=(0, 1.0), ax=axes[i][0])
        axes[i][0].set_title(f"StratPD impact", fontsize=8)
        I = impact_importances(X, y, pdp='ice')
        plot_importances(I, imp_range=(0, 1.0), ax=axes[i][1])
        axes[i][1].set_title(f"ICE/PDP impact", fontsize=8)
        # ntrees=5, min_samples_leaf=10, bootstrap=False, max_features=1)
        print(I)
    plt.suptitle('$'+eqn+'$', y=1.0)


df, coeff, eqn = synthetic_poly_dup_data(1000)
X = df.drop('y', axis=1)
y = df['y']
leaf_xranges, leaf_slopes, slope_counts_at_x, pdpx, pdpy, ignored = \
    plot_stratpd(X, y, colname='x3', targetname='y', min_samples_leaf=5)

print("ignored", ignored)
print(X.sort_values('x1').iloc[:8,:])
r = pd.DataFrame()
r['pdpx'] = pdpx
r['pdpy'] = pdpy

print(r.head(10))

s = pd.DataFrame()
s['left'] = leaf_xranges[:,0]
s['right'] = leaf_xranges[:,1]
s['slope'] = leaf_slopes

print(s.head(10))
#dupcol()
# plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/polydup_strat_vs_ice.pdf", bbox_inches=0)
plt.show()