from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

from stratx.partdep import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True, linewidth=300, threshold=1e10)

#np.random.seed(999)

def synthetic_poly_data(n=1000,max_x=1000,p=2,dtype=float):
    df = pd.DataFrame()
    for i in range(p):
        df[f'x{i + 1}'] = (np.random.random_sample(size=n) * max_x).astype(dtype)
    yintercept = 100
    df['y'] = np.sum(df, axis=1) + yintercept
    terms = [f"x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " where x_i ~ U(0,10)"
    return df, eqn

def synthetic_poly_data_gaussian(n=1000,max_x=1000,p=2,dtype=float):
    df = pd.DataFrame()
    for i in range(p):
        # shift mean up to 5 so it's all positive
        df[f'x{i + 1}'] = ((np.random.standard_normal(size=n)+5) * max_x).astype(dtype)
    yintercept = 100
    df['y'] = np.sum(df, axis=1) + yintercept
    terms = [f"x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " where x_i ~ U(0,10)"
    return df, eqn

# df, eqn = synthetic_poly_data(500, p=2, dtype=int)
# X = df.drop('y', axis=1)
# y = df['y']
#
# plot_stratpd(X, y, colname='x1', targetname='y', n_trials=1,
#              min_slopes_per_x=1,
#              impact_marker_size=.1,
#              show_impact=True,
#              show_slope_lines=False)
# plt.savefig(f"/Users/parrt/Desktop/linear-numeric.pdf", pad_inches=0)
# plt.show()


df, eqn = synthetic_poly_data_gaussian(1000, p=2, max_x=10, dtype=int)
# df, eqn = synthetic_poly_data(1000, p=2, max_x=10, dtype=int)
X = df.drop('y', axis=1)
y = df['y']
uniq_catcodes, avg_per_cat, ignored = \
    plot_catstratpd(X, y, colname='x1', targetname='y',
                    n_trials=1,
                    min_samples_leaf=10,
                    show_x_counts=True,
                    show_xticks=True,
                    show_impact=True,
                    min_y_shifted_to_zero=True,
                    verbose=True,
                    # yrange=(-1000,1000)
                    )

print("ignored",ignored)
print("avg pdp", np.nanmean(avg_per_cat), "std pdp", np.nanstd(avg_per_cat))
plt.savefig(f"/Users/parrt/Desktop/linear-catstrat.pdf", pad_inches=0)
plt.show()