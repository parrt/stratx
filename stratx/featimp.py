import numpy as np
import pandas as pd
from typing import Sequence

from sklearn.ensemble import RandomForestRegressor

from stratx.partdep import PD, discrete_xc_space

def standardize(df):
    # standardize variables
    Z = df.copy()
    for colname in df.columns:
        Z[colname] = (Z[colname] - np.mean(Z[colname])) / np.std(Z[colname])
    return Z


def importances(X:pd.DataFrame, y:pd.Series, colnames:Sequence=None,
                ntrees=1, min_samples_leaf=10, bootstrap=False,
                max_features=1.0,
                verbose=False):
    """
    """
    if colnames is None:
        colnames = X.columns.values

    # standardize variables
    Z = X.copy()
    for colname in colnames:
        Z[colname] = (Z[colname] - np.mean(Z[colname])) / np.std(Z[colname])

    for colname in colnames:
        leaf_xranges, leaf_slopes, pdpx, pdpy, ignored = \
            PD(X=Z, y=y, colname=colname, ntrees=ntrees, min_samples_leaf=min_samples_leaf,
               bootstrap=bootstrap, max_features=max_features, supervised=True,
               verbose=verbose)
        print(pdpx)
        print(pdpy)

    # marginal_xranges, marginal_sizes, marginal_slopes, ignored = \
    #     discrete_xc_space(X[colname], y, verbose=verbose)

