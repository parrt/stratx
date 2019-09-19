import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from stratx.partdep import PD, discrete_xc_space

def importances(X, y, colname,
                ntrees=1, min_samples_leaf=10, bootstrap=False,
                max_features=1.0,
                verbose=False):
    """
    """
    leaf_xranges, leaf_slopes, pdpx, pdpy, ignored = \
        PD(X=X, y=y, colname=colname, ntrees=ntrees, min_samples_leaf=min_samples_leaf,
           bootstrap=bootstrap, max_features=max_features, supervised=True,
           verbose=verbose)

    marginal_xranges, marginal_sizes, marginal_slopes, ignored = \
        discrete_xc_space(X[colname], y, verbose=verbose)

    print(pdpx)
    print(pdpy)
    print(marginal_slopes)
